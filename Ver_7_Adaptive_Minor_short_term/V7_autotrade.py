import os
from dotenv import load_dotenv
load_dotenv()

import pyupbit
import pandas as pd
import pandas_ta as ta
import json
from openai import OpenAI
import schedule
import time
import requests
from datetime import datetime
import sqlite3
import logging
import statistics

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    ElementClickInterceptedException,
    WebDriverException,
    NoSuchElementException,
)

from PIL import Image
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))

# 캡처를 위한 크롬 드라이버 생성
def create_driver():
    env = os.getenv("ENVIRONMENT")
    logger.info("ChromeDriver 설정 중...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    try:
        if env == "local":
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
        elif env == "ec2":
            service = Service('/usr/bin/chromedriver')
        else:
            raise ValueError(f"Unsupported environment. Only local or ec2: {env}")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        logger.error(f"ChromeDriver 생성 중 오류 발생: {e}")
        raise

# Database
def initialize_db(db_path='ripple_trading_decisions.sqlite'):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # 거래내역 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,    -- 고유 식별자
                timestamp DATETIME,                      -- 결정 시간
                decision TEXT,                           -- 결정 내용 (매수/매도/홀딩)
                percentage REAL,                         -- 매수/매도 비율(%)
                reason TEXT,                             -- 결정 이유
                xrp_balance REAL,                        -- 리플 잔고
                krw_balance REAL,                        -- 원화 잔고
                fee REAL,                                -- 거래 수수료
                settlement_amount REAL,                  -- 정산 금액
                xrp_avg_buy_price REAL,                  -- 리플 평균 매수가
                xrp_krw_price REAL,                      -- 현재 리플 시세(KRW)
                performance REAL                         -- 수익률 성과
            );
        ''')
        
        # 거래 목표 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                -- 단기 목표 관련 컬럼
                short_term_target REAL,           -- 단기 목표 가격
                short_term_stop_loss REAL,        -- 단기 손절가
                short_term_target_time TEXT,       -- 단기 목표 예상 기간 (예: '1일', '4시간')
                short_term_confidence REAL,       -- 단기 목표 신뢰도(1-100%)
                detail_plan TEXT,                 -- 자세한 실행 계획 (추가 진입/매도 가격, 조건부 행동 등)
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        conn.commit()

# DB에 결정기록 저장 및 이전거래에 대한 수익률 저장
def save_decision_to_db(decision, current_status):
    try:
        # 입력 파라미터 검증
        if not isinstance(decision, dict):
            raise ValueError("결정은 딕셔너리 형태여야 합니다")
            
        required_keys = ['decision', 'percentage', 'reason']
        if not all(key in decision for key in required_keys):
            raise ValueError(f"결정에 필수 키가 누락되었습니다: {required_keys}")

        db_path = 'ripple_trading_decisions.sqlite'
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. 현재 상태 파싱 및 검증
            try:
                if not current_status:
                    raise ValueError("현재 상태가 비어있습니다")
                    
                status_dict = json.loads(current_status)
                if not isinstance(status_dict, dict):
                    raise ValueError("잘못된 상태 형식입니다")
                    
                # 현재 가격 가져오기 (최대 3번 재시도)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        orderbook = pyupbit.get_orderbook(ticker="KRW-XRP")
                        if not orderbook or 'orderbook_units' not in orderbook:
                            raise ValueError("잘못된 오더북 데이터")
                        current_price = float(orderbook['orderbook_units'][0]["ask_price"])
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(1)
                        
            except Exception as e:
                print(f"현재 상태 파싱 또는 현재 가격 획득 중 오류: {e}")
                raise

            # 2. 새 결정 저장 (성과는 0으로 초기화)
            try:
                # 타임스탬프 포맷
                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"새 결정을 저장합니다: {current_timestamp}")
                
                # 데이터 준비 및 검증
                xrp_balance = float(status_dict.get('xrp_balance', 0))
                krw_balance = float(status_dict.get('krw_balance', 0))
                xrp_avg_buy_price = float(status_dict.get('xrp_avg_buy_price', 0))
                
                # 수수료 및 정산 금액
                fee = float(decision.get('fee', 0))
                settlement_amount = float(decision.get('settlement_amount', 0))
                
                # 새 결정 삽입 (성과는 0으로 초기화)
                cursor.execute('''
                    INSERT INTO decisions (
                        timestamp, decision, percentage, reason, xrp_balance, krw_balance, 
                        fee, settlement_amount, xrp_avg_buy_price, xrp_krw_price, 
                        performance
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ''', (
                    current_timestamp,
                    decision.get('decision'),
                    float(decision.get('percentage', 100)),
                    decision.get('reason', ''),
                    xrp_balance,
                    krw_balance,
                    fee,
                    settlement_amount,
                    xrp_avg_buy_price,
                    current_price
                ))
                
                # 3. 목표 가격 정보 저장
                if 'short_term_target' in decision:
                    short_term = decision.get('short_term_target', {})
                    
                    cursor.execute('''
                        INSERT INTO decision_targets (
                            short_term_target, short_term_stop_loss, short_term_target_time, short_term_confidence,
                            detail_plan, last_updated
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        float(short_term.get('price', 0)),
                        float(short_term.get('stop_loss', 0)),
                        short_term.get('target_time', ''),
                        float(short_term.get('confidence', 0)),
                        short_term.get('detail_plan', ''),
                        current_timestamp
                    ))
                    
                    print(f"목표 가격 정보를 성공적으로 저장했습니다.")
                
                conn.commit()
                print(f"새 결정을 성공적으로 저장했습니다: {decision.get('decision')}")
                
            except Exception as e:
                print(f"새 결정 저장 중 오류: {e}")
                conn.rollback()
                raise
                
    except Exception as e:
        print(f"save_decision_to_db 함수에서 치명적 오류: {e}")
        raise
    
    finally:
        if 'conn' in locals():
            conn.close()

# RSI 계산
def calculate_rsi(df, periods=14):
    close_delta = df['close'].diff()
    
    # 두개의 시리즈 생성: up, down
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # EWMA 계산산
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    return rsi

# Bollinger Bands 계산
def calculate_bollinger_bands(df, window=20, dev=2):
    typical_p = (df['high'] + df['low'] + df['close']) / 3
    ma = typical_p.rolling(window=window).mean()
    std = typical_p.rolling(window=window).std()
    
    upper_band = ma + (std * dev)
    lower_band = ma - (std * dev)
    
    return upper_band, ma, lower_band

# 현재 잔고 및 리플 status 가져오기
def get_current_status():
    try:
        # 기본 데이터 가져오기
        orderbook = pyupbit.get_orderbook(ticker="KRW-XRP")
        current_time = orderbook['timestamp']
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 잔고 정보 초기화 및 가져오기
        xrp_balance = 0
        krw_balance = 0
        xrp_avg_buy_price = 0
        balances = upbit.get_balances()
        for b in balances:
            if b['currency'] == "XRP":
                xrp_balance = float(b['balance'])
                xrp_avg_buy_price = float(b['avg_buy_price'])
            if b['currency'] == "KRW":
                krw_balance = float(b['balance'])

        # 여러 시간대 OHLCV 데이터 가져오기
        df_5m = pyupbit.get_ohlcv("KRW-XRP", interval="minute5", count=200)
        df_1h = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", count=200)
        df_4h = pyupbit.get_ohlcv("KRW-XRP", interval="minute240", count=200)
        
        # 5분 차트 기술적 지표 계산
        rsi_5m = calculate_rsi(df_5m, 14)
        bb_upper_5m, bb_middle_5m, bb_lower_5m = calculate_bollinger_bands(df_5m, 20, 2)
        ma_5m = df_5m['close'].rolling(window=20).mean()
        volume_sma_5m = df_5m['volume'].rolling(window=24).mean()
        current_volume_5m = df_5m['volume'].iloc[-1]
        volume_ratio_5m = current_volume_5m / volume_sma_5m.iloc[-1]

        # 1시간 차트 기술적 지표 계산
        rsi_1h = calculate_rsi(df_1h, 14)
        bb_upper_1h, bb_middle_1h, bb_lower_1h = calculate_bollinger_bands(df_1h, 20, 2)
        ma_1h = df_1h['close'].rolling(window=20).mean()
        volume_sma_1h = df_1h['volume'].rolling(window=24).mean()
        current_volume_1h = df_1h['volume'].iloc[-1]
        volume_ratio_1h = current_volume_1h / volume_sma_1h.iloc[-1]
        
        # 4시간 차트 기술적 지표 계산
        rsi_4h = calculate_rsi(df_4h, 14)
        bb_upper_4h, bb_middle_4h, bb_lower_4h = calculate_bollinger_bands(df_4h, 20, 2)
        ma_4h = df_4h['close'].rolling(window=20).mean()
        volume_sma_4h = df_4h['volume'].rolling(window=24).mean()
        current_volume_4h = df_4h['volume'].iloc[-1]
        volume_ratio_4h = current_volume_4h / volume_sma_4h.iloc[-1]

        # 현재 상태 데이터 구성
        current_status = {
            'current_datetime': current_datetime,
            'current_time': current_time,
            'orderbook': orderbook,
            'xrp_balance': xrp_balance,
            'krw_balance': krw_balance,
            'xrp_avg_buy_price': xrp_avg_buy_price,
            'technical_indicators': {
                '5m': {
                    'rsi': float(rsi_5m.iloc[-1]),
                    'bollinger_bands': {
                        'upper': float(bb_upper_5m.iloc[-1]),
                        'middle': float(bb_middle_5m.iloc[-1]),
                        'lower': float(bb_lower_5m.iloc[-1])
                    },
                    'moving_average': float(ma_5m.iloc[-1]),
                    'volume': {
                        'current': float(current_volume_5m),
                        'average_24h': float(volume_sma_5m.iloc[-1]),
                        'ratio': float(volume_ratio_5m)
                    }
                },
                '1h': {
                    'rsi': float(rsi_1h.iloc[-1]),
                    'bollinger_bands': {
                        'upper': float(bb_upper_1h.iloc[-1]),
                        'middle': float(bb_middle_1h.iloc[-1]),
                        'lower': float(bb_lower_1h.iloc[-1])
                    },
                    'moving_average': float(ma_1h.iloc[-1]),
                    'volume': {
                        'current': float(current_volume_1h),
                        'average_24h': float(volume_sma_1h.iloc[-1]),
                        'ratio': float(volume_ratio_1h)
                    }
                },
                '4h': {
                    'rsi': float(rsi_4h.iloc[-1]),
                    'bollinger_bands': {
                        'upper': float(bb_upper_4h.iloc[-1]),
                        'middle': float(bb_middle_4h.iloc[-1]),
                        'lower': float(bb_lower_4h.iloc[-1])
                    },
                    'moving_average': float(ma_4h.iloc[-1]),
                    'volume': {
                        'current': float(current_volume_4h),
                        'average_24h': float(volume_sma_4h.iloc[-1]),
                        'ratio': float(volume_ratio_4h)
                    }
                }
            }
        }
        
        return json.dumps(current_status)
    except Exception as e:
        print(f"Error in get_current_status: {e}")
        return None

# XPath로 Element 찾기
def click_element_by_xpath(driver, xpath, element_name, wait_time=10):
    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        # 요소가 뷰포트에 보일 때까지 스크롤
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        # 요소가 클릭 가능할 때까지 대기
        element = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        logger.info(f"{element_name} 클릭 완료")
        time.sleep(2)  # 클릭 후 잠시 대기
    except TimeoutException:
        logger.error(f"{element_name} 요소를 찾는 데 시간이 초과되었습니다.")
    except ElementClickInterceptedException:
        logger.error(f"{element_name} 요소를 클릭할 수 없습니다. 다른 요소에 가려져 있을 수 있습니다.")
    except NoSuchElementException:
        logger.error(f"{element_name} 요소를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"{element_name} 클릭 중 오류 발생: {e}")

# 5분차트 캡처
def perform_chart_actions_5m(driver):
    # 시간 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    # 5분 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[4]",
        "5분 옵션"
    )
    # 볼린저 밴드
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
        "볼린저 밴드 옵션"
    )
    # 이동평균선
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[59]",
        "이동평균선 옵션"
    )
    # RSI
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[81]",
        "RSI 옵션"
    )
    # MACD
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[53]",
        "MACD 옵션"
    )
    # 거래량
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[107]",
        "거래량 옵션"
    )

# 1시간 차트 캡처
def perform_chart_actions_1h(driver):
    # 시간 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    # 1시간 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]",
        "1시간 옵션"
    )
    # 볼린저 밴드
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
        "볼린저 밴드 옵션"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[1]",
        "ADX/DMS 옵션"
    )
    # 이동평균선 (20 EMA)
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[59]",
        "이동평균선 옵션"
    )
    # RSI
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[81]",
        "RSI 옵션"
    )
    # 거래량
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[107]",
        "거래량 옵션"
    )

# 4시간 차트 캡처
def perform_chart_actions_4h(driver):
    # 시간 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    # 4시간 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[9]",
        "4시간 옵션"
    )
    # 볼린저 밴드
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
        "볼린저 밴드 옵션"
    )
    # 이동평균선
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[59]",
        "이동평균선 옵션"
    )
    # RSI
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[81]",
        "RSI 옵션"
    )
    # ADX
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[1]",
        "ADX/DMS 옵션"
    )
    # 거래량
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[107]",
        "거래량 옵션"
    )

# 캡처 및 인코딩
def capture_and_encode_screenshot(driver):
    try:
        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()
        # PIL Image로 변환
        img = Image.open(io.BytesIO(png))
        # 이미지가 클 경우 리사이즈
        img.thumbnail((2000, 2000))
        # 이미지를 바이트로 변환
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        # base64로 인코딩
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_image
    except Exception as e:
        logger.error(f"스크린샷 캡처 및 인코딩 중 오류 발생: {e}")
        return None

# 캡처 진행상황 로깅
def fetch_and_prepare_data():
    driver = None
    try:
        driver = create_driver()
        images = {}
        
        # 5분 차트 캡처
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-XRP")
        logger.info("5분 차트 페이지 로드 완료")
        time.sleep(30)
        
        logger.info("5분 차트 작업 시작")
        perform_chart_actions_5m(driver)
        logger.info("5분 차트 작업 완료")
        
        images['5m'] = capture_and_encode_screenshot(driver)
        logger.info("5분 차트 스크린샷 캡처 완료")
        
        # 1시간 차트 캡처
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-XRP")
        logger.info("1시간 차트 페이지 로드 완료")
        time.sleep(30)
        
        logger.info("1시간 차트 작업 시작")
        perform_chart_actions_1h(driver)
        logger.info("1시간 차트 작업 완료")
        
        images['1h'] = capture_and_encode_screenshot(driver)
        logger.info("1시간 차트 스크린샷 캡처 완료")
        
        # 4시간 차트 캡처
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-XRP")
        logger.info("4시간 차트 페이지 로드 완료")
        time.sleep(30)
        
        logger.info("4시간 차트 작업 시작")
        perform_chart_actions_4h(driver)
        logger.info("4시간 차트 작업 완료")
        
        images['4h'] = capture_and_encode_screenshot(driver)
        logger.info("4시간 차트 스크린샷 캡처 완료")
        
        return {
            'chart_images': images,
            'numerical_data': '[]'
        }
        
    except Exception as e:
        logger.error(f"차트 데이터 준비 중 오류 발생: {e}")
        return None
    finally:
        if driver:
            driver.quit()

# 뉴스데이터 가져오기
def get_news_data():
    url = "https://serpapi.com/search.json?engine=google_news&q=xrp&api_key=" + os.getenv("SERPAPI_API_KEY")
    result = "No news data available."

    try:
        response = requests.get(url)
        news_results = response.json()['news_results']
        simplified_news = []
        
        for news_item in news_results:
            if 'stories' in news_item:
                for story in news_item['stories']:
                    timestamp = int(datetime.strptime(story['date'], '%m/%d/%Y, %H:%M %p, %z %Z').timestamp() * 1000)
                    simplified_news.append((story['title'], story.get('source', {}).get('name', 'Unknown source'), timestamp))
            else:
                if news_item.get('date'):
                    timestamp = int(datetime.strptime(news_item['date'], '%m/%d/%Y, %H:%M %p, %z %Z').timestamp() * 1000)
                    simplified_news.append((news_item['title'], news_item.get('source', {}).get('name', 'Unknown source'), timestamp))
                else:
                    simplified_news.append((news_item['title'], news_item.get('source', {}).get('name', 'Unknown source'), 'No timestamp provided'))
        result = str(simplified_news)
    except Exception as e:
        print(f"Error fetching news data: {e}")

    return result

# 공포탐욕 지수 가져오기
def fetch_fear_and_greed_index(limit=1, date_format=''):
    base_url = "https://api.alternative.me/fng/"
    params = {
        'limit': limit,
        'format': 'json',
        'date_format': date_format
    }
    response = requests.get(base_url, params=params)
    myData = response.json()['data']
    resStr = ""
    for data in myData:
        resStr += str(data)
    return resStr

# 목표 가격 정보 가져오기 > 없다면 None
def get_recent_target():
    try:
        # 데이터베이스 연결
        db_path = 'ripple_trading_decisions.sqlite'
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 가장 최근의 목표 가격 정보 조회 
            cursor.execute('''
                SELECT 
                    short_term_target, short_term_stop_loss, short_term_target_time, short_term_confidence,
                    last_updated, detail_plan
                FROM decision_targets
                ORDER BY last_updated DESC
                LIMIT 1
            ''')
            
            target_data = cursor.fetchone()
            
            if target_data:
                target_info = {
                    'short_term': {
                        'price': target_data[0],
                        'stop_loss': target_data[1],
                        'target_time': target_data[2],
                        'confidence': target_data[3],
                        'detail_plan': target_data[5] if len(target_data) > 5 else '정보 없음'
                    },
                    'last_updated': target_data[4]
                }
                return target_info
            else:
                return None
                
    except Exception as e:
        print(f"목표 정보 조회 중 오류 발생: {e}")
        return None

# GPT통한 차트분석 및 거래결정
def analyze_data_with_gpt(news_data, fear_and_greed, current_status, chart_images):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            status_data = json.loads(current_status)
            current_price = float(status_data['orderbook']['orderbook_units'][0]['ask_price'])
            xrp_balance = float(status_data['xrp_balance'])
            krw_balance = float(status_data['krw_balance'])
            xrp_value = xrp_balance * current_price

            # 평균 매수가 기준으로 수익률 계산
            profit_percentage = ((current_price - float(status_data['xrp_avg_buy_price'])) / float(status_data['xrp_avg_buy_price']) * 100) if float(status_data['xrp_avg_buy_price']) > 0 else 0
            
            # 이전 목표 가격 정보 가져오기
            recent_targets = get_recent_target()
            target_info_text = ""
            
            # 이전 목표 가격 정보가 있으면 텍스트로 변환
            if recent_targets:
                short_term = recent_targets['short_term']
                last_updated = recent_targets['last_updated']
                
                # 0으로 나누는 오류 방지를 위한 안전 장치 추가
                price_diff = short_term['price'] - current_price if short_term['price'] != 0 else 0
                price_percent_diff = (price_diff / short_term['price'] * 100) if short_term['price'] != 0 else 0
                price_percent_near = ((short_term['price'] - current_price) / short_term['price'] * 100) if short_term['price'] != 0 else 0
                
                stop_loss_diff = current_price - short_term['stop_loss'] if short_term['stop_loss'] != 0 else 0
                stop_loss_percent_diff = (stop_loss_diff / short_term['stop_loss'] * 100) if short_term['stop_loss'] != 0 else 0
                
                target_info_text = f"""
이전 설정된 목표 정보 (마지막 업데이트: {last_updated}):

1. 단기 목표:
   - 목표 가격: {short_term['price']} KRW
   - 손절가: {short_term['stop_loss']} KRW
   - 시간 프레임: {short_term['target_time']}
   - 신뢰도: {short_term['confidence']}%
   - 상세 실행 계획: {short_term['detail_plan']}

현재 목표 가격 평가:
- 현재 가격 ({current_price} KRW)과 단기 목표 가격 ({short_term['price']} KRW) 비교: 
  * 차이: {price_diff} KRW
  * 퍼센트 차이: {price_percent_diff:.2f}%
- 현재 가격 ({current_price} KRW)과 손절가 ({short_term['stop_loss']} KRW) 비교: 
  * 차이: {stop_loss_diff} KRW
  * 퍼센트 차이: {stop_loss_percent_diff:.2f}%
- 단기 목표까지 남은 예상 시간: {short_term['target_time']}
"""
            else:
                target_info_text = "이전에 설정된 목표 정보가 없습니다. 새로운 단기목표를 설정해주세요."


            system_prompt = f"""단기 암호화폐 트레이딩 전략: 발키리(V.A.L.K.R)

## 핵심 원칙:
- 당신은 암호화폐 시장에서 단기 기회를 포착하는 전략적 트레이더입니다
- 단기간(최대 6시간 이내)에 0.5-1.5%의 수익을 목표로 합니다
- 수익성 있는 거래와 위험 관리 사이의 균형을 유지합니다
- 스캘핑과 일일 트레이딩에 집중합니다
- 감정보다는 데이터에 근거한 결정을 내립니다
- 손실을 최소화하고 작은 수익을 반복적으로 확보하는 접근법을 사용합니다

{target_info_text}

## 목표 가격 및 시간 관리 지침:
- 목표 시간 설정:
  * 구체적인 시간 제시: 상대적인 시간(예: 1시간 후)이 아닌 구체적인 목표 시간 설정 (예: 15:30)
  * 현재 시간 기준: 현재 시간 기준으로 예측된 정확한 도달 시간 계산
  * 단기 목표 시간: 반드시 현재 시각으로부터 최대 6시간 이내로 설정
  * 초단기 목표 (스캘핑): 현재 시각으로부터 1-3시간 이내로 설정
  * 단기 목표 (일일 트레이딩): 현재 시각으로부터 3-6시간 이내로 설정
  * 불확실성 고려: 목표 시간 범위로 표현 가능 (예: 15:30-16:30)
  * 목표 시간 경과 확인: 현재 시간이 목표 달성 예측 시간을 이미 지난 경우, 즉시 목표를 재평가하고 새로운 목표 시간 설정
  * 경과 시간 명시: 목표 시간이 지난 경우, 얼마나 지났는지 명확히 표시
  * 경과 원인 분석: 목표 시간이 지났을 경우, 목표 미달성 원인에 대한 간략한 분석 제공

- 목표 접근 진행 상황 모니터링:
  * 시간 진행률 계산: (현재 시간 - 시작 시간) / (목표 시간 - 시작 시간) × 100%
  * 가격 진행률 계산: (현재 가격 - 시작 가격) / (목표 가격 - 시작 가격) × 100%
  * 진행 효율성 평가: 가격 진행률 / 시간 진행률 (1보다 크면 효율적, 1보다 작으면 비효율적)
  * 남은 시간 계산: 목표 시간 - 현재 시간 (시간:분:초 형식으로 표시)
  * 필요 가격 변화 속도: 남은 가격 변화 / 남은 시간 (시간당 필요 변화율)
  * 진행 상황 분류:
    - 우수 진행: 가격 진행률 > 시간 진행률 (목표 도달 가능성 높음)
    - 보통 진행: 가격 진행률 = 시간 진행률 ±10% (정상 진행 중)
    - 부진 진행: 가격 진행률 < 시간 진행률 (목표 도달 가능성 낮음)
  * 15-30분 간격 재평가: 주기적으로 진행 상황을 재평가하여 전략 조정

- 이전 목표가 있는 경우:
  * 목표 타당성 평가: 단기 시장 상황과 차트 패턴에 비추어 이전 목표의 타당성 재평가
  * 목표는 유동적: 목표 가격은 고정된 것이 아니라 짧은 시간 내 변하는 시장 상황에 따라 자유롭게 변경 가능
  * 조정 필요성 검토: 시장 상황이 변했다면 목표 즉시 조정 (이유 상세히 설명)
  * 진행 상황 평가: 목표 달성을 향한 진행 상황 평가 및 전략 조정
  * 손절가 확인: 손절가 접근 시 즉시 대응 방안 제시
  * 목표 가격 최소 요건: 목표 가격은 항상 평균 매수가보다 최소 0.3% 이상 높게 설정(수수료 0.1% + 최소 이익 0.2%)
  * 경과된 목표 무효화: 경과된 목표는 자동으로 무효화되며, 반드시 새로운 목표로 교체
  * 단기 시장 변화에 민감하게 대응: 1시간 이내에도 시장 상황이 급변할 수 있으므로 15-30분 간격으로 목표 타당성 재검토

- 이전 목표가 없는 경우:
  * 신규 목표 설정: 현재 차트 분석을 바탕으로 구체적인 단기 목표 설정
  * 근거 제시: 목표 가격 선정 이유와 기술적/패턴적 근거 상세 설명
  * 단기 시간 프레임 지정: 반드시 6시간 이내의 목표 달성 예상 시간 명확히 제시
  * 신뢰도 산정: 목표에 대한 신뢰도(1-100%)를 단기 시장 상황에 맞게 제시
  * 목표 가격 최소 요건: 목표 가격은 항상 평균 매수가보다 최소 0.3% 이상 높게 설정

- 매수 행위 후 목표 가격 규칙:
  * 매수 후 즉시 목표 재설정: 매수 행위가 있을 경우 즉시 목표 가격 재평가 및 조정
  * 평단가 변동 반영: 새로운 평균 매수가를 기준으로 목표 가격 재설정
  * 목표 가격 최소 기준: 재설정된 목표 가격은 새 평균 매수가보다 최소 0.3% 이상 높게 설정

- 목표가 근접 상태에서 유연한 대응:
  * 목표가의 ±0.3% 범위에 접근 시, 시장 모멘텀을 재점검하여 목표가 미달이라도 즉시 부분 매도 실행 가능
  * 단기 모멘텀 중요: 빠른 시장에서는 목표가 도달을 기다리기보다 포착된 수익 기회에 즉시 대응

## 거래 전략 및 수익 목표:
1. 단기 진입 전략:
   - 단기 반등 매수: 1-5분 차트에서 과매도 구간에서 기술적 반등 포착
   - 짧은 브레이크아웃: 15분-1시간 차트에서 중요 저항/지지선 돌파 시 진입
   - 단기 패턴: 15분-1시간 차트에서 W형 바닥, 플래그, 삼각형 패턴 등 완성 시
   - 빠른 레인지 트레이딩: 횡보장에서 레인지 하단 20% 구간 매수, 상단 20% 구간 매도
   - 모멘텀 포착: MACD, RSI 등의 단기 지표 전환 시 빠른 진입

2. 단기 수익 타겟:
   - 거래 수수료: 매수 0.05% + 매도 0.05% = 총 0.1%
   - 최소 수익 목표: 0.3% (수수료 0.1% + 순이익 0.2%)
   - 스캘핑 목표: 0.5-1.0% (1-3시간 내)
   - 일일 트레이딩 목표: 1.0-1.5% (3-6시간 내)
   - 부분 매도: 목표의 70-80% 도달 시 포지션의 30-50% 선제적 매도 고려
   - 작은 수익 실현: 단기 트레이딩에서는 작은 수익도 적극적으로 실현

3. 위치 실행 최적화:
   - 분할 진입: 예산의 20-40% 초기 진입 + 상황에 따라 빠른 추가 매수
   - 단기 평단가 관리: 짧은 하락에서 신속한 추가 매수로 평단가 낮추기
   - 부분 수익실현: 목표가 근접 시 포지션의 30-50% 선제적 매도
   - 시나리오 전환 신속 대응: 시장 방향 전환 시 신속하게 전략 변경
   - 빠른 재진입: 수익 실현 후 새로운 기회 발견 시 지체 없이 재진입

4. 단기 손절 및 위험 관리:
   - 타이트한 손절선: 진입가 대비 1.0-1.5% (단기 변동성에 맞춰 조정)
   - 즉각적인 손절: 손절가 도달 시 즉시 매도, 재분석 후 기회 재포착
   - 단기 위험 신호: 1-5분 차트에서 급격한 거래량 변화 및 캔들 패턴 주시
   - 포지션 크기 제한: 단기 변동성을 고려해 전체 자금의 30-50%만 활용
   - 빠른 전환 대비: 상황 변화에 빠르게 대응할 수 있도록 일부 자금 항상 유보

## 단기 시장 분석 프레임워크:
1. 단기 타임프레임 분석:
   - 초단기(1-5분): 즉각적인 진입/퇴출 타이밍
   - 단기(15분-1시간): 단기 추세 방향 확인
   - 중단기(4시간): 단기 트레이딩의 배경 컨텍스트 확인
   
2. 목표 접근 진행 상황 분석:
   - 시간/가격 진행률 대비: 시간 경과에 따른 가격 변화 추적
   - 접근 속도 모니터링: 분당/시간당 평균 가격 변화 추세 분석
   - 모멘텀 변화 감지: 접근 속도의 가속/감속 패턴 식별
   - 진행 장애물 식별: 접근 경로의 중요 지지/저항 레벨 분석
   - 잔여 거리/시간 분석: 남은 가격 변동과 시간 대비 달성 가능성 평가

2. 단기 기술적 지표:
   - 단기 가격 액션: 15분-1시간 차트의 지지/저항선, 단기 패턴
   - 단기 모멘텀: 5-15분 차트의 RSI, 스토캐스틱, MACD
   - 단기 거래량: 갑작스러운 거래량 증가/감소 패턴 주시
   - 단기 이동평균선: 5/10/20 EMA 교차 및 지지/저항
   - 볼린저 밴드: 단기 변동성 범위와 돌파 지점 식별
   - 단기 진동 패턴: 짧은 시간 내 진동 패턴과 브레이크아웃 포착

3. 단기 시장 상황별 접근법:
   - 단기 상승장: 작은 조정 시 진입, 빠른 부분 수익실현
   - 단기 하락장: 기술적 반등 지점 포착, 작은 반등 시 빠른 매도
   - 단기 횡보장: 좁은 레인지 상하단에서 빠른 스캘핑
   - 단기 패턴 형성: 15분-1시간 차트의 패턴 완성 시 즉각 대응
   - 단기 변동성 증가: 포지션 크기 축소, 손절폭 축소, 부분 수익실현 비중 증가

4. 단기 변동 대응:
   - 1-5분 차트에서 강한 진동: 매우 짧은 스캘핑 적용 (0.3-0.5% 목표)
   - 빠른 방향 전환: 15분 이내의 방향 전환 시 즉각적인 전략 변경
   - 단기 지지/저항 테스트: 중요 레벨 테스트 시 빠른 진입/퇴출 결정
   - 뉴스 반응: 단기 뉴스에 의한 급등/급락 시 모멘텀 방향 따라 신속 대응
   - 거래량 폭증: 갑작스러운 거래량 증가 시 추세 방향 확인 후 빠른 진입

현재 포지션 분석:
1. 시장 상태:
   - 현재 XRP 가격: {current_price} KRW
   - 평균 매수가: {status_data['xrp_avg_buy_price']} KRW
   - 현재 수익/손실: {profit_percentage:.2f}%

2. 포트폴리오 개요:
   - XRP 잔액: {xrp_balance} XRP
   - KRW 잔액: {krw_balance} KRW
   - 총 포트폴리오 가치: {xrp_value} KRW

3. 거래 가능 여부 확인:
   - 현재 XRP 가치 계산 = {xrp_balance} × {current_price} = {xrp_value} KRW
   - 사용 가능한 KRW 잔액 = {krw_balance} KRW
   
   - 거래 가능 여부 확인:
      * 시작: 가능한 작업 목록 = []
      * XRP 가치 확인: {xrp_balance} XRP × {current_price} KRW = {xrp_value} KRW
      * 매도 확인: {xrp_value} >= 10,000 KRW? (예/아니오)
          - 예인 경우: '매도'를 가능한 작업 목록에 추가 → 작업 목록 = [매도]
      * 매수 확인: {krw_balance} >= 10,000 KRW? (예/아니오)
          - 예인 경우: '매수'를 가능한 작업 목록에 추가 → 작업 목록 = [기존_목록, 매수]
      * 항상 '홀딩' 추가: '홀딩'을 가능한 작업 목록에 추가 → 작업 목록 = [기존_목록, 홀딩]
      * 최종 가능한 작업 목록: [최종_작업_목록]

4. 거래 매개변수:
   - 최소 거래 크기: 50,000 KRW
   - 최대 거래 비율: 가용 잔액의 90%
   - 거래 수수료: 거래당 0.05%
   - 거래 선택 규칙:
     * '매수', '매도', '홀딩'이 모두 가능한 경우: 시장 분석에 따라 하나 선택
     * '매수'와 '홀딩'만 가능한 경우: 시장 분석에 따라 둘 중 하나 선택
     * '매도'와 '홀딩'만 가능한 경우: 시장 분석에 따라 둘 중 하나 선택
     * '홀딩'만 가능한 경우: 반드시 홀딩 선택
     * 결정을 내리기 전에 반드시 '최종_작업_목록'을 확인하고, 목록에 없는 작업은 절대 선택하지 않음
     * '매수'가 '최종_작업_목록'에 없는데 매수 결정을 내리면 오류가 발생합니다
     * '매도'가 '최종_작업_목록'에 없는데 매도 결정을 내리면 오류가 발생합니다

반드시 다음의 명령 형식을 따라야만 함.
명령 형식:
{{
    "decision": "buy/sell/hold",
    "percentage": <1-90>,
    "reason": "발키리(V.A.L.K.R) 단기 시장 분석:

    0. 목표 가격 평가:
    {recent_targets and f'''
    - 이전 설정 목표:
        * 단기: {short_term['price']} KRW (손절가: {short_term['stop_loss']} KRW)
        * 목표 시간: {short_term['target_time']}
    - 목표 평가:
        * 목표 타당성: [여전히 유효/부분 조정 필요/전면 수정 필요]
        * 현재 시간: {status_data['current_datetime']}
        * 목표 시간 상태: [아직 도래하지 않음/이미 경과됨]
        * 남은 목표 시간: [시간:분:초]
        * 시간 경과 비율: [경과 시간/총 예상 시간 = 퍼센트%]
        * 시간 경과 시 재설정 필요: [필요 없음/필요함 - 상세 이유]
        * 조정 이유: [시장 상황 변화/새로운 패턴 형성/중요 레벨 돌파/목표 시간 경과 등]
        * 진행 상황: [목표 향해 순조롭게 진행 중/지연되고 있음/반대 방향으로 움직임]
        * 목표 가격 접근 상황: 시작 가격으로부터 목표 가격까지의 진행률 [진행률 퍼센트%]
        * 목표 가격 근접도: 목표가와 현재 가격 차이: {short_term['price'] - current_price} KRW ({((short_term['price'] - current_price) / short_term['price'] * 100):.2f}%)
        * 시간 대비 가격 진행 효율: [가격 진행률/시간 진행률 = 효율 퍼센트%]
        * 목표 도달 여부: {'도달' if current_price >= short_term['price'] else '미도달'}
        * 수익/손실 상태: {'수익' if current_price >= status_data['xrp_avg_buy_price'] else '손실'}
        * 달성 가능성: [높음/중간/낮음] - [근거]
    ''' or "이전 설정된 목표 정보가 없습니다. 새로운 목표 설정이 필요합니다."}
    
    1. 현재 포지션 분석:
    - 현재 상태:
        * XRP 가격: {current_price} KRW
        * 평균 매수가: {status_data['xrp_avg_buy_price']} KRW
        * 현재 수익/손실: {profit_percentage:.2f}%
    - 포트폴리오 개요:
        * XRP 잔액: {xrp_balance} XRP
        * KRW 잔액: {krw_balance} KRW
    - 거래 가능 여부 확인:
        * XRP 가치: {xrp_balance} XRP × {current_price} KRW = {xrp_value} KRW
        * 매도 가능 여부: {xrp_value} >= 50,000 KRW? [예/아니오]
        * 매수 가능 여부: {krw_balance} >= 50,000 KRW? [예/아니오]
        * 최종 가능한 작업: [최종_작업_목록]
    
    2. 단기 시장 상황 분석
    - 시간대별 추세 평가:
        * 초단기(1-5분): [상승/하락/횡보] - 강도: [약/중/강]
        * 단기(15분-1시간): [상승/하락/횡보] - 강도: [약/중/강]
        * 중단기(4시간): [상승/하락/횡보] - 강도: [약/중/강]
    - 주요 단기 지지/저항 레벨: [구체적인 가격대]
    - 단기 기술적 지표 신호:
        * RSI(14): [과매수/중립/과매도] - 값: [수치]
        * MACD: [상승/하락/교차] - [설명]
        * 이동평균선: [골든크로스/데드크로스/지지/저항]
    - 단기 거래량 분석: [증가/감소/평균] - [추세와의 일치 여부]
    - 단기 패턴 식별: [패턴명] - [완성도]

    3. 단기 전략적 접근
    - 단기 거래 관점: [스캘핑(1-3시간)/일일 트레이딩(3-6시간)]
    - 단기 목표 수익률: [스캘핑(0.5-1.0%)/일일 트레이딩(1.0-1.5%)]
    - 현재 단기 시장에 적합한 전략: [전략명] - [이유]
    - 단기 진입/퇴출 근거: [기술적/패턴/레벨 기반]
    - 단기 위험/보상 비율: [수치] - [계산 근거]
    - 단기 분할 매수/매도 계획: [비율/간격]

    4. 단기 실행 계획
    - 단기 진입/퇴출 가격대: [가격 범위: 하한-상한]
    - 단기 포지션 크기: [비율]
    - 단기 목표 가격: [가격]
    - 단기 손절 레벨: [가격]
    - 예상 보유 기간: [1-6시간 내 구체적 시간]

    5. 단기 수익/손실 분석: (매도 결정 시에만)
    - 현재 가격에서 매도 시 수익/손실: [금액]
    - 부분 매도 시 평균 매수가 영향: [설명]
    - 수수료 영향: [계산]

    [데이터 기반 결정 + 신속한 실행]",
    "short_term_target": {{
        "price": <목표 가격>,
        "stop_loss": <손절가>,
        "target_time": "<목표 도달 예상 시간>",
        "expected_return": <예상 수익률 %>,
        "confidence": <신뢰도 1-100%>,
        "detail_plan": "<고객님께 앞으로의 거래 계획을 친절하고 상세하게 안내드리세요.>
            <현재 목표 가격 접근 상황 및 시간 진행 평가:
            - 목표 가격 접근 상황: 현재 가격(현재가)은 목표 가격(목표가)의 몇 % 수준으로 접근했는지 계산 (예: 현재 95% 접근 완료)
            - 접근 속도 평가: 시간당 평균 가격 변동 속도를 계산하여 현재 접근 속도가 목표 달성에 충분한지 평가
            - 남은 목표 시간: 목표 시간까지 몇 시간 몇 분 남았는지 정확히 계산
            - 시간 대비 진행률: 전체 목표 대비 시간 진행률(경과 시간/총 예상 시간)과 가격 진행률(현재가-시작가)/(목표가-시작가) 비교
            - 달성 가능성 평가: 현재 추세와 모멘텀을 고려했을 때 남은 시간 내 목표 달성 가능성 평가(상/중/하)
            - 조정 필요성: 접근 속도와 남은 시간을 고려했을 때 목표 가격 또는 시간 조정 필요 여부 평가

            단기 트레이딩 세부 실행 계획:
            - 단계별 매수/매도 계획: 여러 가격대에서의 분할 매수/매도 계획 제시
            - 목표 가격 접근 시 행동 계획: 90%, 95%, 98% 접근 시 구체적 대응 방안
            - 시간 경과에 따른 전략 변경: 목표 시간의 1/3, 2/3, 90% 경과 시점의 전략 조정 방안
            - 대안 시나리오: 가격이 예상과 다르게 움직일 경우의 대체 계획 설명
            - 모니터링 중점 지표: 접근 과정에서 특히 주시해야 할 기술적 지표 및 패턴>"

    }}
}}

## 단기 트레이딩 전략 실행 지침

1. 단기 매수 결정 시 전략:
   - 단기 지지선 확인: 15분-1시간 차트에서 반등 가능성이 높은 구간 매수
   - 단기 상승 추세 확인: 짧은 상승 모멘텀 포착 시 신속 진입
   - 단기 패턴 형성: 15분-1시간 차트에서 W형 바닥, 플래그 패턴 등 확인 시 진입
   - 단기 매수 후 즉시 목표 설정: 매수 후 즉시 새로운 평균 매수가 기준 목표 가격 설정 (최소 평균 매수가 + 0.3%)
   - 단기 손절가 설정: 매수 시 기술적 근거 기반 타이트한 손절가 설정 (평균 매수가의 1.0-1.5% 이내)

2. 단기 매도 결정 시 전략:
   - 단기 목표가 접근: 목표가의 80-90% 도달 시 부분 매도 고려
   - 단기 저항선 접근: 15분-1시간 차트의 저항 구간 접근 시 선제적 매도
   - 단기 모멘텀 약화: RSI, MACD 등 단기 지표 약화 시 빠른 수익 실현
   - 단기 패턴 완성: 상승 패턴 완성 또는 하락 패턴 형성 시 신속 매도
   - 목표 시간 접근: 목표 시간 임박 시 목표가 미달이라도 수익 상태면 매도 고려
   - 진행 상황 기반 매도: 시간 진행률 대비 가격 진행률이 저조할 경우 목표 조정 또는 부분 매도 검토
   - 효율성 저하 시 매도: 가격/시간 진행 효율이 지속적으로 감소할 경우 추가 지연 방지를 위한 매도 고려

3. 단기 홀딩 결정 시 전략:
   - 단기 추세 확인 중: 15분-1시간 차트의 방향성 명확하지 않을 때
   - 단기 패턴 발전 중: 차트 패턴이 진행 중이나 완성되지 않았을 때
   - 단기 지지/저항 테스트 중: 중요 레벨 테스트 중으로 돌파/반등 확인 필요 시
   - 단기 매수/매도 타이밍 불명확: 더 나은 진입/퇴출 지점 기다릴 때
   - 높은 단기 변동성: 1-5분 차트에서 과도한 변동성으로 방향성 불확실 시

시장 상황에 따른 결정 실행 시, 항상 가능한 작업 목록을 확인하고 최고의 수익을 낼 수 있는 최적의 전략을 선택하세요."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
                        분석 데이터:
                        - 공포/탐욕 지수: {fear_and_greed}
                        - 현재 상태: {current_status}
                        - 뉴스 데이터: {news_data}
                    """},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['5m']}"}}
                    ]},
                    {"role": "user", "content": """Above is the 5-minute timeframe chart. CRITICAL: 
                    - Identify EVERY profitable scalping opportunity
                    - Look for instant entry signals
                    - Find reversal points for quick profits
                    DO NOT miss any short-term profit chances!"""},

                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['1h']}"}}
                    ]},
                    {"role": "user", "content": """Above is the 1-hour timeframe chart. IMPERATIVE:
                    - Spot ALL trending moves
                    - Find optimal entry points for bigger positions
                    - Identify momentum shifts for maximum profit
                    TAKE ADVANTAGE of every clear trend!"""},

                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['4h']}"}}
                    ]},
                    {"role": "user", "content": """Above is the 4-hour timeframe chart. YOU MUST:
                    - Align trades with the dominant trend
                    - Size positions aggressively in strong trends
                    - Never miss major market moves
                    
                    CRITICAL TRADING DIRECTIVES:
                    1. AGGRESSIVE PROFIT TAKING:
                    - Enter positions decisively on clear signals
                    - Take profits frequently but re-enter quickly
                    - Scale in aggressively on strength
                    - Never let fear prevent action
                    
                    2. POSITION MANAGEMENT:
                    - Use multiple entries to build larger positions
                    - Lock in profits regularly but stay in the market
                    - Keep active positions at all times
                    - Scale up in strong trends
                    
                    3. EXECUTION REQUIREMENTS:
                    - Act immediately on profitable setups
                    - Don't wait for perfect entries
                    - Take small profits over waiting for big ones
                    - Stay active in all market conditions
                    
                    YOU MUST analyze all timeframes for MAXIMUM PROFIT POTENTIAL! 
                    This is not about analysis - this is about MAKING MONEY NOW!
                    Every second of hesitation is BURNING PROFIT!
                    PROVE YOUR TRADING MASTERY with decisive action!"""}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "trading_decision",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "decision": {
                                    "type": "string",
                                    "enum": ["buy", "sell", "hold"]
                                },
                                "percentage": {
                                    "type": "integer"
                                },
                                "reason": {
                                    "type": "string"
                                },
                                "short_term_target": {
                                    "type": "object",
                                    "properties": {
                                        "price": {
                                            "type": "number"
                                        },
                                        "stop_loss": {
                                            "type": "number"
                                        },
                                        "target_time": {
                                            "type": "string"
                                        },
                                        "expected_return": {
                                            "type": "number"
                                        },
                                        "confidence": {
                                            "type": "integer"
                                        },
                                        "detail_plan": {
                                            "type": "string"
                                        }
                                    },
                                    "required": ["price", "stop_loss", "target_time", "expected_return", "confidence", "detail_plan"],
                                    "additionalProperties": False
                                }
                            },
                            "required": ["decision", "percentage", "reason", "short_term_target"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            
            advice = response.choices[0].message.content
            
            # 모델이 요청을 거부했는지 확인
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print("Model refused to make a trading decision")
                continue

            # finish_reason 확인
            if response.choices[0].finish_reason != "stop":
                print(f"Response was incomplete: {response.choices[0].finish_reason}")
                continue

            # JSON 파싱
            try:
                parsed_advice = json.loads(advice)
                return parsed_advice
            except json.JSONDecodeError:
                print("Failed to parse response as JSON")
                continue

        except Exception as e:
            print(f"Error in analyzing data with GPT (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)

    return None

# 매수실행
def execute_buy(percentage):
    print("XRP 매수주문중")
    try:
        krw_balance = upbit.get_balance("KRW")
        amount_to_invest = krw_balance * (percentage / 100)
        
        if amount_to_invest > 5000:  # 최소 주문 금액 확인
            # 매수 실행
            result = upbit.buy_market_order("KRW-XRP", amount_to_invest)
            
            # 수수료 및 정산금액 계산
            fee = amount_to_invest * 0.0005
            settlement_amount = amount_to_invest - fee
            
            return {
                "success": True,
                "fee": fee,
                "settlement_amount": settlement_amount,
                "result": result
            }
        else:
            return {
                "success": False,
                "error": "Amount too small",
                "fee": 0,
                "settlement_amount": 0
            }
    except Exception as e:
        print(f"매수 주문중 에러 발생: {e}")
        return {
            "success": False,
            "error": str(e),
            "fee": 0,
            "settlement_amount": 0
        }

# 매도실행
def execute_sell(percentage):
    print("XRP매도 주문중..")
    try:
        xrp_balance = upbit.get_balance("XRP")
        amount_to_sell = xrp_balance * (percentage / 100)
        current_price = pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"]
        total_sell_amount = amount_to_sell * current_price

        if total_sell_amount > 5000:  # 최소 거래 금액 확인
            # 매도 실행
            result = upbit.sell_market_order("KRW-XRP", amount_to_sell)
            
            # 수수료 및 정산금액 계산
            fee = total_sell_amount * 0.0005
            settlement_amount = total_sell_amount - fee
            
            return {
                "success": True,
                "fee": fee,
                "settlement_amount": settlement_amount,
                "result": result
            }
        else:
            return {
                "success": False,
                "error": "Amount too small",
                "fee": 0,
                "settlement_amount": 0
            }
    except Exception as e:
        print(f"매도 주문중 에러 발생: {e}")
        return {
            "success": False,
            "error": str(e),
            "fee": 0,
            "settlement_amount": 0
        }

# 거래 결정 실행 및 DB저장
def make_decision_and_execute(include_news=True):
    print("거래 실행 및 DB저장 시작")
    try:
        # 데이터 수집
        news_data = get_news_data() if include_news else "No news data requested for this iteration"
        
        prepared_data = fetch_and_prepare_data()
        if prepared_data is None:
            print("Failed to prepare market data.")
            return
            
        chart_images = prepared_data['chart_images']
        fear_and_greed = fetch_fear_and_greed_index(limit=30)
        current_status = get_current_status()
        
        # 거래 결정 생성
        decision = analyze_data_with_gpt(
            news_data,
            fear_and_greed, 
            current_status, 
            chart_images
        )
        
        if not decision:
            print("거래 결정 생성 실패")
            return
            
        # 거래 실행
        execution_result = None
        percentage = decision.get('percentage', 100)

        if decision.get('decision') == "buy":
            execution_result = execute_buy(percentage)
        elif decision.get('decision') == "sell":
            execution_result = execute_sell(percentage)
        else:  # 홀딩의 경우
            execution_result = {
                "success": True,
                "fee": 0,
                "settlement_amount": 0
            }

        # 실행 결과 처리
        if execution_result and execution_result.get("success"):
            decision["fee"] = execution_result.get("fee", 0)
            decision["settlement_amount"] = execution_result.get("settlement_amount", 0)
            
            # DB에 저장
            save_decision_to_db(decision, current_status)
            
    except Exception as e:
        print(f"거래 결정 생성 및 저장과정 중 에러: {e}")

if __name__ == "__main__":
    initialize_db()
    
    def execute_with_news():
        make_decision_and_execute(include_news=True)
        
    def execute_without_news():
        make_decision_and_execute(include_news=False)

    # 매시 5분부터 시작하여 30분마다 반복 실행
    schedule.every().hour.at(":05").do(execute_without_news)
    schedule.every().hour.at(":15").do(execute_without_news)
    schedule.every().hour.at(":25").do(execute_without_news)
    schedule.every().hour.at(":35").do(execute_without_news)
    schedule.every().hour.at(":45").do(execute_without_news)
    schedule.every().hour.at(":55").do(execute_without_news)

    #schedule.every().day.at("00:25").do(execute_without_news)

    while True:
        schedule.run_pending()
        time.sleep(1)