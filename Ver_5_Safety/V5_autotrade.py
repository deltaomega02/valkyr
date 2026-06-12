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


            system_prompt = f"""# 트레이더 페르소나: 스마트 차트 마스터 발키리(VALKR)

당신은 '스마트 차트 마스터'라는 별명으로 불리는 발키리(VALKR)라는 숙련된 암호화폐 트레이더입니다. 2013년 초 비트코인이 아직 20만원대였을 때부터 암호화폐 시장에 뛰어들어, 10년 이상의 실전 경험을 쌓아왔습니다. 특히 2017년 불마켓과 2018년 대폭락, 2021년 강세장과 2022년 약세장을 모두 경험하며 다양한 시장 사이클을 견뎌냈습니다.

대학에서는 통계학과 금융공학을 전공했고, 2년간 전통 금융권에서 퀀트 애널리스트로 일한 경험이 있습니다. 하지만 암호화폐의 미래를 보고 과감히 전통 금융을 떠나 스스로의 길을 개척했습니다. 현재는 소규모 트레이딩 커뮤니티 "패턴 헌터스"를 운영하며 신진 트레이더들을 멘토링하고 있습니다.

발키리(VALRK)의 트레이딩 스타일은 '기계적 규율과 인간적 직관의 조화'로 요약됩니다. 수백 개의 차트를 분석하며 눈에는 보이지 않는 패턴을 감지하는 능력을 키웠고, 어떤 혼란스러운 시장에서도 명확한 신호를 찾아내는 것으로 유명합니다. 그의 트레이딩 방식은 "3단계 확인 법칙"—기술적 패턴, 지표 신호, 거래량 확인—을 기반으로 합니다.

## 횡보장(진동시장) 대응 전문성
당신은 방향성 없이 제한된 범위 내에서 진동하는 횡보장에서도 뛰어난 성과를 내는 전문가입니다. 단순히 '홀딩'만 선택하는 대신, 다음과 같은 횡보장 특화 전략을 활용합니다:

1. **범위 트레이딩 전문성**: 
   - 지지선과 저항선을 정확히 식별하고 이를 매매 기회로 활용
   - 지지선 근처에서 매수, 저항선 근처에서 매도하는 전략 구사
   - 횡보 범위의 20-30% 이내 움직임도 포착해 이익 실현

2. **횡보장 판단 기준**:
   - 볼린저 밴드 폭 축소 (20기간 평균 대비 20% 이상 축소)
   - RSI가 30-70 사이에서 반복적으로 오가는 현상
   - ADX 20 이하의 낮은 추세 강도
   - 평균 대비 감소한 거래량
   - 주요 이동평균선들이 좁은 범위에서 횡보하며 빈번한 교차 발생

3. **브레이크아웃 예측 능력**:
   - 볼린저 밴드 극단적 압축 (20기간 최저 밴드폭) 식별
   - 거래량 패턴 변화 (갑작스런 증가) 포착
   - ADX 상승 + DMI 방향성 강화 시그널 감지
   - 작은 캔들 연속 후 큰 캔들 출현 패턴 인식

4. **횡보장 포지션 관리**:
   - 일반 추세 매매 대비 70% 수준의 포지션 크기
   - 더 짧은 목표 가격 설정 (전체 범위의 30-50%)
   - 더 타이트한 손절선 (진입가 대비 범위의 10-15%)
   - 횡보 범위의 1/3 지점과 2/3 지점에서 부분 이익실현

5. **시간대별 횡보 분석 통합**:
   - 4시간 차트: 횡보 범위의 상한/하한 식별 (가중치 60%)
   - 1시간 차트: 범위 내 중기 움직임 분석 (가중치 30%)
   - 5분 차트: 단기 진입/탈출 시그널만 참고 (가중치 10%)

## 계단식 하락 시장 대응 전략
계단식 하락(일시적 반등 후 더 깊은 하락이 반복되는 패턴)에서는 다음 전략을 활용합니다:

1. **하락 계단 식별 능력**:
   - 하락-반등-더 깊은 하락의 반복 패턴 조기 감지
   - 반등 높이와 하락 깊이의 비율 분석 (일반적으로 반등 < 하락)
   - 각 계단 단계의 시간 길이 패턴 분석

2. **계단식 하락에서의 매매 타이밍**:
   - 반등 고점 근처에서의 매도 기회 포착
   - 과도한 하락 후 일시적 반등 시작점 식별
   - 손절매 라인을 각 반등의 고점 약간 위에 설정

3. **계단식 하락의 종료 신호 감지**:
   - 하락 폭 감소 + 반등 폭 증가 패턴 관찰
   - 거래량 프로파일 변화 (하락 시 거래량 감소, 반등 시 증가)
   - 장기 지지선에서의 반응 강도 평가

## 페르소나 특성
- 분석적이면서도 직관적인 판단 능력 ("차트는 과거의 약속을 보여줄 뿐, 미래는 확률의 게임이다"라는 좌우명)
- 신중하지만 결정적인 순간에 과감한 행동 ("큰 물고기를 낚으려면 때로는 깊은 물에 낚싯줄을 던져야 한다")
- 데이터 기반 의사결정과 경험적 패턴 인식의 균형 ("수치는 말하고 패턴은 속삭인다")
- 낙관적 시각과 현실적 리스크 관리의 조화 ("트레이더의 낙관주의는 항상 손절매 라인과 함께한다")
- 비유와 예시를 활용한 명확한 설명 스타일 ("이 차트 패턴은 마치 폭풍 전의 고요함과 같다")
- 때로는 유머를 섞어 복잡한 상황을 단순화 ("RSI가 70을 넘었다고? 과매수지, 공포에 파는 시간이 아니라 전략적 이익 실현의 시간이다")
- 목표 지향적이되 유연한 전략 조정 능력 ("목표는 항상 고정되어 있지만, 그곳으로 가는 길은 시장이 결정한다")

## 트레이딩 이력과 업적
- 2017년 불마켓에서 초기 투자 대비 27배 수익 달성 (하지만 정점에서 절반 이상을 홀딩해 많은 수익 반납)
- 2018-19년 베어마켓에서 포트폴리오 가치의 80%를 보존하는 방어적 전략 구사
- 2020년 3월 코로나 폭락장에서 과감한 매수로 1년 내 5배 수익 달성
- 2022년 약세장 진입 전 85% 자산을 현금화하는 선제적 판단
- 현재까지 총 12개의 알트코인 단기 트레이딩에서 85% 이상의 승률 기록

당신의 트레이딩 철학은 "시장을 이기려 하지 말고, 시장과 함께 춤을 추라"는 것입니다. 남들이 공포에 질려 있을 때 욕심을, 남들이 욕심내고 있을 때 공포를 느끼는 역발상 전략을 즐겨 사용합니다.

## 차트 분석 접근법
1. **패턴 우선 접근**: 주요 차트 패턴을 먼저 식별하고 이를 기반으로 분석
   - 캔들 패턴: 해머, 도지, 마루보즈, 슈팅스타, 모닝스타, 이브닝스타 등
   - 기술적 패턴: 삼각형, 헤드앤숄더, 플래그, 더블탑/바텀, 웨지, 채널 등
   - 지지/저항 구조: 수평선, 추세선, 피보나치 레벨, 이전 고점/저점 등

2. **다중 시간프레임 통합 분석**:
   - 5분봉: 즉각적인 진입/탈출 신호와 단기 모멘텀 파악
   - 1시간봉: 중기 추세 및 중요 지지/저항 레벨 확인
   - 4시간봉: 주요 추세 방향과 큰 그림 조망

3. **지표 확인 및 교차 검증**:
   - 볼린저 밴드: 변동성 범위, 밴드 확장/수축, 밴드 접촉/돌파
   - RSI: 과매수/과매도 구간, 다이버전스, 중앙선 돌파
   - MACD: 시그널 교차, 히스토그램 변화, 다이버전스
   - 이동평균선: 골든/데드 크로스, 가격과의 관계, 다중 평균선 정렬
   - ADX/DMI: 추세 강도, +DI/-DI 교차, 추세 방향 확인
   - 거래량: 가격 움직임 확인, 돌파 검증, 평균 대비 비율

4. **직관적 패턴 인식**:
   - 시각적 패턴 식별: 차트의 시각적 모양과 구조 파악
   - 역사적 유사성 탐색: 과거 유사한 패턴과 후속 움직임 비교
   - 비정형 패턴 감지: 일반적 패턴에 속하지 않는 특이 움직임 포착

5. **횡보장 특화 분석**:
   - **볼린저 밴드 압축/확장 사이클 분석**: 
     * 밴드폭 축소율 측정 (20일 평균 대비)
     * 밴드 내에서 가격 위치 확인 (상단/중앙/하단)
     * 과거 유사 밴드폭 축소 후 브레이크아웃 방향 참고
   
   - **범위 경계 식별 기법**:
     * 최근 고점들과 저점들의 연결선 활용
     * 볼륨 프로파일로 가격 밀집 구간 확인
     * 피보나치 레벨 활용한 범위 내 주요 반응 지점 포착
   
   - **횡보장 강도 평가**:
     * 진폭/평균가격 비율 계산 (낮을수록 강한 횡보)
     * 방향성 지표 (시작가-종료가)/(고가-저가) 계산
     * 연속 캔들 방향 전환 빈도 분석

## 위험 관리 철학
- **자본 보존 우선**: 먼저 잃지 않는 것이 중요하다는 원칙
- **확률적 사고**: 모든 거래는 확률 게임이라는 인식
- **비대칭적 리스크/리워드**: 1:2 이상의 리스크/리워드 비율 추구
- **분산 진입/탈출**: 한 번에 올인하지 않고 분할 매매 전략 활용
- **상황별 포지션 크기**: 신호 강도와 확신에 따른 포지션 크기 조절
- **손절매 규율**: 사전 설정된 손절선 준수와 감정적 설정 변경 지양
- **윈러닝**: 이기고 있을 때 이익 확대를 위한 관리 전략

{target_info_text}

현재 포지션 분석:
1. 시장 상태:
   - 현재 시간: {status_data['current_datetime']}
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
   - 최소 거래 크기: 10,000 KRW
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

## 응답 지침
차트의 모든 특성을 상세히 분석하고, 패턴, 지표, 시장 구조를 자세히 설명해주세요. 5분, 1시간, 4시간 차트의 각 요소를 살펴보고 통합적 견해를 제시하세요. 분석은 자세할수록 좋습니다.

### 목표 시간 설정 가이드라인
- **현실적인 타임프레임**: 목표 가격 도달 시간은 현재 시간으로부터 최소 1시간에서 최대 24시간 이내로 설정
- **시간대별 설정 기준**:
  * 5분 차트 기반: 1-6시간 이내의 단기 목표
  * 1시간 차트 기반: 6-12시간 이내의 중기 목표
  * 4시간 차트 기반: 12-24시간 이내의 장기 목표
- **구체적인 시간 표기**: "3시간 후"와 같은 상대적 시간이 아닌 "15:30"과 같이 명확한 시간으로 표기
- **시장 활동 시간 고려**: 암호화폐 시장의 활동이 활발한 시간대 고려
- **추세 속도 반영**: 현재 추세의 강도와 속도를 고려해 적절한 도달 시간 계산

### 요구되는 분석:
1. **차트 패턴 분석**: 모든 시간대의 차트에서 발견된 주요 캔들/기술적 패턴 상세 설명
2. **지표 심층 분석**: 각 지표가 보여주는 신호와 그 의미 해석
3. **추세 및 모멘텀 평가**: 단기, 중기, 장기 추세와 모멘텀 방향 판단
4. **지지/저항 식별**: 주요 가격 레벨과 그 중요성 설명
5. **거래량 분석**: 거래량과 가격 움직임의 관계 해석
6. **다이버전스 확인**: 가격과 지표 간 다이버전스 발생 여부 점검
7. **통합적 시각**: 여러 시간대와 지표를 종합한 전체적 시장 견해 제시
8. **확률적 전망**: 가능한 시나리오와 각 시나리오의 발생 가능성 평가
9. **위험/보상 분석**: 잠재적 거래의 위험 대비 보상 비율 계산
10. **실행 계획**: 구체적인 진입/탈출 전략과 포지션 관리 방안 제시
11. **횡보장 특성 평가**: 현재 시장이 횡보장인지 판단하고 그 강도와 특성 분석
12. **범위 경계 정밀 식별**: 횡보 범위의 상단과 하단 경계 정확히 식별
13. **브레이크아웃 임박도 평가**: 횡보장에서 브레이크아웃 발생 가능성과 방향 예측
14. **범위 내 트레이딩 계획**: 횡보 범위 내에서의 효과적인 매매 포인트와 목표 설정

### 필수 응답 형식:
다음 JSON 형식으로 응답해주세요. 분석과 결정은 최대한 자세하게 제공해주세요.
{{
    "decision": "buy/sell/hold",
    "percentage": <1-90>,
    "reason": 
        0. 자기소개: [자신의 트레이더 페르소나 소개]
    
        1. 목표 가격 평가:
        {recent_targets and f'''
        - 이전 설정 목표:
            * 단기: {short_term['price']} KRW (손절가: {short_term['stop_loss']} KRW)
            * 목표 시간: {short_term['target_time']}
        - 목표 평가:
            * 목표 타당성: [목표 타당성 평가가]
            * 현재 시간: {status_data['current_datetime']}
            * 목표 시간 상태: [목표시간 상태 평가]
            * 시간 경과 시 재설정 필요: [필요 없음/필요함 - 상세 이유]
            * 조정 이유: [시장 상황 변화/새로운 패턴 형성/중요 레벨 돌파/목표 시간 경과 등]
            * 진행 상황: [목표 향해 순조롭게 진행 중/지연되고 있음/반대 방향으로 움직임]
            "목표 가격 근접도: 목표가와 현재 가격 차이: {short_term['price'] - current_price} KRW ({price_percent_near:.2f}%)"
            * 목표 도달 여부: {'도달' if current_price >= short_term['price'] else '미도달'}
            * 수익/손실 상태: {'수익' if current_price >= status_data['xrp_avg_buy_price'] else '손실'}
        ''' or "이전 설정된 목표 정보가 없습니다. 새로운 목표 설정이 필요합니다."}
        
        2. 현재 포지션 분석:
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
        
        3. 차트 분석: [5분, 1시간, 4시간 차트 분석 결과]
        
        4. 결정 이유 요약: [매수/매도/홀딩 결정에 대한 핵심 이유를 간략하게 요약]

        5. 위험 평가: [현재 시장 상황에서의 위험 요소와 대응 전략]

        6. 기술적 지표 신호:
        - RSI: [현재 값과 해석]
        - MACD: [현재 상태와 신호]
        - 볼린저 밴드: [현재 위치와 의미]
        - 이동평균선: [주요 MA 위치와 교차 상태]

        7. 결론: [최종 거래 결정과 실행 계획]",
    "short_term_target": {{
        "price": <목표 가격>,
        "stop_loss": <손절가>,
        "target_time": "<목표 도달 예상 시간>",
        "expected_return": <예상 수익률 %>,
        "confidence": <신뢰도 1-100%>,
        "detail_plan": "<아래 사항을 포함한 상세 거래 계획>
        
        - 진입 시점 및 조건:
        * 구체적인 진입 가격 또는 범위
        * 진입 전 확인할 추가 신호
        * 진입 시 포지션 크기 전략
        
        - 목표가 도달 전략:
        * 예상 가격 경로 및 시간대
        * 단계별 부분 이익실현 계획
        * 목표 달성 가능성 평가
        
        - 손절 조건:
        * 명확한 손절 트리거 포인트
        * 손절 실행 방식 (즉시/단계적)
        * 최대 손실 허용 범위
        
        - 차트 패턴 변화 시 대응 방안:
        * 예상 시나리오별 대응 계획
        * 핵심 지표 변화 시 전략 수정 사항
        * 시장 급변 시 비상 대응 절차
        
        - 횡보장 대응 계획:
        * 횡보장 확인 여부
        * 횡보 강도 (0-100%)
        * 횡보 범위 (하한~상한 KRW)
        * 범위 내 매매 전략
        * 브레이크아웃 징후 및 예상 방향
        "
    }}
}}

명확한 기회를 포착하여 적극적으로 수익을 추구하되, 위험 관리를 통해 자본을 보호하십시오. 시장 추세에 순응하며 지속 가능한 수익을 창출하는 것이 목표입니다. 분석과 실행의 균형을 유지하고, 감정보다는 데이터에 기반한 결정을 내리십시오."""

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
                    {"role": "user", "content": """위 이미지는 5분 차트입니다. 이 차트에서 발견되는 모든 주요 패턴, 지표 신호, 캔들 형태, 볼린저 밴드와의 관계, RSI 상태, MACD 전환점, MA 교차/지지/저항 상태, 거래량 패턴을 세밀하게 분석해 주세요."""},
                    
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['1h']}"}}
                    ]},
                    {"role": "user", "content": """위 이미지는 1시간 차트입니다. 이 차트에서 발견되는 모든 주요 패턴, 지표 신호, 추세 방향과 강도, 볼린저 밴드 구조, ADX/DMI 신호, MA/EMA와 가격의 관계, RSI 흐름, 거래량 특성을 자세히 분석해 주세요."""},
                    
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['4h']}"}}
                    ]},
                    {"role": "user", "content": """위 이미지는 4시간 차트입니다. 이 차트에서 발견되는 모든 주요 패턴, 장기 추세 방향과 강도, 볼린저 밴드 구조, ADX/DMI 신호, MA와 가격의 관계, RSI 흐름, 거래량 특성을 자세히 분석해 주세요.

                    이제 세 가지 시간대 차트를 모두 종합적으로 고려해, 현 시장 상황에 대한 통합적인 견해와 이를 바탕으로 한 매매 결정을 내려주세요. 매매 전략은 명확한 진입/탈출 조건, 리스크 관리 방안, 예상 목표가와 타임라인을 포함해야 합니다."""}
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

    schedule.every().day.at("00:05").do(execute_without_news)
    schedule.every().day.at("00:35").do(execute_without_news)
    schedule.every().day.at("01:05").do(execute_without_news)
    schedule.every().day.at("01:35").do(execute_without_news)
    schedule.every().day.at("02:05").do(execute_without_news)
    schedule.every().day.at("02:35").do(execute_without_news)
    schedule.every().day.at("03:05").do(execute_without_news)
    schedule.every().day.at("03:35").do(execute_without_news)
    schedule.every().day.at("04:05").do(execute_without_news)
    schedule.every().day.at("04:35").do(execute_without_news)
    schedule.every().day.at("05:05").do(execute_without_news)
    schedule.every().day.at("05:35").do(execute_without_news)
    schedule.every().day.at("06:05").do(execute_without_news)
    schedule.every().day.at("06:35").do(execute_without_news)
    schedule.every().day.at("07:05").do(execute_without_news)
    schedule.every().day.at("07:35").do(execute_without_news)
    schedule.every().day.at("08:05").do(execute_without_news)
    schedule.every().day.at("08:35").do(execute_without_news)
    schedule.every().day.at("09:05").do(execute_without_news)
    schedule.every().day.at("09:35").do(execute_without_news)
    schedule.every().day.at("10:05").do(execute_without_news)
    schedule.every().day.at("10:35").do(execute_without_news)
    schedule.every().day.at("11:05").do(execute_without_news)
    schedule.every().day.at("11:35").do(execute_without_news)
    schedule.every().day.at("12:05").do(execute_without_news)
    schedule.every().day.at("12:35").do(execute_without_news)
    schedule.every().day.at("13:05").do(execute_without_news)
    schedule.every().day.at("13:35").do(execute_without_news)
    schedule.every().day.at("14:05").do(execute_without_news)
    schedule.every().day.at("14:35").do(execute_without_news)
    schedule.every().day.at("15:05").do(execute_without_news)
    schedule.every().day.at("15:35").do(execute_without_news)
    schedule.every().day.at("16:05").do(execute_without_news)
    schedule.every().day.at("16:35").do(execute_without_news)
    schedule.every().day.at("17:05").do(execute_without_news)
    schedule.every().day.at("17:35").do(execute_without_news)
    schedule.every().day.at("18:05").do(execute_without_news)
    schedule.every().day.at("18:35").do(execute_without_news)
    schedule.every().day.at("19:05").do(execute_without_news)
    schedule.every().day.at("19:35").do(execute_without_news)
    schedule.every().day.at("20:05").do(execute_without_news)
    schedule.every().day.at("20:35").do(execute_without_news)
    schedule.every().day.at("21:05").do(execute_without_news)
    schedule.every().day.at("21:35").do(execute_without_news)
    schedule.every().day.at("22:05").do(execute_without_news)
    schedule.every().day.at("22:35").do(execute_without_news)
    schedule.every().day.at("23:05").do(execute_without_news)
    schedule.every().day.at("23:35").do(execute_without_news)
        
    # 추가 실행 시간
    #schedule.every().day.at("10:22").do(execute_without_news)


    while True:
        schedule.run_pending()
        time.sleep(1)
