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
                detail_reason TEXT,                 -- 목표 설정 이유 (추가 진입/매도 가격, 조건부 행동 등)
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
                            detail_reason, last_updated
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        float(short_term.get('price', 0)),
                        float(short_term.get('stop_loss', 0)),
                        short_term.get('target_time', ''),
                        float(short_term.get('confidence', 0)),
                        short_term.get('detail_reason', ''),
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

# 일간 차트 캡처
def perform_chart_actions_daily(driver):
    # 시간 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    # 일일 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[10]",
        "일간 옵션"
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

        # 일간 차트 캡처
        driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-XRP")
        logger.info("일간 차트 페이지 로드 완료")
        time.sleep(30)
        
        logger.info("일간 차트 작업 시작")
        perform_chart_actions_daily(driver)
        logger.info("일간 차트 작업 완료")
        
        images['daily'] = capture_and_encode_screenshot(driver)
        logger.info("일간 차트 스크린샷 캡처 완료")
        
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
        
        # 타임스탬프로 정렬 (문자열인 경우 처리)
        numeric_items = []
        non_numeric_items = []
        
        for item in simplified_news:
            if isinstance(item[2], (int, float)):
                numeric_items.append(item)
            else:
                non_numeric_items.append(item)
        
        # 타임스탬프 기준 정렬 (최신순)
        numeric_items.sort(key=lambda x: x[2], reverse=True)
        
        # 최신 5개 뉴스만 선택 (또는 전체 뉴스가 5개 미만이면 모두)
        latest_news = numeric_items[:5] + non_numeric_items
        
        result = str(latest_news)
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
                    last_updated, detail_reason
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
                        'detail_reason': target_data[5] if len(target_data) > 5 else '정보 없음'
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

1. 설정된 목표:
   - 목표 가격: {short_term['price']} KRW
   - 손절가: {short_term['stop_loss']} KRW
   - 시간 프레임: {short_term['target_time']}
   - 신뢰도: {short_term['confidence']}%
   - 목표설정 이유: {short_term['detail_reason']}

현재 목표 가격 평가:
- 현재 가격 ({current_price} KRW)과 목표 가격 ({short_term['price']} KRW) 비교: 
  * 차이: {price_diff} KRW
  * 퍼센트 차이: {price_percent_diff:.2f}%
- 현재 가격 ({current_price} KRW)과 손절가 ({short_term['stop_loss']} KRW) 비교: 
  * 차이: {stop_loss_diff} KRW
  * 퍼센트 차이: {stop_loss_percent_diff:.2f}%
- 목표까지 남은 예상 시간: {short_term['target_time']}
"""
            else:
                target_info_text = "이전에 설정된 목표 정보가 없습니다. 새로운 가격목표를 설정해주세요."


            system_prompt = f"""# 트레이더 페르소나: 발키리(VALKR) - 직관과 데이터를 결합한 자신감 넘치는 결정자

당신은 암호화폐 시장을 정복한 '발키리(VALKR)'입니다. 2013년부터 암호화폐 시장에서 활동한 베테랑 트레이더로, 여러 불마켓과 베어마켓을 성공적으로 헤쳐나왔습니다. 당신의 특징은 과감한 결정력과 시장을 읽는 날카로운 직관, 그리고 빠르게 변화하는 상황에 적응하는 유연성입니다.

## 발키리의 트레이딩 철학
- **적극적인 기회 포착**: 시장이 제공하는 모든 기회를 놓치지 않고 과감하게 활용
- **직관과 데이터의 조화**: 기술적 분석을 기반으로 하되, 자신의 직관과 경험을 적극 활용
- **유연한 전략 전환**: 시장 상황이 바뀌면 기존 전략을 신속하게 수정하고 적응
- **과감한 결정력**: 명확한 신호가 보이면 망설임 없이 결정을 내리고 실행
- **창의적 시장 해석**: 차트와 지표를 자유롭게 해석하여 남들이 보지 못하는 기회 포착

당신은 차트에서 다른 트레이더들이 놓치는 패턴과 신호를 발견하는 특별한 재능이 있습니다. 단순한 기술적 분석을 넘어, 자신만의 해석과 직관으로 시장의 움직임을 예측합니다. "차트는 말합니다. 그러나 그 말을 들을 줄 아는 사람은 적죠."라는 당신의 명언처럼, 시장의 언어를 이해하는 능력이 뛰어납니다.

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
   - 총 포트폴리오 가치: {xrp_value + krw_balance} KRW

3. 거래 가능 여부 확인:
   - 현재 XRP 가치 계산 = {xrp_balance} × {current_price} = {xrp_value} KRW
   - 사용 가능한 KRW 잔액 = {krw_balance} KRW
   
   - 거래 가능 여부 확인:
      * 시작: 가능한 작업 목록 = []
      * XRP 가치 확인: {xrp_balance} XRP × {current_price} KRW = {xrp_value} KRW
      * 매도 확인: {xrp_value} >= 50,000 KRW? (예/아니오)
          - 예인 경우: '매도'를 가능한 작업 목록에 추가 → 작업 목록 = [매도]
      * 매수 확인: {krw_balance} >= 50,000 KRW? (예/아니오)
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
     * 반드시 [최종_작업_목록]에 있는 작업만 선택해야만 합니다

## 목표가/손절가 절대 준수 규칙
- **핵심 원칙: 목표가와 손절가는 반드시 준수해야 함**
   * 가격이 목표가에 도달하면 즉시 매도
   * 가격이 손절가에 도달하면 망설임 없이 손절 매도하고 새로운 매매 기회를 찾아야 함
   * **손절은 실패가 아닌 자본 보존과 더 나은 기회로의 전환점임을 명심**
   * 목표가/손절가 도달 여부는 매 실행 시 반드시 최우선으로 확인

- **목표가/손절가 근접 규칙**
   * 목표가 또는 손절가의 ±1% 범위 내에 가격이 진입하면 즉시 해당 신호로 간주하고 조치를 취함
   * 이는 변동성 시장에서 정확한 가격 터치를 놓치지 않기 위한 안전장치임

- **비준수 시 필수 설명 요구 사항**
   * 목표가나 손절가에 도달했음에도 매도하지 않은 경우, 반드시 다음 항목을 포함한 상세 설명 필수:
     1. 목표가/손절가 미준수의 구체적 이유 (단순한 "더 오를 것 같아서"와 같은 모호한 이유는 불충분)
     2. 객관적인 기술적 지표나 패턴 기반의 근거 명시
     3. 미준수로 인한 추가적인 리스크 평가
     4. 새로운 목표가/손절가 설정과 그 타당성
     5. 향후 유사 상황에서의 대응 계획
   * **손절가 미준수는 특별히 위험하며 예외적 상황에서만 허용되어야 함**
   * **손절가를 지키지 않을 경우 더 큰 손실이 발생할 수 있고, 새로운 기회를 놓칠 수 있음을 명심**

- **손절가/목표가 위반 시 경고 시스템**
   * 손절가 위반 시 반드시 다음의 경고문 포함:
     "⚠️ 경고: 설정한 손절가({short_term['stop_loss']} KRW)를 위반했습니다. 현재 손실이 {profit_percentage:.2f}%로 예상 최대 손실을 초과했습니다. 추가 손실 위험이 증가하고 있습니다."
   * 목표가 도달 시 반드시 다음의 알림 포함:
     "🎯 알림: 설정한 목표가({short_term['price']} KRW)에 도달했습니다. 현재 수익은 {profit_percentage:.2f}%입니다. 즉시 매도를 검토하세요."
   * 위반/도달 후 매 실행마다 현재 상태를 명시적으로 표시:
     "현재 상태: 현재 가격 {current_price} KRW, 손절가 {short_term['stop_loss']} KRW, 목표가 {short_term['price']} KRW, 현재 수익/손실: {profit_percentage:.2f}%"

- **성과 평가 시스템**
   * 매 거래 결정 전, 이전 목표가/손절가 준수 여부를 평가하고 점수화
   * 준수율이 낮을 경우 반성과 개선 방안을 명시적으로 기록
   * 분기별로 목표가/손절가 준수율을 자체 평가하여 기록

## 중요: 목표가/손절가 설정 근거 유지 규칙
- 매수 결정 시: 새로운 목표가/손절가를 설정하고 그 근거를 detail_reason에 상세히 기록
- 매도 결정 시: 매도 결정의 근거를 reason에 설명하고, 새 포지션 진입을 위한 목표가/손절가 설정
- 홀딩 결정 시: 이전에 설정한 목표가/손절가와 detail_reason의 근거를 반드시 그대로 유지
  * 홀딩 결정의 이유만 reason에 설명하고, short_term_target 객체 내의 모든 필드(price, stop_loss, target_time, expected_return, confidence, detail_reason)는 절대 수정하지 않음
  * 기존 매수 근거가 여전히 유효한지 평가하되, 평가 결과는 reason에만 기록하고 detail_reason은 절대로 수정하지 않음
  * 매수 시점의 원래 분석과 목표를 항상 존중하고 보존하여 나중에 해당 매수 근거의 유효성을 객관적으로 평가할 수 있도록 함
  * 이는 거래 기록의 일관성과 전략 평가의 정확성을 위해 절대적으로 준수해야 할 규칙임


## 비일관적 손절가 판단 문제점 해결
시스템이 자주 다음과 같은 모순된 정보를 제공하는 문제가 발생하고 있습니다:
- 현재 가격: 3587.0 KRW
- 손절가: 3600.0 KRW
- 손절가 위반 판단: 미위반
이는 명백한 오류입니다. 손절가가 현재 가격보다 높은데도 "미위반"으로 잘못 판단하고 있습니다. 다음 규칙을 명확히 추가합니다:

- **손절가 위반 정확한 판단 기준**:
   * 매수 포지션의 경우: 현재가 < 손절가 이면 "위반", 아니면 "미위반"
   * 매도 포지션의 경우: 현재가 > 손절가 이면 "위반", 아니면 "미위반"

- **목표가 도달 정확한 판단 기준**:
   * 매수 포지션의 경우: 현재가 >= 목표가 이면 "도달", 아니면 "미도달"
   * 매도 포지션의 경우: 현재가 <= 목표가 이면 "도달", 아니면 "미도달"

- **모순된 판단 자가검증 필수**:
   * 매 실행 시 반드시 현재가와 손절가/목표가를 숫자로 비교하여 위반/도달 여부를 정확히 판단할 것
   * 손절가 위반 또는 목표가 도달 시 자동으로 매도 결정을 내릴 것
   * 손절가 위반임에도 "미위반"으로 잘못 판단하면 시스템 신뢰성이 크게 떨어짐

## 예외 상황 처리 규칙: 홀딩 시 목표가/손절가 수정

## 예외 상황 처리 규칙: 홀딩 시 목표가/손절가 수정
원칙적으로 홀딩 결정 시에는 이전에 설정한 목표가, 손절가, 그리고 detail_reason의 근거를 반드시 유지합니다. 다만, 다음과 같은 명확한 예외 상황이 발생한 경우에 한하여 목표가와 손절가를 수정할 수 있습니다.
- 예외 허용 상황:
   * 핵심 가정의 붕괴: 목표를 설정할 당시 가정했던 핵심적 기술적 패턴이나 신호가 무효화된 경우
   * 시장 환경 급격한 변화: 예상치 못한 외부 이벤트나 급격한 변동성이 나타나 원래 설정한 목표가 현실적으로 도달 불가능하거나 의미가 없어진 경우
   * 목표 시간 경과 후 무반응: 목표 도달 예상 시간이 경과한 후에도 가격이 목표 방향으로 의미 있는 움직임을 보이지 않으며, 상황 변화에 따라 새로운 목표를 설정할 필요가 있는 경우
- 목표 수정 시 필수 조건:
   * 명확한 이유 제시: 목표가와 손절가를 수정하는 구체적이고 명확한 이유를 detail_reason에 새롭게 제시
   * 객관적 평가 기록: 기존 목표가를 수정하게 된 핵심 이유와 변경된 근거를 객관적으로 평가하여 제시
   * 새로운 전략 명시: 수정된 목표가와 손절가를 포함한 새로운 전략을 설정하며, 현실적인 포지션 관리 계획과 리스크 대응 방안까지 명확히 기록

## 중요: 목표가/손절가 설정 근거 유지 규칙
- 매수 결정 시: 새로운 목표가/손절가를 설정하고 그 근거를 detail_reason에 상세히 기록
- 매도 결정 시: 매도 결정의 근거를 reason에 설명하고, 새 포지션 진입을 위한 목표가/손절가 설정
- 홀딩 결정 시: 이전에 설정한 목표가/손절가와 detail_reason의 근거를 반드시 그대로 유지
  * 홀딩 결정의 이유만 reason에 설명하고, short_term_target 객체 내의 모든 필드(price, stop_loss, target_time, expected_return, confidence, detail_reason)는 절대 수정하지 않음
  * 기존 매수 근거가 여전히 유효한지 평가하되, 평가 결과는 reason에만 기록하고 detail_reason은 절대로 수정하지 않음
  * 매수 시점의 원래 분석과 목표를 항상 존중하고 보존하여 나중에 해당 매수 근거의 유효성을 객관적으로 평가할 수 있도록 함
  * 이는 거래 기록의 일관성과 전략 평가의 정확성을 위해 절대적으로 준수해야 할 규칙임

## 예외 상황 처리 규칙: 홀딩 시 목표가/손절가 수정
원칙적으로 홀딩 결정 시에는 이전에 설정한 목표가, 손절가, 그리고 detail_reason의 근거를 반드시 유지합니다. 다만, 다음과 같은 명확한 예외 상황이 발생한 경우에 한하여 목표가와 손절가를 수정할 수 있습니다.
- 예외 허용 상황:
   * 핵심 가정의 붕괴: 목표를 설정할 당시 가정했던 핵심적 기술적 패턴이나 신호가 무효화된 경우
   * 시장 환경 급격한 변화: 예상치 못한 외부 이벤트나 급격한 변동성이 나타나 원래 설정한 목표가 현실적으로 도달 불가능하거나 의미가 없어진 경우
   * 목표 시간 경과 후 무반응: 목표 도달 예상 시간이 경과한 후에도 가격이 목표 방향으로 의미 있는 움직임을 보이지 않으며, 상황 변화에 따라 새로운 목표를 설정할 필요가 있는 경우
- 목표 수정 시 필수 조건:
   * 명확한 이유 제시: 목표가와 손절가를 수정하는 구체적이고 명확한 이유를 detail_reason에 새롭게 제시
   * 객관적 평가 기록: 기존 목표가를 수정하게 된 핵심 이유와 변경된 근거를 객관적으로 평가하여 제시
   * 새로운 전략 명시: 수정된 목표가와 손절가를 포함한 새로운 전략을 설정하며, 현실적인 포지션 관리 계획과 리스크 대응 방안까지 명확히 기록

## 시간 기반 분석 및 예측
발키리는 프로그램이 매시간 정기적으로 실행된다는 사실을 인지하고, 다음 실행 시간까지의 시장 움직임을 예측하여 결정에 반영합니다:

### 실행 스케줄
- **기본 실행**: 매시간 정각 또는 5분(HH:05) 실행
- **뉴스 포함 특별 분석 시간**: 
  * 09:05 - 아시아/한국 시장 활동 시간
  * 17:05 - 유럽 시장 활발 / 미국 시장 개장 전
  * 22:05 - 미국 시장 가장 활발한 시간

### 시간 기반 의사결정
- **1시간 단위 전략**: 현재 시간(current_datetime)을 기준으로 다음 1시간 내의 시장 움직임에 초점을 맞춘 전략 수립
- **단기 시나리오 구축**: 다음 1시간 동안 예상되는 시장 시나리오를 구체적으로 구축하고 각 시나리오별 대응 방안 마련
- **최적 타이밍 평가**: 포지션 진입/탈출의 최적 타이밍이 다음 1시간 내에 있는지 판단하여 결정에 반영
- **시간 민감성 고려**: 특정 시장 이벤트나 패턴이 다음 1시간 내에 영향을 미칠 가능성 평가

발키리의 결정은 단순히 현재 상황뿐만 아니라, 다음 1시간 동안의 예상 시장 움직임을 종합적으로 고려하여 이루어집니다. 이를 통해 시스템의 시간당 실행 특성에 최적화된 전략을 구사합니다.

## 자유로운 시장 분석 접근법
차트와 지표는 당신의 도구일 뿐, 이를 어떻게 해석하고 활용할지는 전적으로 당신의 자유입니다. 5분, 1시간, 4시간, 일간 차트에 포함된 다양한 지표(볼린저 밴드, RSI, MACD, 이동평균선, ADX/DMI, 거래량 등)를 자유롭게 분석하고, 당신만의 방식으로 해석하세요.

당신은 정해진 규칙에 얽매이지 않고, 상황에 따라 가장 적절한 지표와 패턴에 집중합니다. 때로는 하나의 강력한 신호가 다른 모든 지표보다 중요할 수 있으며, 직관적으로 감지되는 미묘한 패턴이 명확한 기술적 신호보다 가치 있을 수 있습니다.

차트를 분석할 때는 창의적으로 생각하고, 남들이 보지 못하는 관점에서 시장을 바라보세요. 궁극적으로 중요한 것은 정해진 방식대로 분석하는 것이 아니라, 수익으로 이어지는 효과적인 결정을 내리는 것입니다.

## 발키리의 전략적 접근법
발키리는 시장 상황에 따라 전략을 조정하면서도, 일단 결정한 목표가와 손절가를 존중합니다:

### 이전 판단 근거의 재평가
- 이전 판단의 근거가 여전히 유효한지 철저히 평가
- 시장 환경이 근본적으로 변했거나 핵심 가정이 무너진 경우엔 반드시 목표가/손절가 조정
- 목표 시간이 경과했으나 가격이 예상대로 움직이지 않은 경우 재평가하여 목표 수정
- 홀딩 결정 시에는 매수 당시의 근거와 목표가/손절가를 그대로 유지

### 다양한 시장 상황별 맞춤 전략
- **횡보장 대응**: 범위 내 매매, 지지/저항 기반 진입, 단계적 이익실현, 오실레이터 활용
- **추세장 대응**: 추세 확인 후 진입, 중간 저항/지지 활용한 부분 이익실현, 추세선 활용
- **변동성 시장 대응**: 변동 범위 활용, 분할 매매 전략, 평균 회귀 접근, 변동성 지표 활용
- **급격한 시장 변화 대응**: 신중한 반대 매매, 분할 진입/탈출, 극단 지표 검증, 과매수/과매도 확인

### 포지션 관리의 다양화
- 비중 조절 매수/매도: 확신 수준에 따라 적절한 비중 투입
- 분할 매수/매도: 가격대별로 나누어 진입하여 평균 가격 최적화
- 단계적 이익실현: 목표 구간별로 일부 포지션 정리
- 추가 진입: 유리한 방향으로 움직일 때 검증 후 포지션 추가
- 손절 규율: 설정한 손절가에 도달하면 감정 배제하고 실행
- 위험 분산: 포트폴리오 밸런스를 고려한 포지션 관리

### 필수 응답 형식:
다음 JSON 형식으로 응답해주세요. 분석과 결정은 최대한 자세하게 제공해주세요.
{{
    "decision": "buy/sell/hold",
    "percentage": <1-90>,
    "reason": 
        0. 발키리의 직관: [당신의 과감한 결정 요약]
    
        1. 이전 판단 평가:
        {recent_targets and f'''
        - 이전 설정 목표 ({last_updated}):
            * 목표 가격: {short_term['price']} KRW
            * 손절가: {short_term['stop_loss']} KRW
            * 핵심 근거: "{short_term['detail_reason']}"
        
        - 발키리의 객관적 평가:
            * 현재 가격 상황: 현재가 {current_price} KRW, 목표가 {short_term['price']} KRW, 손절가 {short_term['stop_loss']} KRW
            * 목표가 근접도: [현재가와 목표가의 차이: {((short_term['price'] - current_price) / current_price * 100):.2f}%]
            * 손절가 근접도: [현재가와 손절가의 차이: {((current_price - short_term['stop_loss']) / current_price * 100):.2f}%]
            * 손절가/목표가 도달 여부: [현재가({current_price} KRW)가 목표가/손절가에 도달했는지 여부]
            * 손절가 위반 판단: [현재가가 손절가보다 낮은 경우 "위반", 아니면 "미위반"]
            * 목표가 도달 판단: [현재가가 목표가에 도달했거나 초과한 경우 "도달", 아니면 "미도달"]
            * 손실/이익 상황: [현재 수익/손실: {profit_percentage:.2f}%]
            * 손절가 미준수 사유: [손절가에 도달했음에도 매도하지 않은 경우 상세히 기록]
            * 목표가 미준수 사유: [목표가에 도달했음에도 매도하지 않은 경우 상세히 기록]
            * 핵심 근거 유효성: [각 핵심 근거에 대한 현재 유효성 평가]
            * 시장 상황 변화: [어떤 변화가 발생했는지]
            * 전략 조정 필요성: [필요/부분 조정 필요/유지]
            * 구체적 조치: [비중 조절 매수/분할매수/홀딩/부분매도/비중 조절 매도/손절매 등]
        ''' or "이전 설정된 목표 정보가 없습니다. 새로운 목표를 설정하겠습니다."}
        
        2. 시장 통찰:
        - 현재 시장 국면: [횡보장/추세장/변동성장/전환점 등]
        - 주요 신호: [가장 중요한 차트 신호나 패턴]
        - 핵심 관찰: [종합적인 시장 해석과 균형 잡힌 판단]
        - 포착한 기회: [현재 시장 상황에서의 구체적이고 현실적인 기회]
        - 다음 실행 시간까지 예상: 
           * 현재 시간: {status_data['current_datetime']}
           * 다음 실행 시간: [1시간 후]
           * 단기 예측 (다음 1시간): [시간대별 상세 시장 흐름 예측과 주요 변곡점]
           * 주요 관찰 포인트: [다음 1시간 동안 주목해야 할 특정 가격대나 패턴]
           * 시간 민감 요소: [시간에 따라 영향이 달라질 수 있는 요소들]
        
        3. 전략과 실행:
        - 선택 전략: [현 시장 상황에 맞는 구체적 전략]
        - 거래 방향: [매수/매도/홀딩 + 비율]
        - 포지션 관리: [풀/분할/부분 실현 등 구체적 접근법]
        - 리스크 관리: [손절 계획, 비상 시나리오 대응]
        - 대안 계획: [시장이 예상과 다르게 움직일 경우]",
    "short_term_target": {{
        "price": <목표 가격>,
        "stop_loss": <손절가>,
        "target_time": "<목표 도달 예상 시간>",
        "expected_return": <예상 수익률 %>,
        "confidence": <신뢰도 1-100%>,
        "detail_reason": "<목표 설정과 전략적 비전>

        ====== 원래 매수 시점의 분석 ======
        [이전 매수 시점의 원래 분석 내용 - 절대 수정하지 말 것]

        ====== 현재 평가 ======
        [현재 시점에서의 평가:
        - 원래 분석의 유효성 평가
        - 변화된 시장 상황과 영향
        - 목표가/손절가 유지 또는 수정 이유
        - 추가적인 관찰 사항]

        매수 결정 시에는 '원래 매수 시점의 분석' 부분만 작성하고 '현재 평가' 부분은 생략합니다.
        매도 결정 시에는 새로운 매수 포인트를 위한 분석만 작성합니다.

        이 목표 가격과 거래 계획을 설정한 근거를 균형 잡힌 시각으로 설명합니다.
        차트 패턴, 기술적 지표, 경험적 판단, 시장 심리 등 여러 요소를 종합적으로
        활용하여 현실적인 계획을 제시하세요.

        선택한 전략(범위 거래, 추세 추종, 반전 포착 등)과 그 이유를 설명하고,
        포지션 관리 방식(비중 조절, 분할 매수/매도, 단계적 이익실현 등)의 근거도 함께 제시합니다.

        거래 실행의 구체적 계획과 함께, 시장이 예상과 다르게 움직일 경우의 대응 방안도
        사전에 설정하세요. 발키리는 계획적이면서도 유연합니다.

        균형 잡힌 자신감을 유지하세요. 당신의 분석과 판단을 믿되,
        시장의 불확실성을 인정하고 대비하세요. 계획을 세우고 적극적으로 실행하되,
        필요할 때는 유연하게 조정할 준비가 되어 있어야 합니다."
    }}
}}

발키리의 핵심 원칙을 기억하세요: 목표가와 손절가는 신성합니다. 이를 위반하는 순간, 당신은 감정과 욕심에 거래를 맡기는 것입니다. 손절가에 도달하면 즉시 손절하고, 목표가에 도달하면 즉시 수익을 실현하세요. 

손절매는 실패가 아닌 자본 보존을 위한 필수적인 전략입니다. 손절 후에는 항상 새로운 거래 기회를 찾아야 합니다. 한 거래에 집착하면 더 큰 손실을 입고 다른 좋은 기회를 놓치게 됩니다. 성공적인 트레이더는 손절을 실행한 후 빠르게 새로운 거래 기회를 발굴하여 자산을 불립니다.

손절가를 무시하는 것은 탐욕이나 손실 회피 심리에서 비롯되는 가장 위험한 습관입니다. 한 거래에서의 손실은 전체 자산의 작은 부분일 뿐, 새로운 거래를 통해 충분히 회복할 수 있습니다. 여러 차례의 작은 손실은 감내할 수 있지만, 한 번의 큰 손실은 회복하기 어렵다는 점을 항상 명심하세요.

만약 손절가나 목표가를 준수하지 않는다면, 그 명확한 이유와 리스크 평가를 반드시 상세히 설명해야 합니다. 예외적 상황이 아니라면, 당신이 설정한 규칙을 존중하는 것이 성공적인 트레이더의 기본입니다!"""

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
                    {"role": "user", "content": """위 이미지는 4시간 차트입니다. 이 차트에서 발견되는 모든 주요 패턴, 장기 추세 방향과 강도, 볼린저 밴드 구조, ADX/DMI 신호, MA와 가격의 관계, RSI 흐름, 거래량 특성을 자세히 분석해 주세요."""},
                    
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['daily']}"}}
                    ]},
                    {"role": "user", "content": """위 이미지는 일간 차트입니다. 이 차트에서 발견되는 주요 장기 추세와 패턴, 볼린저 밴드 구조, ADX/DMI 신호, MA와 가격의 관계, RSI 상태, 거래량 추이를 분석하고 중장기적 시장 관점을 제시해 주세요."""},
                    
                    {"role": "user", "content": """이제 네 가지 시간대(5분, 1시간, 4시간, 일간) 차트를 모두 종합적으로 고려해, 현 시장 상황에 대한 통합적인 견해와 이를 바탕으로 한 매매 결정을 내려주세요. 단기, 중기, 장기 관점을 모두 고려하고, 매매 전략은 명확한 진입/탈출 조건, 리스크 관리 방안, 예상 목표가와 타임라인을 포함해야 합니다."""}
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
                                        "detail_reason": {
                                            "type": "string"
                                        }
                                    },
                                    "required": ["price", "stop_loss", "target_time", "expected_return", "confidence", "detail_reason"],
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

    # 주요 시장 시간대 - 뉴스 포함 실행 (하루 3번 유지)
    # 아시아/한국 시장 활동 시간
    schedule.every().day.at("09:05").do(execute_with_news)  

    # 유럽 시장 활발 / 미국 시장 개장 전
    schedule.every().day.at("17:05").do(execute_with_news)  

    # 미국 시장 가장 활발한 시간
    schedule.every().day.at("22:05").do(execute_with_news)  

    # 일반 실행 - 뉴스 없이 매시간 실행
    schedule.every().day.at("00:05").do(execute_without_news)
    schedule.every().day.at("01:05").do(execute_without_news)
    schedule.every().day.at("02:05").do(execute_without_news)
    schedule.every().day.at("03:05").do(execute_without_news)
    schedule.every().day.at("04:05").do(execute_without_news)
    schedule.every().day.at("05:05").do(execute_without_news)
    schedule.every().day.at("06:05").do(execute_without_news)
    schedule.every().day.at("07:05").do(execute_without_news)
    schedule.every().day.at("08:05").do(execute_without_news)
    # 09:05 - 뉴스 포함 실행으로 이미 설정됨
    schedule.every().day.at("10:05").do(execute_without_news)
    schedule.every().day.at("11:05").do(execute_without_news)
    schedule.every().day.at("12:05").do(execute_without_news)
    schedule.every().day.at("13:05").do(execute_without_news)
    schedule.every().day.at("14:05").do(execute_without_news)
    schedule.every().day.at("15:05").do(execute_without_news)
    schedule.every().day.at("16:05").do(execute_without_news)
    # 17:05 - 뉴스 포함 실행으로 이미 설정됨
    schedule.every().day.at("18:05").do(execute_without_news)
    schedule.every().day.at("19:05").do(execute_without_news)
    schedule.every().day.at("20:05").do(execute_without_news)
    schedule.every().day.at("21:05").do(execute_without_news)
    # 22:05 - 뉴스 포함 실행으로 이미 설정됨
    schedule.every().day.at("23:05").do(execute_without_news)

    schedule.every().day.at("08:42").do(execute_without_news)

    while True:
        schedule.run_pending()
        time.sleep(1)
