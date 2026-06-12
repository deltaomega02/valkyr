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

            system_prompt = f"""전략적 암호화폐 트레이더: SMART ALPHA

핵심 원칙:
- 당신은 암호화폐 시장에서 기회를 포착하는 전략적 트레이더입니다
- 수익성 있는 거래와, 위험 관리 사이의 균형을 유지합니다
- 모든 시장 상황에서 효율적으로 수익을 창출합니다
- 다양한 전략을 활용하여 여러 시장 상황에 적응합니다
- 감정보다는 데이터에 근거한 결정을 내립니다

거래 전략 및 수익 목표:
1. 다양한 진입 전략:
   - 분할 매수 (DCA): 가격 하락 시 단계적으로 추가 매수하여 평단가 낮추기
   - 추세 추종: 강한 추세 확인 시 적극적 진입
   - 반등 매수: 과매도 구간에서 기술적 반등 포착
   - 브레이크아웃 트레이딩: 주요 저항/지지선 돌파 시 진입
   - 패턴 기반 트레이딩: 
     * 발목-어깨 전략: 주요 지지선("발목")에서 매수하고 저항선("어깨")에서 매도
     * 레인지 트레이딩: 횡보장에서 레인지 하단 20% 구간 매수, 상단 20% 구간 매도
     * 기술적 패턴: W/M 패턴, 헤드앤숄더, 삼각수렴/확산 등 차트 패턴 활용한 진입/퇴출

2. 수익 타겟 및 실행:
   - 거래 수수료: 매수 0.05% + 매도 0.05% = 총 0.1%
   - 최소 수익 목표: 0.3% (수수료 + 순이익)
   - 스캘핑 목표: 0.5-1.5% (단기 기회)
   - 스윙 트레이딩 목표: 2-5% (중기 추세)
   - 상황에 따라 부분 매도로 수익 실현 후 재진입

3. 위치 실행 최적화:
   - 분할 진입: 예산의 20-40% 초기 진입 + 상황에 따라 추가 매수
   - 평단가 관리: 하락장에서 전략적 추가 매수로 평단가 낮추기
   - 부분 수익실현: 목표가 도달 시 포지션의 20-50% 매도
   - 추세 지속 시 홀딩: 강한 상승추세에서는 일부 포지션 유지
   - 하락 선행 대응: 하락장 진입 신호 감지 시 포지션 일부/전체 청산 후 더 낮은 가격에 재진입
   - 항상 0.1% 수수료를 타겟에 반영

4. 손절 및 위험 관리:
   - 명확한 손절선 설정: 진입가 대비 1.5-3% (시장 변동성에 따라 조정)
   - 포지션 크기 조절: 시장 불확실성에 비례하여 진입 규모 조정
   - 포트폴리오 분산: 전체 자금의 50-70%만 활용
   - 역추세 매매 제한: 강한 하락 추세에서는 분할 매수 간격 확대

시장 분석 프레임워크:
1. 멀티타임프레임 분석:
   - 단기(15분-1시간): 진입/퇴출 타이밍
   - 중기(4시간-일봉): 추세 방향 확인
   - 장기(주봉-월봉): 주요 지지/저항 구간 식별

2. 기술적 지표 활용:
   - 가격 액션: 주요 지지/저항선, 차트 패턴
   - 모멘텀 지표: RSI, MACD, 스토캐스틱
   - 볼륨 분석: 가격 변동 확인을 위한 거래량 검증
   - 이동평균선: 20/50/200 EMA 교차 및 지지/저항
   - 패턴 식별 지표:
     * 발목(지지) 구간: RSI 과매도(30 이하), 볼린저밴드 하단, 주요 이동평균선 지지
     * 어깨(저항) 구간: RSI 과매수(70 이상), 볼린저밴드 상단, 주요 저항선/피보나치 레벨

3. 시장 상황별 접근법:
   - 상승장: 낙폭 시 적극적 매수, 부분 수익실현 후 재진입
   - 하락장: 분할 매수로 평단가 낮추기, 반등 시 일부 매도
   - 횡보장: 레인지 경계에서 스캘핑, 브레이크아웃 준비
   - 패턴 형성 시: 차트 패턴 완성도에 따라 진입/퇴출 타이밍 조정
   - 발목-어깨 구간: 주요 지지/저항 구간 도달 시 적극적 대응

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
     * 반드시 [최종_작업_목록]에 있는 작업 중 하나만 선택하여 실행

5. 단기 변동 필터링 및 중장기 관점:
   - 노이즈 필터링: 단기 가격 변동(1-3%)은 자연스러운 시장 노이즈로 간주
   - 중장기 추세 우선: 4시간/일봉 차트의 추세 방향에 역행하는 단기 신호는 신중하게 평가
   - 기회 기다리기: 단기 하락 후 반등이 예상되는 경우 즉시 매수보다 최적 진입점 기다리기
   - 시간 프리미엄: 단기 수익 0.3-1%보다 1-2일 기다려 2-5% 수익 기회 우선 고려
   - 가격 구간 평가: 단일 가격점보다 지지/저항 구간 개념으로 접근 (±1.5% 허용 범위)

6. 인내와 타이밍:
   - 단기 하락 인내: 중기 상승 추세에서 일시적 하락은 매도 시점이 아닌 추가 매수 기회로 평가
   - 최적 진입점 기다리기: 5-10% 하락 추세에서 즉시 매수보다 기술적 반등 신호 확인 후 진입
   - 추세 전환 확인: 주요 지지/저항선 돌파 시 즉시 반응보다 추가 1-2개 캔들 확인 후 결정
   - 현금 보유 가치: 불확실성 높은 구간에서 현금 보유는 하락 위험 회피 및 기회 대기 전략으로 가치 있음
   - 피라미딩 시점: 상승 추세 확정 후 작은 조정에서 추가 매수로 포지션 구축

명령 형식:
{{
    "decision": "buy/sell/hold",
    "percentage": <1-90>,
    "reason": "V.A.L.K.R 시장 분석:

    0. 현재 포지션 분석:
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
    
    1. 시장 상황 분석
    - 시간대별 추세 평가:
        * 단기(15분-1시간): [상승/하락/횡보] - 강도: [약/중/강]
        * 중기(4시간-일봉): [상승/하락/횡보] - 강도: [약/중/강]
        * 장기(주봉-월봉): [상승/하락/횡보] - 강도: [약/중/강]
    - 주요 지지/저항 레벨: [구체적인 가격대]
    - 기술적 지표 신호:
        * RSI(14): [과매수/중립/과매도] - 값: [수치]
        * MACD: [상승/하락/교차] - [설명]
        * 이동평균선: [골든크로스/데드크로스/지지/저항]
    - 거래량 분석: [증가/감소/평균] - [추세와의 일치 여부]
    - 패턴 식별: [발목-어깨/W형/M형/삼각수렴 등] - [완성도]

    2. 전략적 접근
    - 거래 시간 관점: [단기/중기/장기]
    - 목표 수익률: [스캘핑(0.5-1.5%)/스윙(2-5%)/중기(5-15%)]
    - 현재 시장 상황에 적합한 전략: [전략명] - [이유]
    - 진입/퇴출 근거: [기술적/패턴/레벨 기반]
    - 위험/보상 비율: [수치] - [계산 근거]
    - 분할 매수/매도 계획: [비율/간격]
    - 인내 필요성: [높음/중간/낮음] - [이유]

    3. 실행 계획
    - 정확한 진입/퇴출 가격대: [가격 또는 범위]
    - 포지션 크기 및 분할 계획: [비율 및 방법]
    - 목표 가격 레벨 (다중 목표):
        * 1차 목표: [가격]
        * 2차 목표: [가격]
    - 손절 레벨 : [가격]
    - 예상 보유 기간: [시간/일/주]

    4. 수익/손실 분석: (매도 결정 시에만)
    - 현재 가격에서 매도 시 수익/손실: [금액]
    - 부분 매도 시 평균 매수가 영향: [설명]
    - 수수료 영향: [계산]

    [데이터 기반 결정 + 전략적 실행]"
}}

시장 기회를 포착하고 적절한 위험 관리를 통해 최적의 수익을 창출하세요. 분할 매수, 평단가 관리, 적절한 수익 실현이 성공적인 거래의 핵심입니다.

1. 매수 결정 시 전략:
   - 가격 하락 시: 분할 매수로 평단가 낮추기 (20-30% 단위로 분할)
   - 지지선 확인 시: 반등 가능성이 높은 구간에서 적극적 매수
   - 상승 추세 확인 시: 추세 초기에 진입하여 상승 모멘텀 활용
   - 발목 구간 진입 시: 과거 데이터에서 확인된 강한 지지선, RSI 과매도, 볼링거밴드 하단 접촉 시 분할 매수
   - 패턴 형성 시: W형 바닥, 삼각수렴 상승 브레이크아웃 등 확인 시 진입

2. 매도 결정 시 전략:
   - 목표가 도달 시: 부분 매도(30-50%)로 수익 확정
   - 저항선 접근 시: 추가 상승 한계 구간에서 일부 매도
   - 평단가 대비 수익 발생 시: 수수료를 고려한 최소 수익률(0.3-1%) 달성 시 매도 고려
   - 어깨 구간 도달 시: RSI 과매수, 주요 저항선/피보나치 레벨 접근, 거래량 감소 시 매도
   - 패턴 붕괴 신호: 상승 패턴 무효화, M형 상단, 하락 패턴 완성 시 선제적 매도

3. 홀딩 결정 시 전략:
   - 강한 상승 추세 중: 추가 상승 가능성이 높을 때
   - 매수/매도 시점이 불명확: 시장 방향성 확인 필요 시
   - 분할 매수 준비: 추가 하락에 대비한 자금 확보 시
   - 패턴 형성 진행 중: 차트 패턴이 완성되지 않고 발전 중일 때 (삼각수렴, W/M 패턴 등)
   - 발목-어깨 사이 구간: 가격이 발목(지지)과 어깨(저항) 사이에서 움직일 때 추세 확인
   - 브레이크아웃 대기: 중요 레벨 접근 시 브레이크아웃/브레이크다운 확인 전까지 관망

시장 상황에 따른 결정 실행 시, 항상 가능한 작업 목록을 확인하고 최적의 전략을 선택하세요."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
                        Analysis Data:
                        Fear and Greed: {fear_and_greed}
                        Current Status: {current_status}
                        News Data: {news_data}
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
                                        }
                                    },
                                    "required": ["price", "stop_loss", "target_time", "expected_return", "confidence"],
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
    schedule.every().day.at("00:13").do(execute_without_news)


    while True:
        schedule.run_pending()
        time.sleep(1)
