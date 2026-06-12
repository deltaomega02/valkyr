# 초기설정: 환경변수 로딩 및 필요한 라이브러리 임포트
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

# 초기설정: 로깅 및 API 클라이언트 초기화
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))

# ChromeDriver 생성 메서드: 환경 변수에 따라 로컬 또는 ec2용 드라이버 초기화
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


# DB 초기화 - 거래 패턴 테이블 생성 메서드
def initialize_pattern_db(db_path='ripple_trading_decisions.sqlite'):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                market_condition TEXT,
                price_action TEXT,
                outcome TEXT,
                profit_percentage REAL,
                success_factors TEXT,
                confidence_score REAL,
                occurrence_count INTEGER,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()


# DB 초기화 - decisions 테이블 생성 메서드
def initialize_db(db_path='ripple_trading_decisions.sqlite'):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                decision TEXT,
                percentage REAL,
                reason TEXT,
                xrp_balance REAL,
                krw_balance REAL,
                fee REAL,                     -- 거래 수수료
                settlement_amount REAL,       -- 정산금액
                xrp_avg_buy_price REAL,
                xrp_krw_price REAL,
                reflection TEXT,
                performance REAL
            );
        ''')
        conn.commit()

# 트레이딩 반영 분석 메서드: 주어진 반영 텍스트, 시장 데이터, 성과 데이터를 기반으로 트레이딩 패턴을 추출
def analyze_reflection_for_patterns(reflection_text, market_data, performance_data):
    """Analyze a reflection and extract trading patterns"""
    try:
        prompt = f"""
        Analyze this trading reflection and extract key performance patterns:
        
        Reflection: {reflection_text}
        Market Data: {market_data}
        Performance: {performance_data}
        
        Extract the following:
        1. Trading decision details
        2. Price movement patterns
        3. Performance metrics
        4. Success/failure factors
        
        Format response as:
        {{
            "pattern_type": "buy/sell/hold",
            "price_action": "price movement description",
            "outcome": "profit/loss",
            "profit_percentage": number,
            "identified_pattern": "pattern description"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a trading pattern analysis expert focusing on performance metrics and price movements."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error analyzing reflection: {e}")
        return None

# 패턴 DB 업데이트 메서드: 새로운 패턴 데이터를 DB에 업데이트하거나 삽입함
def update_pattern_database(pattern_data, db_path='ripple_trading_decisions.sqlite'):
    """Update the pattern database with new insights"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # 기존에 유사한 패턴이 있는지 확인
        cursor.execute('''
            SELECT id, confidence_score, occurrence_count, profit_percentage 
            FROM trading_patterns 
            WHERE pattern_type = ? 
            AND market_condition = ? 
            AND price_action = ?
        ''', (
            pattern_data['pattern_type'],
            pattern_data['market_condition'],
            pattern_data['price_action']
        ))
        
        existing_pattern = cursor.fetchone()
        
        if existing_pattern:
            # 기존 패턴이 있으면 업데이트
            pattern_id, current_confidence, occurrences, avg_profit = existing_pattern
            new_occurrences = occurrences + 1
            new_profit = ((avg_profit * occurrences) + pattern_data['profit_percentage']) / new_occurrences
            new_confidence = (current_confidence * occurrences + (1 if pattern_data['outcome'] == 'profit' else 0)) / new_occurrences
            
            cursor.execute('''
                UPDATE trading_patterns 
                SET confidence_score = ?,
                    occurrence_count = ?,
                    profit_percentage = ?,
                    last_updated = datetime('now')
                WHERE id = ?
            ''', (new_confidence, new_occurrences, new_profit, pattern_id))
        else:
            # 유사한 패턴이 없으면 새로운 패턴으로 삽입
            cursor.execute('''
                INSERT INTO trading_patterns (
                    pattern_type, market_condition, price_action,
                    outcome, profit_percentage, success_factors,
                    confidence_score, occurrence_count, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, datetime('now'))
            ''', (
                pattern_data['pattern_type'],
                pattern_data['market_condition'],
                pattern_data['price_action'],
                pattern_data['outcome'],
                pattern_data['profit_percentage'],
                pattern_data['success_factors'],
                1.0 if pattern_data['outcome'] == 'profit' else 0.0
            ))
        
        conn.commit()

# 패턴 인사이트 반환 메서드: 최근 30일간의 거래 데이터를 분석하여, 결정별 성과와 성공률 등 인사이트를 반환
def get_pattern_insights(current_conditions, db_path='ripple_trading_decisions.sqlite'):
    """Get relevant pattern insights based on historical performance"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
            WITH performance_stats AS (
                SELECT 
                    decision,
                    AVG(performance) as avg_performance,
                    COUNT(*) as occurrence_count,
                    AVG(CASE WHEN performance > 0 THEN 1 ELSE 0 END) as success_rate,
                    MAX(CASE WHEN performance > 0 THEN performance ELSE 0 END) as best_performance,
                    MIN(CASE WHEN performance < 0 THEN performance ELSE 0 END) as worst_performance
                FROM decisions
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY decision
                HAVING COUNT(*) >= 3
            )
            SELECT 
                decision,
                avg_performance,
                occurrence_count,
                success_rate,
                best_performance,
                worst_performance
            FROM performance_stats
            ORDER BY avg_performance * success_rate DESC
            LIMIT 5
        ''')
        
        patterns = cursor.fetchall()
        
        return {
            "patterns": [
                {
                    "type": p[0],
                    "avg_performance": p[1],
                    "occurrences": p[2],
                    "success_rate": p[3],
                    "best_case": p[4],
                    "worst_case": p[5]
                }
                for p in patterns
            ],
            "current_conditions": current_conditions
        }

# 성과 계산 메서드: 현재 상태와 이전 상태를 기반으로 매수/매도/홀딩에 따른 수익률을 계산
def calculate_performance(current_status, previous_status):
    """
    매수/매도/홀딩 각 케이스별로 수익률을 계산
    """
    try:
        # JSON 문자열을 딕셔너리로 변환
        current = json.loads(current_status)
        previous = json.loads(previous_status)
        
        # 현재 XRP 시장가 조회
        current_market_price = float(pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"])
        
        # 이전 시점의 XRP 가격 조회
        previous_xrp_price = float(previous['xrp_krw_price'])
        
        # 가장 최근 거래 내역을 기반으로 이전 결정 조회
        with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT decision 
                FROM decisions 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            last_decision = cursor.fetchone()
            last_decision = last_decision[0] if last_decision else 'hold'

        # XRP 가격 변동률 계산
        price_change = ((current_market_price - previous_xrp_price) / previous_xrp_price * 100)

        # 거래 유형에 따른 수익률 계산
        if last_decision == 'buy':
            # 매수의 경우: 매수 시점 대비 현재 가격의 변동률
            performance = price_change
            
        elif last_decision == 'sell':
            # 매도의 경우: 매도 시점 대비 현재 가격의 반대 방향 변동률
            performance = -price_change
            
        else:  # 'hold'
            # 홀딩의 경우: 
            if float(previous['xrp_balance']) > 0:
                # XRP를 보유한 상태라면 가격 변동률만큼 수익/손실
                performance = price_change
            else:
                # XRP를 보유하지 않은 상태라면 수익/손실 없음
                performance = 0
                
        # 소수점 둘째 자리까지 반올림하여 반환
        return round(performance, 2)
        
    except Exception as e:
        print(f"수익률 계산 중 오류 발생: {e}")
        return 0

# 거래 결정을 DB에 저장하고, 24시간 전 결정에 대해 reflection을 생성하는 메서드
def save_decision_to_db(decision, current_status):
    """
        현재 거래 결정을 저장하고,
        24시간 전 거래 결정에 대한 반성을 생성하는 함수
        
        매개변수:
            decision (dict): 거래 결정 데이터가 담긴 딕셔너리
            current_status (str): 현재 시장 상태를 나타내는 JSON 형식의 문자열
    """
    try:
        # 입력값 검증
        if not isinstance(decision, dict):
            raise ValueError("Decision must be a dictionary")
            
        required_keys = ['decision', 'percentage', 'reason']
        if not all(key in decision for key in required_keys):
            raise ValueError(f"Decision missing required keys: {required_keys}")

        db_path = 'ripple_trading_decisions.sqlite'
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 현재 상태 파싱 및 현재 가격 조회 (재시도 포함)
            try:
                if not current_status:
                    raise ValueError("Current status is empty")
                    
                status_dict = json.loads(current_status)
                if not isinstance(status_dict, dict):
                    raise ValueError("Invalid status format")

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        orderbook = pyupbit.get_orderbook(ticker="KRW-XRP")
                        if not orderbook or 'orderbook_units' not in orderbook:
                            raise ValueError("Invalid orderbook data")
                        current_price = float(orderbook['orderbook_units'][0]["ask_price"])
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(1)
            except Exception as e:
                print(f"Error parsing current status or getting current price: {e}")
                raise
            
            # 새로운 결정 DB에 삽입
            try:
                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Saving new decision at: {current_timestamp}")
                
                xrp_balance = float(status_dict.get('xrp_balance', 0))
                krw_balance = float(status_dict.get('krw_balance', 0))
                xrp_avg_buy_price = float(status_dict.get('xrp_avg_buy_price', 0))
                
                fee = float(decision.get('fee', 0))
                settlement_amount = float(decision.get('settlement_amount', 0))

                cursor.execute('''
                    INSERT INTO decisions (
                        timestamp, decision, percentage, reason, xrp_balance, krw_balance, 
                        fee, settlement_amount, xrp_avg_buy_price, xrp_krw_price, 
                        reflection, performance
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
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
                    current_price,
                    0  
                ))
                
                conn.commit()  
                
            except Exception as e:
                print(f"Error saving new decision: {e}")
                conn.rollback()
                raise
            
            # 24시간 전 결정에 대한 reflection 생성
            try:
                print("Searching for decisions needing reflection...")
                cursor.execute('''
                    SELECT id, timestamp, decision, percentage, reason, xrp_balance,
                           krw_balance, xrp_avg_buy_price, xrp_krw_price
                    FROM decisions 
                    WHERE timestamp BETWEEN datetime(?, '-24 hours', '-5 minutes') AND datetime(?, '-24 hours', '+5 minutes')
                    AND reflection IS NULL
                    ORDER BY timestamp DESC
                ''', (current_timestamp, current_timestamp))

                old_decisions = cursor.fetchall()
                old_decisions_count = len(old_decisions)
                print(f"Found {old_decisions_count} decisions needing reflection")

                if old_decisions:
                    for old_decision in old_decisions:
                        try:
                            if None in old_decision:
                                print(f"Invalid data found for decision ID {old_decision[0]}: Contains NULL values")
                                continue
                                
                            try:
                                xrp_balance = float(old_decision[5])
                                krw_balance = float(old_decision[6])
                                xrp_avg_buy_price = float(old_decision[7])
                                xrp_krw_price = float(old_decision[8])
                            except (ValueError, TypeError) as e:
                                print(f"Invalid numeric data for decision ID {old_decision[0]}: {e}")
                                continue
                            
                            old_status = json.dumps({
                                'xrp_balance': xrp_balance,
                                'krw_balance': krw_balance,
                                'xrp_avg_buy_price': xrp_avg_buy_price,
                                'xrp_krw_price': xrp_krw_price
                            })
                            
                            print(f"Generating reflection for decision ID: {old_decision[0]} from {old_decision[1]}")
                            
                            max_retries = 2
                            reflection_result = None
                            
                            for attempt in range(max_retries):
                                try:
                                    reflection_result = generate_reflection(old_decision[0], old_status)
                                    if reflection_result:
                                        break
                                except Exception as e:
                                    if attempt == max_retries - 1:
                                        print(f"All reflection generation attempts failed for decision ID {old_decision[0]}")
                                        break
                                    time.sleep(2)
                            
                            if reflection_result is None:
                                print(f"Reflection generation failed for decision ID {old_decision[0]}")
                                continue
                            
                            cursor.execute('''
                                UPDATE decisions 
                                SET reflection = ?,
                                    performance = ?
                                WHERE id = ?
                            ''', (reflection_result['text'], reflection_result['performance'], old_decision[0]))
                            
                            conn.commit()  
                            print(f"Successfully updated reflection for decision ID: {old_decision[0]}")
                            
                        except Exception as e:
                            print(f"Error processing reflection for decision ID {old_decision[0]}: {e}")
                            conn.rollback()
                            continue
                            
                else:
                    print("No decisions found in the 23-25 hour time window for reflection")
                    
            except Exception as e:
                print(f"Error processing old decisions: {e}")
                conn.rollback()
                raise
                
    except Exception as e:
        print(f"Critical error in save_decision_to_db: {e}")
        raise
    
    finally:
        if 'conn' in locals():
            conn.close()

# 최근 결정 내역을 불러오는 메서드
def fetch_last_decisions(db_path='ripple_trading_decisions.sqlite', num_decisions=10):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, decision, percentage, reason, xrp_balance, krw_balance, 
                   xrp_avg_buy_price, xrp_krw_price, reflection, performance 
            FROM decisions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (num_decisions,))
        decisions = cursor.fetchall()

        if decisions:
            formatted_decisions = []
            for decision in decisions:
                ts = datetime.strptime(decision[0], "%Y-%m-%d %H:%M:%S")
                ts_millis = int(ts.timestamp() * 1000)
                
                formatted_decision = {
                    "timestamp": ts_millis,
                    "decision": decision[1],
                    "percentage": decision[2],
                    "reason": decision[3],
                    "xrp_balance": decision[4],
                    "krw_balance": decision[5],
                    "xrp_avg_buy_price": decision[6],
                    "xrp_krw_price": decision[7],
                    "reflection": decision[8],
                    "performance": decision[9]
                }
                formatted_decisions.append(str(formatted_decision))
            return "\n".join(formatted_decisions)
        else:
            return "No decisions found."

# 기술적 지표 계산 메서드: RSI(Relative Strength Index)를 계산
def calculate_rsi(df, periods=14):
    close_delta = df['close'].diff()
    
    # 상승과 하락 변동 분리
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # 지수 가중 이동 평균(EWMA) 계산
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    return rsi

# 기술적 지표 계산 메서드: Bollinger Bands(상단, 중간, 하단) 계산
def calculate_bollinger_bands(df, window=20, dev=2):
    typical_p = (df['high'] + df['low'] + df['close']) / 3
    ma = typical_p.rolling(window=window).mean()
    std = typical_p.rolling(window=window).std()
    
    upper_band = ma + (std * dev)
    lower_band = ma - (std * dev)
    
    return upper_band, ma, lower_band

# 현재 상태 데이터 구성 메서드: 잔고, 시장 데이터, OHLCV 및 기술적 지표를 가져와 JSON 문자열로 반환
def get_current_status():
    try:
        # 기본 데이터: orderbook과 현재 시간 조회
        orderbook = pyupbit.get_orderbook(ticker="KRW-XRP")
        current_time = orderbook['timestamp']
        
        # 잔고 정보 초기화 및 조회 (XRP, KRW, 평균 매수가)
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

        # OHLCV 데이터 조회 및 기술적 지표 계산 (RSI, Bollinger Bands, 이동평균선, 거래량)
        df = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", count=200)
        rsi = calculate_rsi(df, 14)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, 20, 2)
        ma = df['close'].rolling(window=20).mean()
        volume_sma = df['volume'].rolling(window=24).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_sma.iloc[-1]

        # 현재 상태 데이터 구성 후 JSON 문자열로 반환
        current_status = {
            'current_time': current_time,
            'orderbook': orderbook,
            'xrp_balance': xrp_balance,
            'krw_balance': krw_balance,
            'xrp_avg_buy_price': xrp_avg_buy_price,
            'technical_indicators': {
                'rsi': float(rsi.iloc[-1]),
                'bollinger_bands': {
                    'upper': float(bb_upper.iloc[-1]),
                    'middle': float(bb_middle.iloc[-1]),
                    'lower': float(bb_lower.iloc[-1])
                },
                'moving_average': float(ma.iloc[-1]),
                'volume': {
                    'current': float(current_volume),
                    'average_24h': float(volume_sma.iloc[-1]),
                    'ratio': float(volume_ratio)
                }
            }
        }
        
        return json.dumps(current_status)
    except Exception as e:
        print(f"Error in get_current_status: {e}")
        return None

# XPath 요소 클릭 메서드: 주어진 XPath로 요소를 찾아 클릭
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

# 차트 설정 메서드 (1시간 차트)
def perform_chart_actions(driver):
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

# 차트 설정 메서드 (4시간 차트)
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

# 차트 설정 메서드 (5분 차트)
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
    # MACD (5분 차트에서 빠른 반전 신호용)
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

# 스크린샷 캡처 및 Base64 인코딩 메서드
def capture_and_encode_screenshot(driver):
    try:
        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()
        # PIL Image로 변환
        img = Image.open(io.BytesIO(png))
        # 이미지 크기 조절 (최대 2000x2000)
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

# 차트 데이터 준비 메서드: 5분, 1시간, 4시간 차트의 스크린샷과 데이터를 준비
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
        perform_chart_actions(driver)
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

# 최근 3개월 거래 데이터를 바탕으로 관련 패턴을 분석하는 메서드
def get_relevant_patterns(current_market_conditions):
    with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
        cursor = conn.cursor()
        
        # 최근 3개월 이내의 거래 패턴 분석
        cursor.execute('''
            WITH trades AS (
                SELECT 
                    decision,
                    performance,
                    timestamp,
                    julianday('now') - julianday(timestamp) as days_old
                FROM decisions
                WHERE timestamp >= date('now', '-90 days')
            )
            SELECT 
                decision,
                AVG(performance) as avg_performance,
                COUNT(*) as occurrence_count,
                AVG(CASE WHEN performance > 0 THEN 1 ELSE 0 END) as success_rate,
                MIN(days_old) as recency
            FROM trades
            GROUP BY decision
            HAVING COUNT(*) >= 2
            ORDER BY (avg_performance * success_rate * (1.0 / (1.0 + MIN(days_old)/30.0))) DESC
            LIMIT 5
        ''')
        return cursor.fetchall()

# 패턴 DB 유지 관리 메서드: 오래된 패턴 아카이브 및 성과가 떨어지는 패턴 재평가
def maintain_pattern_database():
    with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
        cursor = conn.cursor()
        
        # 오래된 패턴 아카이브: 90일 이상 업데이트되지 않은 패턴을 pattern_archive 테이블로 이동
        cursor.execute('''
            INSERT INTO pattern_archive
            SELECT *, datetime('now')
            FROM trading_patterns
            WHERE last_updated < date('now', '-90 days')
        ''')
        
        # 성과가 떨어지는 패턴 재평가: 30일 이상 업데이트된 패턴 중 최근 긍정적 성과가 없는 경우 신뢰도를 낮춤
        cursor.execute('''
            UPDATE trading_patterns
            SET confidence_score = confidence_score * 0.9
            WHERE last_updated < date('now', '-30 days')
            AND NOT EXISTS (
                SELECT 1 FROM trading_decisions
                WHERE trading_decisions.timestamp > date('now', '-30 days')
                AND trading_decisions.pattern_id = trading_patterns.id
                AND trading_decisions.performance > 0
            )
        ''')

# 패턴 가중치 계산 메서드: 신뢰도, 이익률, 시간 감쇠를 고려한 패턴 가중치를 계산함
def calculate_pattern_weight(confidence_score, profit_percentage, days_old):
    time_decay = 1.0 / (1.0 + days_old/30.0)  # 30일마다 가중치 감소
    return confidence_score * profit_percentage * time_decay

# 뉴스 데이터 가져오기 메서드: SerpApi를 사용하여 XRP 관련 뉴스를 가져오기기
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

# 공포와 탐욕 지수(Fear and Greed Index) 가져오기 메서드
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

# 현재 차트의 Base64 인코딩된 스크린샷 가져오기 메서드
def get_current_base64_image():
    try:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")

        service = Service('/usr/bin/chromedriver')  
        driver = webdriver.Chrome(service=service, options=chrome_options)

        try:
            driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-XRP")
            time.sleep(30)

            wait = WebDriverWait(driver, 10)
            first_menu_item = wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id='fullChartiq']/div/div/div[1]/div/div/cq-menu[1]")))
            first_menu_item.click()

            one_hour_option = wait.until(EC.element_to_be_clickable((By.XPATH, "//cq-item[@stxtap=\"Layout.setPeriodicity(1,60,'minute')\"]")))
            one_hour_option.click()

            indicators_menu_item = wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id='fullChartiq']/div/div/div[1]/div/div/cq-menu[3]")))
            indicators_menu_item.click()

            time.sleep(2)

            screenshot = driver.get_screenshot_as_png()
            return base64.b64encode(screenshot).decode('utf-8')
            
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"Error making current image: {e}")
        return ""

# 파일 경로에서 지침(instructions) 텍스트를 읽어오는 메서드
def get_instructions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred while reading the file:", e)

# 볼린저 밴드 가격 위치 분석 메서드
def get_bb_position(price, bb_data):
    """볼린저 밴드 상의 가격 위치 분석"""
    if price > bb_data['upper']:
        return "above upper band"
    elif price < bb_data['lower']:
        return "below lower band"
    else:
        relative_position = (price - bb_data['lower']) / (bb_data['upper'] - bb_data['lower']) * 100
        return f"at {relative_position:.1f}% of band range"

# 거래량 패턴 분석 메서드: volume_change 값을 기반으로 거래량 추세를 분류
def get_volume_pattern_description(volume_change):
    """거래량 패턴 분석"""
    if volume_change > 0.5:
        return "Significantly increasing volume trend"
    elif volume_change > 0.2:
        return "Moderately increasing volume"
    elif volume_change < -0.5:
        return "Significantly decreasing volume trend"
    elif volume_change < -0.2:
        return "Moderately decreasing volume"
    else:
        return "Relatively stable volume"

# 가격 위치 변화 분석 메서드: 이전 및 현재 볼린저 밴드 데이터를 기반으로 가격 위치 변화를 반환함
def get_price_position_analysis(old_bb, new_bb, old_price, new_price):
    """가격 위치 변화 분석"""
    old_position = get_bb_position(old_price, old_bb)
    new_position = get_bb_position(new_price, new_bb)
    return f"moved from {old_position} to {new_position}"

# 거래 결정 reflection 생성 메서드: 주어진 거래 결정 ID와 당시 상태 정보를 기반으로 reflection 및 성과 분석을 생성합니다.
def generate_reflection(decision_id, status_at_decision_time):
    """
    특정 거래 결정에 대한 반성을 생성하는 함수입니다.
    
    매개변수:
        decision_id: 분석할 거래 결정의 ID입니다.
        status_at_decision_time: 거래 결정 시점의 상태를 나타내는 JSON 형식의 문자열입니다.
    """
    try:
        # 입력 상태 데이터 검증
        if status_at_decision_time is None:
            print(f"No valid status data for decision ID: {decision_id}")
            return None
            
        try:
            decision_data = json.loads(status_at_decision_time)
            if not isinstance(decision_data, dict):
                print(f"Invalid status data format for decision ID: {decision_id}")
                return None
        except json.JSONDecodeError as e:
            print(f"Invalid JSON data for decision ID {decision_id}: {e}")
            return None

        # 1. 현재 XRP 시장가 조회
        try:
            current_price = float(pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"])
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error getting current price: {e}")
            return None
        
        # 2. 거래 결정 세부사항 및 주간 데이터 조회
        with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, decision, percentage, reason, xrp_balance,
                       krw_balance, xrp_avg_buy_price, xrp_krw_price
                FROM decisions
                WHERE id = ?
            ''', (decision_id,))
            
            decision_data = cursor.fetchone()
            if not decision_data:
                print(f"No decision found with ID: {decision_id}")
                return None
                
            if None in decision_data:
                print(f"Found null values in decision data for ID: {decision_id}")
                return None

            cursor.execute('''
                SELECT timestamp, decision, percentage, reason, xrp_balance,
                       krw_balance, xrp_avg_buy_price, xrp_krw_price, performance
                FROM decisions
                WHERE timestamp >= datetime(?, '-7 days')
                AND timestamp <= ?
                ORDER BY timestamp DESC
            ''', (decision_data[0], decision_data[0]))
            
            week_decisions = cursor.fetchall()

        # 3. 거래 결정 데이터 및 수치 값 검증
        try:
            timestamp, trade_decision, percentage, reason = decision_data[0:4]
            xrp_balance, krw_balance = decision_data[4:6]
            xrp_avg_price, xrp_price_at_time = decision_data[6:8]
            
            xrp_balance = float(xrp_balance)
            krw_balance = float(krw_balance)
            xrp_avg_price = float(xrp_avg_price)
            xrp_price_at_time = float(xrp_price_at_time)
        except (ValueError, TypeError) as e:
            print(f"Error parsing decision data values: {e}")
            return None
        
        # 거래 시점 및 현재 시점 기술적 지표 데이터 수집
        try:
            historical_df = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", to=timestamp, count=200)
            current_df = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", count=200)
            
            decision_time_indicators = {
                'rsi': calculate_rsi(historical_df).iloc[-1],
                'bollinger': {
                    'upper': calculate_bollinger_bands(historical_df)[0].iloc[-1],
                    'middle': calculate_bollinger_bands(historical_df)[1].iloc[-1],
                    'lower': calculate_bollinger_bands(historical_df)[2].iloc[-1]
                },
                'volume': {
                    'current': historical_df['volume'].iloc[-1],
                    'sma': historical_df['volume'].rolling(window=24).mean().iloc[-1]
                }
            }
            
            current_indicators = {
                'rsi': calculate_rsi(current_df).iloc[-1],
                'bollinger': {
                    'upper': calculate_bollinger_bands(current_df)[0].iloc[-1],
                    'middle': calculate_bollinger_bands(current_df)[1].iloc[-1],
                    'lower': calculate_bollinger_bands(current_df)[2].iloc[-1]
                },
                'volume': {
                    'current': current_df['volume'].iloc[-1],
                    'sma': current_df['volume'].rolling(window=24).mean().iloc[-1]
                }
            }
            
            indicator_changes = {
                'rsi_change': current_indicators['rsi'] - decision_time_indicators['rsi'],
                'bb_position_change': (current_price - current_indicators['bollinger']['middle']) - 
                                    (xrp_price_at_time - decision_time_indicators['bollinger']['middle']),
                'volume_ratio_change': (current_indicators['volume']['current'] / current_indicators['volume']['sma']) - 
                                    (decision_time_indicators['volume']['current'] / decision_time_indicators['volume']['sma'])
            }
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            indicator_changes = None

        # 4. 가격 및 포트폴리오 변화 계산
        try:
            if xrp_price_at_time <= 0:
                raise ValueError("Invalid initial price")
                
            price_change_pct = ((current_price - xrp_price_at_time) / xrp_price_at_time) * 100
            
            current_balances = upbit.get_balances()
            current_xrp = 0
            current_krw = 0
            for b in current_balances:
                if b['currency'] == "XRP":
                    current_xrp = float(b.get('balance', 0))
                if b['currency'] == "KRW":
                    current_krw = float(b.get('balance', 0))
            
            previous_portfolio_value = xrp_balance * xrp_price_at_time + krw_balance
            if previous_portfolio_value <= 0:
                raise ValueError("Invalid previous portfolio value")
                
            current_portfolio_value = current_xrp * current_price + current_krw
            portfolio_change_pct = ((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value) * 100
            
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error calculating changes: {e}")
            return None

        # 5. 거래 결과 분석 및 결과 산출
        if trade_decision not in ['buy', 'sell', 'hold']:
            print(f"Invalid trade decision type: {trade_decision}")
            return None
            
        if trade_decision == 'buy':
            if price_change_pct > 0:
                result = f"Price increased by {price_change_pct:.2f}% after purchase. Portfolio value changed by {portfolio_change_pct:.2f}%."
                outcome = 'profit'
            else:
                result = f"Price decreased by {abs(price_change_pct):.2f}% after purchase. Portfolio value changed by {portfolio_change_pct:.2f}%."
                outcome = 'loss'
        elif trade_decision == 'sell':
            if price_change_pct < 0:
                result = f"Price decreased by {abs(price_change_pct):.2f}% after sale. Good timing. Portfolio value changed by {portfolio_change_pct:.2f}%."
                outcome = 'profit'
            else:
                result = f"Price increased by {price_change_pct:.2f}% after sale. Missed opportunity for higher sale. Portfolio value changed by {portfolio_change_pct:.2f}%."
                outcome = 'loss'
        else:  # hold
            result = f"Price changed by {price_change_pct:.2f}% during holding period. Portfolio value changed by {portfolio_change_pct:.2f}%."
            outcome = 'profit' if portfolio_change_pct > 0 else 'loss'

        # 6. 주간 성과 지표 계산
        try:
            total_performance = sum(decision[8] for decision in week_decisions if decision[8] is not None)
            successful_trades = len([d for d in week_decisions if d[8] is not None and d[8] > 0])
            total_trades = len([d for d in week_decisions if d[8] is not None])
            success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        except Exception as e:
            print(f"Error calculating weekly metrics: {e}")
            return None

        # 7. 포트폴리오 통계 계산
        weekly_stats = {
            'total_performance': sum(decision[8] for decision in week_decisions if decision[8] is not None),
            'trades_count': len(week_decisions),
            'successful_trades': len([d for d in week_decisions if d[8] is not None and d[8] > 0]),
            'buy_count': len([d for d in week_decisions if d[1] == 'buy']),
            'sell_count': len([d for d in week_decisions if d[1] == 'sell']),
            'avg_profit': statistics.mean([d[8] for d in week_decisions if d[8] is not None and d[8] > 0] or [0]),
            'avg_loss': statistics.mean([d[8] for d in week_decisions if d[8] is not None and d[8] < 0] or [0])
        }

        # 8. 시장 심리 데이터 수집
        try:
            fear_greed = fetch_fear_and_greed_index(limit=1)
            orderbook = pyupbit.get_orderbook(ticker="KRW-XRP")
            if not orderbook or 'orderbook_units' not in orderbook:
                raise ValueError("Invalid orderbook data")
                
            total_bid_size = sum(unit['bid_size'] for unit in orderbook['orderbook_units'])
            total_ask_size = sum(unit['ask_size'] for unit in orderbook['orderbook_units'])
            if total_bid_size + total_ask_size == 0:
                raise ValueError("Invalid orderbook sizes")
                
            order_imbalance = (total_bid_size - total_ask_size) / (total_bid_size + total_ask_size)
        except Exception as e:
            print(f"Error calculating market sentiment: {e}")
            return None

        # 9. GPT를 사용한 reflection 생성
        try:
            reflection_prompt = f"""
            Generate a comprehensive cryptocurrency trading analysis using the following data:

            1. Weekly Performance Overview:
            - Total Performance: {weekly_stats['total_performance']:.2f}%
            - Number of Trades: {weekly_stats['trades_count']}
            - Success Rate: {success_rate:.2f}%
            - Average Profit: {weekly_stats['avg_profit']:.2f}%
            - Average Loss: {weekly_stats['avg_loss']:.2f}%
            
            2. Recent Trading Activity:
            - Buy Trades: {weekly_stats['buy_count']}
            - Sell Trades: {weekly_stats['sell_count']}
            - Latest Decision: {trade_decision.upper()} ({percentage}%)
            - Result: {result}
            
            3. Portfolio Status:
            - Initial Portfolio Value: {previous_portfolio_value:,.0f} KRW
            - Current Portfolio Value: {current_portfolio_value:,.0f} KRW
            - Portfolio Change: {portfolio_change_pct:.2f}%
            - XRP Price Change: {price_change_pct:.2f}%
            - Initial Price: {xrp_price_at_time:,.0f} KRW
            - Current Price: {current_price:,.0f} KRW
            
            4. Market Context:
            - Fear & Greed Index: {fear_greed}
            - Order Book Imbalance: {order_imbalance:.2f}
            - Total Bid Size: {total_bid_size:.2f} XRP
            - Total Ask Size: {total_ask_size:.2f} XRP

            5. Technical Indicators Comparison:
            - Decision Time:
            - RSI: {decision_time_indicators['rsi']:.2f}
            - BB Position: Price was {get_bb_position(xrp_price_at_time, decision_time_indicators['bollinger'])}
            - Volume vs SMA: {(decision_time_indicators['volume']['current'] / decision_time_indicators['volume']['sma'] - 1) * 100:.2f}% from average

            - Current Time:
            - RSI: {current_indicators['rsi']:.2f}
            - BB Position: Price is {get_bb_position(current_price, current_indicators['bollinger'])}
            - Volume vs SMA: {(current_indicators['volume']['current'] / current_indicators['volume']['sma'] - 1) * 100:.2f}% from average

            - Indicator Changes:
            - RSI Change: {indicator_changes['rsi_change']:.2f}
            - BB Position Change: {indicator_changes['bb_position_change']:.2f}
            - Volume Pattern Change: {indicator_changes['volume_ratio_change']:.2f}
            
            Original Trade Reason:
            {reason}

            Analyze and provide insights in the following structure:
            1. Brief Reflection on Recent Trading Decisions:
            - Weekly performance evaluation
            - Analysis of this specific trade
            - Key factors that influenced the outcome

            2. Insights on What Worked Well and What Didn't:
            - Successful strategies and decisions
            - Areas where improvement is needed
            - Missed opportunities or overlooked signals

            3. Portfolio and Risk Assessment:
            - Portfolio balance evaluation
            - Risk management effectiveness
            - Position sizing analysis

            4. Market Analysis and Future Outlook:
            - Current market sentiment assessment
            - Order book analysis
            - Short-term market direction prediction

            Format Requirements:
            - Use markdown formatting
            - Highlight key metrics with bold (**number**)
            - Response must be in Korean
            - Keep concise (approximately 250 words)
            """

            reflection_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency trading analyst specializing in performance analysis and portfolio management. Focus on actual trading results, risk management, and portfolio optimization. Your analysis should follow this structure: '**1. Brief Reflection on Recent Trading Decisions:** [Insert your analysis of the actual performance data provided]'"
                    },
                    {
                        "role": "user",
                        "content": reflection_prompt
                    }
                ]
            )
            
            if not reflection_response.choices or not reflection_response.choices[0].message:
                raise ValueError("Invalid GPT-4 response")

        except Exception as e:
            print(f"Error generating reflection response: {e}")
            return None

        # 10. 패턴 분석 생성 및 DB 업데이트
        try:
            pattern_prompt = f"""
            Analyze this trading event and extract key patterns:

            Trade Decision:
            - Type: {trade_decision}
            - Portfolio Change: {portfolio_change_pct:.2f}%
            - Success Rate: {success_rate:.2f}%
            
            Market Context:
            - Price Movement: {price_change_pct:.2f}%
            - Order Imbalance: {order_imbalance:.2f}
            - Fear & Greed: {fear_greed}
            
            Weekly Performance:
            - Total Performance: {weekly_stats['total_performance']:.2f}%
            - Success Count: {weekly_stats['successful_trades']} of {weekly_stats['trades_count']}

            Technical Context:
            - RSI Movement: {decision_time_indicators['rsi']:.2f} -> {current_indicators['rsi']:.2f}
            - Volume Pattern: {get_volume_pattern_description(indicator_changes['volume_ratio_change'])}
            - Price Position: {get_price_position_analysis(
                decision_time_indicators['bollinger'],
                current_indicators['bollinger'],
                xrp_price_at_time,
                current_price
            )}

            Provide analysis in JSON format:
            {{
                "pattern_type": "{trade_decision}",
                "market_condition": "description of market state",
                "price_action": "price movement pattern",
                "outcome": "{outcome}",
                "profit_percentage": {portfolio_change_pct},
                "success_factors": "key elements that contributed to success/failure",
                "identified_pattern": "pattern description for future reference"
            }}

            Focus on:
            1. Market condition patterns that led to this outcome
            2. Success/failure factors analysis
            3. Pattern reliability for future trades
            4. Key indicators that predicted this outcome
            """

            pattern_response = client.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {
                        "role": "system",
                        "content": "You are a trading pattern analysis specialist focusing on market conditions and outcome correlation."
                    },
                    {
                        "role": "user",
                        "content": pattern_prompt
                    }
                ],
                response_format={"type": "json_object"}
            )

            if not pattern_response.choices or not pattern_response.choices[0].message:
                raise ValueError("Invalid pattern analysis response")

            pattern_data = json.loads(pattern_response.choices[0].message.content)
            if not all(key in pattern_data for key in [
                'pattern_type', 
                'market_condition', 
                'price_action', 
                'outcome', 
                'profit_percentage', 
                'success_factors', 
                'identified_pattern'
            ]):
                raise ValueError("Invalid pattern data structure")

            update_pattern_database(pattern_data)

        except Exception as e:
            print(f"Error generating pattern analysis: {e}")
            return None

        # 11. reflection 및 성과 반환
        return {
            'text': reflection_response.choices[0].message.content,
            'performance': price_change_pct if trade_decision != 'sell' else -price_change_pct
        }

    except Exception as e:
        print(f"Unexpected error in generate_reflection: {e}")
        return None

# 거래 성과 계산 메서드: 거래 당시 가격, 현재 가격, 거래 유형을 기반으로 수익률(%)을 계산
def calculate_trade_performance(decision_price, current_price, decision_type):
    """
    거래 성과를 계산함
    
    Args:
        decision_price (float): 거래 당시의 가격
        current_price (float): 현재 가격
        decision_type (str): 거래 유형 ('buy', 'sell', 'hold')
    
    Returns:
        float: 수익률 (%)
    """
    try:
        if decision_price <= 0 or current_price <= 0:
            return 0.0
            
        price_change_pct = ((current_price - decision_price) / decision_price) * 100
        
        # 매수의 경우: 가격 상승이 이익
        if decision_type == 'buy':
            return price_change_pct
            
        # 매도의 경우: 가격 하락이 이익 (반대 방향)
        elif decision_type == 'sell':
            return -price_change_pct
            
        # 홀딩의 경우: 가격 변동 그대로 반영
        else:  # hold
            return price_change_pct
            
    except Exception as e:
        print(f"Error calculating trade performance: {e}")
        return 0.0

# 24시간 이상 지난 거래들의 성과를 업데이트하는 메서드
def update_performance_for_old_decisions():
    """
    24시간이 지난 거래들의 성과를 업데이트합니다.
    """
    try:
        with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
            cursor = conn.cursor()
            
            # 24시간 이상 지난, 성과가 업데이트되지 않은 거래들 조회
            cursor.execute('''
                SELECT id, decision, xrp_krw_price
                FROM decisions
                WHERE timestamp <= datetime('now', '-24 hours')
                AND (performance IS NULL OR performance = 0)
            ''')
            
            old_decisions = cursor.fetchall()
            if not old_decisions:
                print("No decisions to update")
                return
                
            # 현재 XRP 가격 조회
            current_price = float(pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"])
            
            # 각 거래에 대해 성과 업데이트
            for decision_id, decision_type, decision_price in old_decisions:
                try:
                    performance = calculate_trade_performance(
                        float(decision_price),
                        current_price,
                        decision_type
                    )
                    
                    cursor.execute('''
                        UPDATE decisions
                        SET performance = ?
                        WHERE id = ?
                    ''', (performance, decision_id))
                    
                    print(f"Updated performance for decision {decision_id}: {performance:.2f}%")
                    
                except Exception as e:
                    print(f"Error updating decision {decision_id}: {e}")
                    continue
                    
            conn.commit()
            
    except Exception as e:
        print(f"Error in update_performance_for_old_decisions: {e}")

# 최근 거래 내역을 가져오는 메서드
def get_recent_trading_history(cursor, limit=5):
    """최근 거래 내역을 가져오는 함수
    
    Args:
        cursor: SQLite cursor
        limit: 가져올 거래 수 (기본값: 5)
        
    Returns:
        List of dicts containing trading decisions and reasons
    """
    cursor.execute('''
        SELECT 
            decision,
            reason
        FROM decisions 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    trading_history = []
    
    for row in results:
        trading_history.append({
            'decision': row[0],
            'reason': row[1]
        })
    
    return trading_history

# GPT를 사용해 데이터 분석 및 거래 결정 생성 메서드
def analyze_data_with_gpt4(news_data, last_decisions, fear_and_greed, current_status, chart_images):
    #instructions_path = "instructions_vk.md"
    max_retries = 3
    
    # 현재 XRP 보유 기준 수수료 계산 메서드
    def calculate_relevant_fees(current_xrp, avg_buy_price, current_price):
        try:
            # XRP 보유량이 0이거나 가격이 0인 경우 처리
            if current_xrp <= 0 or current_price <= 0:
                return {
                    'total_fees': 0,
                    'required_profit_percentage': 0
                }
                
            buy_fees = current_xrp * avg_buy_price * 0.0005  # 매수시 발생한 수수료
            potential_sell_fees = current_xrp * current_price * 0.0005  # 매도시 발생할 수수료
            total_fees = buy_fees + potential_sell_fees
            
            # 0으로 나누기 방지
            total_value = current_xrp * current_price
            required_profit_percentage = (total_fees / total_value * 100) if total_value > 0 else 0
            
            return {
                'total_fees': total_fees,
                'required_profit_percentage': required_profit_percentage
            }
        except Exception as e:
            print(f"Error calculating fees: {e}")
            return {
                'total_fees': 0,
                'required_profit_percentage': 0
            }

    # 성공적인 거래 패턴 분석 메서드
    def analyze_successful_patterns():
        try:
            with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    WITH profit_trades AS (
                        SELECT 
                            d1.decision,
                            d1.xrp_krw_price as entry_price,
                            d2.xrp_krw_price as exit_price,
                            ((d2.xrp_krw_price - d1.xrp_krw_price) / d1.xrp_krw_price * 100) as profit,
                            d1.performance
                        FROM decisions d1
                        JOIN decisions d2 ON d2.timestamp > d1.timestamp
                        WHERE d1.decision IN ('buy', 'sell')
                        AND ((d1.decision = 'buy' AND d2.decision = 'sell') OR 
                            (d1.decision = 'sell' AND d2.decision = 'buy'))
                        AND d2.timestamp = (
                            SELECT MIN(timestamp)
                            FROM decisions d3
                            WHERE d3.timestamp > d1.timestamp
                            AND ((d1.decision = 'buy' AND d3.decision = 'sell') OR 
                                (d1.decision = 'sell' AND d3.decision = 'buy'))
                        )
                    )
                    SELECT 
                        decision,
                        AVG(profit) as avg_profit,
                        COUNT(*) as frequency,
                        AVG(performance) as avg_performance
                    FROM profit_trades
                    GROUP BY decision
                    ORDER BY avg_profit DESC
                ''')
                return cursor.fetchall()
        except Exception as e:
            print(f"Error in analyze_successful_patterns: {e}")
            return []
    
    # 과거 성공 패턴에 따른 가중치 계산 메서드
    def calculate_success_weights():
        success_patterns = analyze_successful_patterns()
        weights = {
            'decision_types': {'buy': 1.0, 'sell': 1.0, 'hold': 1.0}  # 기본값
        }
        
        if not success_patterns:  # 패턴이 없으면 기본값 반환
            return weights
            
        try:
            for pattern in success_patterns:
                decision, avg_profit, freq, performance = pattern
                weight = (avg_profit * freq * (1 + performance)) / 100  # 수익률, 빈도, 성과를 모두 고려
                weights['decision_types'][decision] = weight if weight > 1.0 else 1.0  # 최소 가중치 보장
                
        except Exception as e:
            print(f"Error in calculate_success_weights: {e}")
            return weights
            
        return weights

    success_weights = {'decision_types': {'buy': 1.0, 'sell': 1.0, 'hold': 1.0}}
    for attempt in range(max_retries):
        try:
            # instructions = get_instructions(instructions_path)
            # if not instructions:
            #     print("No instructions found.")
            #     return None

            # 시장 상태 분석
            with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
                cursor = conn.cursor()

                # 최근 거래 내역 가져오기
                trading_history = get_recent_trading_history(cursor)

                # 최근 매수 정보 분석
                cursor.execute('''
                    SELECT xrp_krw_price, performance
                    FROM decisions
                    WHERE decision = 'buy'
                    AND timestamp >= datetime('now', '-48 hours')
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''')
                last_buy = cursor.fetchone()
                
                # 거래 패턴 분석
                cursor.execute('''
                    SELECT decision, COUNT(*) as count
                    FROM decisions
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY decision
                ''')
                recent_decisions = dict(cursor.fetchall())

            status_data = json.loads(current_status)
            current_price = float(status_data['orderbook']['orderbook_units'][0]['ask_price'])
            xrp_balance = float(status_data['xrp_balance'])
            krw_balance = float(status_data['krw_balance'])
            xrp_value = xrp_balance * current_price
            fees_info = calculate_relevant_fees(xrp_balance, float(status_data['xrp_avg_buy_price']), current_price)

            # 평균 매수가 기준으로 수익률 계산
            profit_percentage = ((current_price - float(status_data['xrp_avg_buy_price'])) / float(status_data['xrp_avg_buy_price']) * 100) if float(status_data['xrp_avg_buy_price']) > 0 else 0
            success_weights = calculate_success_weights()

            profit_since_buy = ((current_price - float(last_buy[0])) / float(last_buy[0]) * 100) if last_buy and last_buy[0] else 0
            success_weights = calculate_success_weights()
            
            trading_history_info = f"""
            Recent Trading History:
            Last 5 Trading Decisions:
            """
            for idx, trade in enumerate(trading_history, 1):
                trading_history_info += f"""
            Trade {idx}:
            - Decision: {trade['decision']}
            - Reason: {trade['reason']}
            """

            system_prompt = f"""전략적 암호화폐 트레이더: SMART ALPHA

핵심 원칙:
- 당신은 암호화폐 시장에서 기회를 포착하는 전략적 트레이더입니다
- 수익성 있는 거래와, 위험 관리 사이의 균형을 유지합니다
- 모든 시장 상황에서 효율적으로 수익을 창출합니다
- 다양한 전략을 활용하여 여러 시장 상황에 적응합니다
- 감정보다는 데이터에 근거한 결정을 내립니다
- 하락장은 손실의 원인이 아닌 새로운 수익 기회로 인식합니다

거래 전략 및 수익 목표:
1. 다양한 진입 전략:
   - 분할 매수 (DCA): 가격 하락 시 단계적으로 추가 매수하여 평단가 낮추기
   - 추세 추종: 강한 추세 확인 시 적극적 진입
   - 반등 매수: 과매도 구간에서 기술적 반등 포착
   - 브레이크아웃 트레이딩: 주요 저항/지지선 돌파 시 진입

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
   - 항상 0.1% 수수료를 타겟에 반영

4. 손절 및 위험 관리:
   - 명확한 손절선 설정: 진입가 대비 1.5-3% (시장 변동성에 따라 조정)
   - 포지션 크기 조절: 시장 불확실성에 비례하여 진입 규모 조정
   - 포트폴리오 분산: 전체 자금의 50-70%만 활용
   - 하락 트렌드 대응: 명확한 하락세 확인 시 부분 매도로 현금 확보 후 더 낮은 가격에 재진입 계획
   - 단계적 손절 전략: 손실 -0.7% 도달 시 포지션 20% 매도, -1.5% 도달 시 추가 30% 매도
   - 손절 후 재진입: 손절은 실패가 아닌 더 좋은 진입점을 위한 전략적 선택

5. 트렌드 변화 인식 요소:
   - 하락 트렌드 신호: 연속적인 하락 캔들, 주요 이동평균선 하향 돌파, 거래량 증가와 함께 하락
   - 상승 트렌드 신호: 연속적인 상승 캔들, 주요 이동평균선 상향 돌파, 거래량 증가와 함께 상승
   - 추세 전환점: 주요 지지/저항선 돌파, 다이버전스 출현, 패턴 완성(이중바닥/이중톱)
   - 하락 후 저점 매수 신호: RSI 30 이하 과매도, 강한 다이버전스, 주요 지지선 접근 시

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

3. 시장 상황별 접근법:
   - 상승장: 낙폭 시 적극적 매수, 부분 수익실현 후 재진입
   - 하락장:
    * 초기 하락 감지 시 30-40% 즉시 매도로 현금 확보
    * 분할 매도로 평균 매도가 최적화
    * 주요 지지선 도달 시 분할 매수 시작
    * 반등 확인 후 포지션 점진적 확대
    * 매도로 확보한 현금으로 저점에서 더 많은 물량 확보
   - 횡보장: 레인지 경계에서 스캘핑, 브레이크아웃 준비

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
      * 목표 달성 분석: [목표_달성_시_매도_우선]

4. 거래 매개변수:
   - 최소 거래 크기: 10,000 KRW
   - 최대 거래 비율: 가용 잔액의 90%
   - 거래 수수료: 거래당 0.05%
   - 거래 선택 규칙:
     * '매수', '매도', '홀딩'이 모두 가능한 경우: 시장 분석에 따라 하나 선택
     * '매수'와 '홀딩'만 가능한 경우: 시장 분석에 따라 둘 중 하나 선택
     * '매도'와 '홀딩'만 가능한 경우: 시장 분석에 따라 둘 중 하나 선택
     * '홀딩'만 가능한 경우: 반드시 홀딩 선택
     * 중요: 이전 가격 목표에 도달한 경우, '매도'를 강력히 고려

명령 형식:
{{
    "decision": "buy/sell/hold",
    "percentage": <1-90>,
    "reason": "SMART ALPHA 시장 분석:

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
        * 매도 가능 여부: {xrp_value} >= 10,000 KRW? [예/아니오]
        * 매수 가능 여부: {krw_balance} >= 10,000 KRW? [예/아니오]
        * 최종 가능한 작업목록: [최종_작업_목록]
        * 목표 달성 분석: [목표_달성_시_매도_우선]
    - 최종 결정:
        * 실행 계획: [매수/매도/홀딩] [X%]
        * 결정 근거: [간략한 핵심 이유]
    
    1. 시장 상황 분석
    - 현재 가격 추세와 강도
    - 주요 지지/저항 레벨
    - 기술적 지표 신호 (RSI, MACD, 이동평균선)
    - 거래량 분석
    - 다중 타임프레임 일관성 평가

    2. 전략적 접근
    - 현재 시장 상황에 적합한 전략
    - 진입/퇴출 근거
    - 위험/보상 비율 평가
    - 분할 매수/매도 계획
    - 반등/추가 하락 가능성 평가

    3. 실행 계획
    - 정확한 진입/퇴출 가격대
    - 포지션 크기 및 분할 계획
    - 목표 가격 레벨 (다중 목표)
    - 손절 레벨 (위험 관리)
    - 추가 시장 변화 대비 계획

    4. 수익/손실 분석: (매도 결정 시에만)
    - 현재 가격에서 매도 시 수익/손실 설명
    - 부분 매도 시 평균 매수가 영향
    - 매도 후 재진입 전략 (하락 추세 시)

    [데이터 기반 결정 + 전략적 실행]"
}}

시장 기회를 포착하고 적절한 위험 관리를 통해 최적의 수익을 창출하세요. 분할 매수, 평단가 관리, 적절한 수익 실현이 성공적인 거래의 핵심입니다.

1. 매수 결정 시 전략:
   - 가격 하락 시: 분할 매수로 평단가 낮추기 (20-30% 단위로 분할)
   - 지지선 확인 시: 반등 가능성이 높은 구간에서 적극적 매수
   - 상승 추세 확인 시: 추세 초기에 진입하여 상승 모멘텀 활용

2. 매도 결정 시 전략:
   - 목표가 도달 시: 부분 매도(30-50%)로 수익 확정
   - 저항선 접근 시: 추가 상승 한계 구간에서 일부 매도
   - 평단가 대비 수익 발생 시: 수수료를 고려한 최소 수익률(0.3-1%) 달성 시 매도 고려
   - 하락 추세 확인 시: 지체없이 포지션 30-40% 즉시 매도, 손실 -0.7% 도달 시 추가 20% 매도, -1.5% 도달 시 추가 30% 매도
   - 매도 후 현금 확보: 추가 하락 시 더 낮은 가격에 재진입 계획 수립

3. 홀딩 결정 시 전략:
   - 강한 상승 추세 중: 추가 상승 가능성이 높을 때
   - 매수/매도 시점이 불명확: 시장 방향성 확인 필요 시
   - 분할 매수 준비: 추가 하락에 대비한 자금 확보 시
   - 기술적 반등 신호 확인: RSI 과매도, 지지선 접근, 다이버전스 발생 시
   - 하락장 중 현금 비율: 30-50% 현금 유지로 저점 매수 여력 확보

시장 상황에 따른 결정 실행 시, 항상 가능한 작업 목록을 확인하고 최적의 전략을 선택하세요. 기술적 지표와 시장 상황을 종합적으로 분석하여 결정하고, 특히 하락 추세에서는 손절을 지체하지 말고 현금을 확보한 후 더 좋은 진입점을 준비하세요. 하락장은 손실의 원인이 아닌 더 좋은 진입점을 통한 수익 기회입니다."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
                        Analysis Data:
                        Fear and Greed: {fear_and_greed}
                        Recent History: {last_decisions}
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
                    - Execute quick exits when downtrends are confirmed to minimize losses
                    - Identify early reversal signals for re-entry at better prices
                    DO NOT miss any short-term profit chances!"""},
                    
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['1h']}"}}
                    ]},
                    {"role": "user", "content": """Above is the 1-hour timeframe chart. IMPERATIVE:
                    - Spot ALL trending moves
                    - Find optimal entry points for bigger positions
                    - Identify momentum shifts for maximum profit
                    - Detect early downtrend signals and exit positions proactively
                    - Map key support levels for re-entry during downtrends
                    TAKE ADVANTAGE of every clear trend!"""},
                    
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['4h']}"}}
                    ]},
                    {"role": "user", "content": """Above is the 4-hour timeframe chart. YOU MUST:
                    - Align trades with the dominant trend
                    - Size positions aggressively in strong trends
                    - Never miss major market moves
                    - Capitalize on both uptrends AND downtrends for maximum profits
                    - Reduce position size immediately when major downtrends are detected

                    CRITICAL TRADING DIRECTIVES:
                    1. AGGRESSIVE PROFIT TAKING & LOSS PREVENTION:
                    - Enter positions decisively on clear signals
                    - Take profits frequently but re-enter quickly
                    - Scale in aggressively on strength
                    - Never let fear prevent action
                    - Execute stop-losses without hesitation when downtrends are confirmed
                    - View stop-losses as strategic repositioning, not failures
                    - Immediately plan re-entries at lower levels after exiting positions

                    2. POSITION MANAGEMENT:
                    - Use multiple entries to build larger positions
                    - Lock in profits regularly but stay in the market
                    - Keep active positions at all times
                    - Scale up in strong trends
                    - Reduce position size during downtrends and increase cash reserves
                    - Use cash from exits to acquire larger positions at market bottoms
                    - Maintain 30-50% cash during downtrends for buying power

                    3. EXECUTION REQUIREMENTS:
                    - Act immediately on profitable setups
                    - Don't wait for perfect entries
                    - Take small profits over waiting for big ones
                    - Stay active in all market conditions
                    - Exit portions of positions at first sign of downtrends to minimize losses
                    - Implement staged exits to optimize average selling price
                    - Prepare for aggressive buying when RSI reaches oversold levels (below 30)

                    4. DOWNTREND ACTION PROTOCOL:
                    - Sell 30-40% of position immediately when downtrend is confirmed
                    - Sell additional 20% when losses reach -0.7%
                    - Sell another 30% when losses reach -1.5%
                    - Begin staged buying when key support levels are reached
                    - Gradually increase position size after reversal is confirmed
                    - Target 2-5% profits on swing trades during rebounds after downtrends

                    YOU MUST analyze all timeframes for MAXIMUM PROFIT POTENTIAL! 
                    This is not about analysis - this is about MAKING MONEY NOW in ANY MARKET DIRECTION!
                    Every second of hesitation is BURNING PROFIT!
                    PROVE YOUR TRADING MASTERY with decisive action in BOTH rising AND falling markets!"""}
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
                                }
                            },
                            "required": ["decision", "percentage", "reason"],
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
            print(f"Error in analyzing data with GPT-4 (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2)

    return None

def execute_buy(percentage):
    """매수 실행만 담당하고 실행 결과를 반환"""
    print("Attempting to buy XRP with a percentage of KRW balance...")
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
        print(f"Failed to execute buy order: {e}")
        return {
            "success": False,
            "error": str(e),
            "fee": 0,
            "settlement_amount": 0
        }

def execute_sell(percentage):
    """매도 실행만 담당하고 실행 결과를 반환"""
    print("Attempting to sell a percentage of XRP...")
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
        print(f"Failed to execute sell order: {e}")
        return {
            "success": False,
            "error": str(e),
            "fee": 0,
            "settlement_amount": 0
        }

def make_decision_and_execute(include_news=True):
    """거래 결정을 생성하고 실행합니다."""
    print("Making decision and executing...")
    try:
        # 데이터 수집
        news_data = get_news_data() if include_news else "No news data requested for this iteration"
        
        prepared_data = fetch_and_prepare_data()
        if prepared_data is None:
            print("Failed to prepare market data.")
            return
            
        chart_images = prepared_data['chart_images']
        last_decisions = fetch_last_decisions()
        fear_and_greed = fetch_fear_and_greed_index(limit=30)
        current_status = get_current_status()
        
        # 거래 결정 생성
        decision = analyze_data_with_gpt4(
            news_data, last_decisions, 
            fear_and_greed, current_status, chart_images
        )
        
        if not decision:
            print("Failed to make a decision.")
            return
            
        # 거래 실행
        execution_result = None
        percentage = decision.get('percentage', 100)

        if decision.get('decision') == "buy":
            execution_result = execute_buy(percentage)
        elif decision.get('decision') == "sell":
            execution_result = execute_sell(percentage)
        else:  # hold case
            execution_result = {
                "success": True,
                "fee": 0,
                "settlement_amount": 0
            }

        # 실행 결과 처리
        if execution_result and execution_result.get("success"):
            # decision 딕셔너리에 수수료와 정산금액 추가
            decision["fee"] = execution_result.get("fee", 0)
            decision["settlement_amount"] = execution_result.get("settlement_amount", 0)
            
            # DB에 저장
            save_decision_to_db(decision, current_status)
            
    except Exception as e:
        print(f"Error in make_decision_and_execute: {e}")

if __name__ == "__main__":
    initialize_db()
    initialize_pattern_db()
    
    def execute_with_news():
        make_decision_and_execute(include_news=True)
        
    def execute_without_news():
        make_decision_and_execute(include_news=False)

    # # 매시간 05분과 35분에 실행
    # for hour in range(24):
    #     schedule.every().day.at(f"{hour:02d}:05").do(execute_without_news)
    #     schedule.every().day.at(f"{hour:02d}:35").do(execute_without_news)
    


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
    schedule.every().day.at("02:41").do(execute_without_news)

    while True:
        schedule.run_pending()
        time.sleep(1)