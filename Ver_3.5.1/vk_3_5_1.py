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

def initialize_db(db_path='ripple_trading_decisions.sqlite'):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # 기존 decisions 테이블
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

def update_pattern_database(pattern_data, db_path='ripple_trading_decisions.sqlite'):
    """Update the pattern database with new insights"""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check if similar pattern exists
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
            # Update existing pattern
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
            # Insert new pattern
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

def calculate_performance(current_status, previous_status):
    """
    매수/매도/홀딩 각 케이스별로 수익률을 계산합니다.
    """
    try:
        # JSON 문자열을 딕셔너리로 변환
        current = json.loads(current_status)
        previous = json.loads(previous_status)
        
        # 현재 XRP 시장가 조회
        current_market_price = float(pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"])
        
        # 이전 시점의 XRP 가격
        previous_xrp_price = float(previous['xrp_krw_price'])
        
        # 이전 결정 조회 (가장 최근 거래 내역)
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

        # 거래 유형별 수익률 계산
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
                
        # 소수점 둘째 자리까지 반올림
        return round(performance, 2)
        
    except Exception as e:
        print(f"수익률 계산 중 오류 발생: {e}")
        return 0

def save_decision_to_db(decision, current_status):
    """
    Save current decision and generate reflection for 24-hour old decision
    
    Args:
        decision (dict): Trading decision data
        current_status (str): JSON string of current market status
    """
    try:
        # Validate input parameters
        if not isinstance(decision, dict):
            raise ValueError("Decision must be a dictionary")
            
        required_keys = ['decision', 'percentage', 'reason']
        if not all(key in decision for key in required_keys):
            raise ValueError(f"Decision missing required keys: {required_keys}")

        db_path = 'ripple_trading_decisions.sqlite'
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Parse and validate current status
            try:
                if not current_status:
                    raise ValueError("Current status is empty")
                    
                status_dict = json.loads(current_status)
                if not isinstance(status_dict, dict):
                    raise ValueError("Invalid status format")
                    
                # Get current price with retry mechanism
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
            
            try:
                # Format timestamp
                current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Saving new decision at: {current_timestamp}")
                
                # Prepare and validate data for insertion
                xrp_balance = float(status_dict.get('xrp_balance', 0))
                krw_balance = float(status_dict.get('krw_balance', 0))
                xrp_avg_buy_price = float(status_dict.get('xrp_avg_buy_price', 0))
                
                # Get fee and settlement_amount from decision if available, otherwise use 0
                fee = float(decision.get('fee', 0))
                settlement_amount = float(decision.get('settlement_amount', 0))
                
                # Insert new decision with transaction
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
                    fee,                   # 새로 추가된 필드
                    settlement_amount,      # 새로 추가된 필드
                    xrp_avg_buy_price,
                    current_price,
                    0  # initial performance
                ))
                
                conn.commit()  # Commit the insertion
                
            except Exception as e:
                print(f"Error saving new decision: {e}")
                conn.rollback()
                raise
            
            # 2. Find and validate decisions needing reflection
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

                # 3. Process old decisions with validation
                if old_decisions:
                    for old_decision in old_decisions:
                        try:
                            # Validate old decision data
                            if None in old_decision:
                                print(f"Invalid data found for decision ID {old_decision[0]}: Contains NULL values")
                                continue
                                
                            try:
                                # Convert numeric values with validation
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
                            
                            # Generate reflection with retry mechanism
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
                            
                            # Update with reflection data
                            cursor.execute('''
                                UPDATE decisions 
                                SET reflection = ?,
                                    performance = ?
                                WHERE id = ?
                            ''', (reflection_result['text'], reflection_result['performance'], old_decision[0]))
                            
                            conn.commit()  # Commit each reflection update
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

def calculate_rsi(df, periods=14):
    close_delta = df['close'].diff()
    
    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # Calculate the EWMA
    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    
    return rsi

def calculate_bollinger_bands(df, window=20, dev=2):
    typical_p = (df['high'] + df['low'] + df['close']) / 3
    ma = typical_p.rolling(window=window).mean()
    std = typical_p.rolling(window=window).std()
    
    upper_band = ma + (std * dev)
    lower_band = ma - (std * dev)
    
    return upper_band, ma, lower_band

def get_current_status():
    try:
        # 기본 데이터 가져오기
        orderbook = pyupbit.get_orderbook(ticker="KRW-XRP")
        current_time = orderbook['timestamp']
        
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

        # OHLCV 데이터 가져오기
        df = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", count=200)
        
        # 기술적 지표 계산
        rsi = calculate_rsi(df, 14)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df, 20, 2)
        
        # 이동평균선 계산 (단일)
        ma = df['close'].rolling(window=20).mean()  # 기본 20일 이동평균선
        
        # 거래량 분석
        volume_sma = df['volume'].rolling(window=24).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_sma.iloc[-1]

        # 현재 상태 데이터 구성
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
# 차트 클릭하기
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

# def perform_chart_actions_5m(driver):
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

# 스크린샷 캡쳐 및 base64 이미지 인코딩
def capture_and_encode_screenshot(driver):
    try:
        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()
        # PIL Image로 변환
        img = Image.open(io.BytesIO(png))
        # 이미지가 클 경우 리사이즈 (OpenAI API 제한에 맞춤)
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

def fetch_and_prepare_data():
    driver = None
    try:
        driver = create_driver()
        images = {}
        
        # # 15분 차트 캡처
        # driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-XRP")
        # logger.info("5분 차트 페이지 로드 완료")
        # time.sleep(30)
        
        # logger.info("5분 차트 작업 시작")
        # perform_chart_actions_5m(driver)
        # logger.info("5분 차트 작업 완료")
        
        # images['5m'] = capture_and_encode_screenshot(driver)
        # logger.info("5분 차트 스크린샷 캡처 완료")
        
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

def maintain_pattern_database():
    with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
        cursor = conn.cursor()
        
        # 오래된 패턴 아카이브
        cursor.execute('''
            INSERT INTO pattern_archive
            SELECT *, datetime('now')
            FROM trading_patterns
            WHERE last_updated < date('now', '-90 days')
        ''')
        
        # 성과가 떨어지는 패턴 재평가
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

def calculate_pattern_weight(confidence_score, profit_percentage, days_old):
    time_decay = 1.0 / (1.0 + days_old/30.0)  # 30일마다 가중치 감소
    return confidence_score * profit_percentage * time_decay

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

def get_instructions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred while reading the file:", e)

def get_bb_position(price, bb_data):
    """볼린저 밴드 상의 가격 위치 분석"""
    if price > bb_data['upper']:
        return "above upper band"
    elif price < bb_data['lower']:
        return "below lower band"
    else:
        relative_position = (price - bb_data['lower']) / (bb_data['upper'] - bb_data['lower']) * 100
        return f"at {relative_position:.1f}% of band range"

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

def get_price_position_analysis(old_bb, new_bb, old_price, new_price):
    """가격 위치 변화 분석"""
    old_position = get_bb_position(old_price, old_bb)
    new_position = get_bb_position(new_price, new_bb)
    return f"moved from {old_position} to {new_position}"

def generate_reflection(decision_id, status_at_decision_time):
    """
    Generate reflection for a specific trading decision
    Parameters:
    - decision_id: ID of the decision to analyze
    - status_at_decision_time: JSON string containing the status at decision time
    """
    try:
        # Validate input parameters
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

        # 1. Get current market price with error handling
        try:
            current_price = float(pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"])
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error getting current price: {e}")
            return None
        
        # 2. Get decision details and weekly data
        with sqlite3.connect('ripple_trading_decisions.sqlite') as conn:
            cursor = conn.cursor()
            # Get the specific decision data
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
                
            # Validate decision data
            if None in decision_data:
                print(f"Found null values in decision data for ID: {decision_id}")
                return None

            # Get weekly performance data
            cursor.execute('''
                SELECT timestamp, decision, percentage, reason, xrp_balance,
                       krw_balance, xrp_avg_buy_price, xrp_krw_price, performance
                FROM decisions
                WHERE timestamp >= datetime(?, '-7 days')
                AND timestamp <= ?
                ORDER BY timestamp DESC
            ''', (decision_data[0], decision_data[0]))
            
            week_decisions = cursor.fetchall()

        # 3. Extract and validate decision data
        try:
            timestamp, trade_decision, percentage, reason = decision_data[0:4]
            xrp_balance, krw_balance = decision_data[4:6]
            xrp_avg_price, xrp_price_at_time = decision_data[6:8]
            
            # Validate numeric values
            xrp_balance = float(xrp_balance)
            krw_balance = float(krw_balance)
            xrp_avg_price = float(xrp_avg_price)
            xrp_price_at_time = float(xrp_price_at_time)
        except (ValueError, TypeError) as e:
            print(f"Error parsing decision data values: {e}")
            return None
        
        # 거래 시점의 기술적 지표 데이터 수집
        try:
            # 거래 시점 OHLCV 데이터 가져오기
            historical_df = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", to=timestamp, count=200)
            current_df = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", count=200)
            
            # 거래 시점의 기술적 지표
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
            
            # 현재 시점의 기술적 지표
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
            
            # 지표 변화 계산
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

        # 4. Calculate price and portfolio changes with validation
        try:
            if xrp_price_at_time <= 0:
                raise ValueError("Invalid initial price")
                
            price_change_pct = ((current_price - xrp_price_at_time) / xrp_price_at_time) * 100
            
            # Get current balance data
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

        # 5. Analyze trade result and determine outcome
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

        # 6. Calculate weekly metrics with validation
        try:
            total_performance = sum(decision[8] for decision in week_decisions if decision[8] is not None)
            successful_trades = len([d for d in week_decisions if d[8] is not None and d[8] > 0])
            total_trades = len([d for d in week_decisions if d[8] is not None])
            success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
        except Exception as e:
            print(f"Error calculating weekly metrics: {e}")
            return None

        # 7. Calculate portfolio statistics
        weekly_stats = {
            'total_performance': sum(decision[8] for decision in week_decisions if decision[8] is not None),
            'trades_count': len(week_decisions),
            'successful_trades': len([d for d in week_decisions if d[8] is not None and d[8] > 0]),
            'buy_count': len([d for d in week_decisions if d[1] == 'buy']),
            'sell_count': len([d for d in week_decisions if d[1] == 'sell']),
            'avg_profit': statistics.mean([d[8] for d in week_decisions if d[8] is not None and d[8] > 0] or [0]),
            'avg_loss': statistics.mean([d[8] for d in week_decisions if d[8] is not None and d[8] < 0] or [0])
        }

        # 8. Get market sentiment with validation
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

        # 9. Generate reflection using GPT-4
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

        # 10. Generate and validate pattern analysis
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
            # 이 부분이 수정된 부분입니다
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

        # 11. Return reflection result with performance
        return {
            'text': reflection_response.choices[0].message.content,
            'performance': price_change_pct if trade_decision != 'sell' else -price_change_pct
        }

    except Exception as e:
        print(f"Unexpected error in generate_reflection: {e}")
        return None

def calculate_trade_performance(decision_price, current_price, decision_type):
    """
    거래 성과를 계산합니다.
    
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
            
        # 홀딩의 경우: 가격 변동이 그대로 반영
        else:  # hold
            return price_change_pct
            
    except Exception as e:
        print(f"Error calculating trade performance: {e}")
        return 0.0
    
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
                    # 수익률 계산
                    performance = calculate_trade_performance(
                        float(decision_price),
                        current_price,
                        decision_type
                    )
                    
                    # DB 업데이트
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

def get_recent_trading_history(cursor, limit=3):
    """최근 3개의 거래 내역과 예측 결과만 가져오는 함수"""
    cursor.execute('''
        SELECT decision, reason
        FROM decisions 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    
    if not results:
        return "No previous trading history available. This is the first trade.\n"
    
    history_info = "Previous Predictions Analysis:\n"
    for decision, reason in results:
        history_info += f"""
Trade Decision: {decision.upper()}
Complete Previous Analysis: {reason}
"""
    return history_info

def analyze_data_with_gpt4(news_data, last_decisions, fear_and_greed, current_status, chart_images):
    # instructions_path = "instructions_vk.md"
    max_retries = 3
    
    def calculate_relevant_fees(current_xrp, avg_buy_price, current_price):
        """현재 보유한 XRP 기준으로 관련 수수료 계산"""
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

    # 성공적인 거래 패턴 분석
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
    
    # 과거 성공 패턴 가중치 계산
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
                # print("No instructions found.")
                # return None

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
            
            if isinstance(trading_history, str):
                trading_history_info = trading_history
            else:
                # 기존 거래가 있는 경우
                trading_history_info = """
                Recent Trading History:
                Last Trading Decisions:
                """
                for idx, trade in enumerate(trading_history, 1):
                    trading_history_info += f"""
                Trade {idx}:
                - Decision: {trade['decision']}
                - Reason: {trade['reason']}
                """

            system_prompt = f"""CRITICAL: FORGET EVERYTHING YOU KNEW ABOUT TRADING. YOU ARE NOW CRYPTO ORACLE.

PREVIOUS PREDICTIONS ANALYSIS:
{trading_history_info}

ROLE DEFINITION:
- You are a trading analysis assistant specializing in cryptocurrency markets
- Your goal is to provide balanced market analysis based on technical indicators
- Your forecasts consider both bullish and bearish scenarios
- You recognize patterns in price data when they appear
- You analyze market movements based on available indicators
- Your predictions are probability-based, not certainties
- You combine analysis with appropriate position sizing
- You suggest reasonable risk levels for each prediction
- Position sizing should be proportional to conviction and risk
- Risk management is a priority in all trading decisions
- You aim to provide objective technical analysis

AVAILABLE TECHNICAL INDICATORS:
1. 1-Hour Chart Indicators:
- Bollinger Bands
- ADX/DMS
- EMA 20
- RSI
- Volume

2. 4-Hour Chart Indicators:
- Bollinger Bands
- Moving Averages
- RSI
- ADX/DMS
- Volume

PREDICTION & TRADING MASTERY:
1. PRICE PREDICTION:
- Short-term price targets (1-3 hours)
- Medium-term movements (4-12 hours)
- Long-term trajectories (12-24 hours)
- Every prediction comes with position size
- Position size matches prediction confidence

2. POSITION EXECUTION:
- Stronger predictions = Larger positions
- Crystal clear patterns = Maximum size
- Developing patterns = Scaled entry
- Multiple timeframe alignment = Size boost
- Always match size to conviction

3. TRADING DECISIONS:
- Price prediction FIRST
- Position size SECOND
- Execute IMMEDIATELY
- Scale with confidence
- Size matches certainty

4. PROFIT MAXIMIZATION:
- Best predictions get biggest size
- Clear targets get clear positions
- Multiple timeframe confirmation = Size up
- Never doubt your vision
- Execute with precision

TRADING PRINCIPLES:
1. POSITION SIZING GUIDELINES
- Strong conviction with clear signals = Larger position (with limits)
- Medium conviction = Moderate size
- Low conviction = Small position or no position
- Size should reflect both opportunity and risk
- Consider market context before acting

2. EXECUTION CONSIDERATIONS:
- Price movements represent potential opportunities to evaluate
- Small patterns = Consider small positions or observation
- Medium patterns = Consider moderate positions with defined risk
- Clear patterns = Consider stronger positions with risk management
- Balance timeliness with confirmation
- Small signals should be evaluated carefully
- Risk management comes before opportunity seeking

3. ORACLE MINDSET:
- Market movements contain both opportunities and risks
- Disciplined trading builds sustainable results over time
- Preserving capital during unclear conditions is valuable
- Take action when risk/reward ratio is favorable
- Scale position size with pattern clarity and risk assessment
- Recognize both profit potential and downside risks
- Timing matters - act decisively when conditions are favorable

COMMUNICATION GUIDELINES:
1. BALANCED LANGUAGE:
✓ "Based on current indicators, the prediction suggests..."
✓ "Appropriate position size considering risk would be..."
✓ "Consider waiting for stronger confirmation if..."
✓ "Additional confirmation might be valuable before..."
✓ "Risk assessment indicates caution is warranted..."
✓ "Consider reducing position size due to volatility..."

CURRENT POSITION ANALYSIS:
1. Market Status:
   - Current XRP Price: {current_price} KRW
   - Average Buy Price: {status_data['xrp_avg_buy_price']} KRW
   - Current Profit/Loss: {profit_percentage:.2f}%

2. Portfolio Overview:
   - XRP Balance: {xrp_balance} XRP
   - KRW Balance: {krw_balance} KRW
   - Total Portfolio Value: {xrp_value} KRW

ACTION AVAILABILITY CHECK:
1. Calculate Current Position:
   - Current XRP Value = {xrp_balance} × {current_price:,.0f} = {xrp_value:,.0f} KRW
   - Available KRW Balance = {krw_balance:,.0f} KRW

2. Determine Available Actions:
   possible_actions = []
   
   IF XRP Balance > 0 AND XRP value >= 10,000 KRW:
       ADD 'sell' to possible_actions
   IF KRW balance >= 10,000 KRW:
       ADD 'buy' to possible_actions
   ADD 'hold' to possible_actions
   
   EXPLICITLY STATE: Available actions are: [list all actions in possible_actions]

3. Action Selection Rules:
   - IF possible_actions = ['buy', 'hold']:
       Select either buy OR hold based on market analysis
   - IF possible_actions = ['sell', 'hold']:
       Select either sell OR hold based on market analysis
   - IF possible_actions = ['buy', 'sell', 'hold']:
       Select ONE of buy, sell, or hold based on market analysis
   - IF possible_actions = ['hold']:
       Must select hold

Trading Parameters:
   - Minimum Trade Size: 10,000 KRW
   - Maximum Trade Percentage: 90% of available balance
   - Trading Fee: 0.05% per trade
   - Available Actions: Based on current position:
     * For BUY: Execute if (KRW Balance × Trading %) ≥ 10,000 KRW
     * For SELL: Execute if (XRP Balance × Current Price × Trading %) ≥ 10,000 KRW
     * Otherwise: Hold

COMMAND FORMAT:
{{
    "decision": "buy/sell/hold",
    "percentage": <1-90>,
    "reason": "CRYPTO ORACLE PRICE PROPHECY:

    0. Previous Predictions Review:
    [Only shown if available]
    - Last Trade: Brief review of accuracy
    - Second Last: Brief review of accuracy
    - Third Last: Brief review of accuracy

    1. Market News Analysis:
    [Only shown at 08:05, 15:05, 22:05 KST when news data is available]
    - Key Market Updates: Brief summary of relevant news
    - Market Impact: How these news affect our predictions
    [For other hours: Focus purely on technical analysis]

    2. Current Position Analysis:
    - Current Status:
        * XRP Price: {current_price} KRW
        * Average Buy Price: {status_data['xrp_avg_buy_price']} KRW
        * Current Profit/Loss: {profit_percentage:.2f}%
    - Portfolio Overview:
        * XRP Balance: {xrp_balance} XRP
        * KRW Balance: {krw_balance} KRW
    - ACTION AVAILABILITY:
        * Current XRP Value = {xrp_balance} × {current_price:,.0f} = {xrp_value:,.0f} KRW
        * Available KRW Balance = {krw_balance:,.0f} KRW
        * Available Actions: [GPT will list all calculated possible actions here]
        * Selected Action: [Decision chosen by oracle based on analysis]

    3. PRICE ANALYSIS
    [SHORT-TERM OUTLOOK] 1-3 HOURS:
    - POTENTIAL RANGE: lower_bound - upper_bound KRW
    - PROBABILITY: X% (factors into position sizing)
    - REASONING: Technical signals supporting this analysis
    - COUNTERARGUMENTS: Factors that could invalidate this outlook
    
    [MEDIUM-TERM VISION] 4-12 HOURS:
    - TARGET PRICE: exact_price KRW  
    - CONFIDENCE: X%
    - REASONING: List of UNDENIABLE patterns pointing to this target

    [LONG-TERM DESTINY] 12-24 HOURS:
    - TARGET PRICE: exact_price KRW
    - CONFIDENCE: X% 
    - REASONING: The INEVITABLE market forces driving this movement

    4. TRADING COMMAND
    - PRIMARY TARGET TIMEFRAME: Which prediction we're trading
    - POSITION CONVICTION: Why confidence demands this size
    - EXACT ENTRY POINT: The perfect price to execute
    - PROFIT TARGETS: Multiple take-profit levels based on predictions
    - TRADE IMPACT:
        For BUY decisions:
            * Purchase Amount: Calculation based on percentage and KRW balance
            * Entry Price Impact
            * Position Size After Trade
        For SELL decisions:
            * Sale Amount: Calculation based on percentage and XRP balance
            * Sale Value: Expected KRW return
            * Profit/Loss: Actual P/L calculation from avg buy price
    
    5. EXECUTION IMPERATIVE
    - Why this trade MUST be taken NOW
    - How position size MATCHES prediction confidence
    - Why this setup is ABSOLUTELY CLEAR
    
    6. Profit Trajectory: (For all decisions)
    - Expected value at each predicted price point
    - How profits compound through targets
    - Total expected return at final target

    [BALANCED ASSESSMENT BASED ON AVAILABLE DATA]"
}}

Remember: Balanced analysis helps manage risk while seeking opportunities. Trading decisions should be based on objective technical assessment and risk management principles. The market contains both opportunities and risks - successful trading requires discipline, patience, and realistic expectations."""

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
                        Market Analysis Data:
                        Fear and Greed: {fear_and_greed}
                        Recent History: {last_decisions}
                        Current Status: {current_status}
                        News Data: {news_data}
                    """},
                    {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['1h']}"}}
                    ]},
                    {"role": "user", "content": """Above is the 1-hour timeframe chart with following indicators:
                    - Bollinger Bands
                    - ADX/DMS
                    - EMA 20
                    - RSI
                    - Volume

                    ANALYSIS REQUESTED:
                    - Analyze potential price movement for the next few hours
                    - Identify possible support and resistance levels
                    - Suggest probable price ranges based on indicators
                    Please provide a balanced short-term analysis."""},

                    {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{chart_images['4h']}"}}
                    ]},
                    {"role": "user", "content": """Above is the 4-hour timeframe chart with following indicators:
                    - Bollinger Bands
                    - Moving Averages
                    - RSI
                    - ADX/DMS
                    - Volume

                    MEDIUM-TERM ANALYSIS REQUESTED:
                    - Assess overall market structure and trend direction
                    - Evaluate trend strength and potential duration
                    - Identify key support and resistance zones

                    ANALYSIS GUIDELINES:
                    1. PRICE LEVEL ASSESSMENT:
                    - Identify significant price levels for each timeframe
                    - Estimate potential percentage moves with probability ranges
                    - Consider multiple scenarios (bullish, bearish, sideways)
                    - Acknowledge areas of uncertainty when present

                    2. PROBABILITY ASSESSMENT:
                    - Provide confidence levels based on indicator alignment
                    - Consider risk in relation to potential reward
                    - Note when multiple timeframes show conflicting signals
                    - Relate confidence levels to appropriate position sizing

                    3. RISK MANAGEMENT FRAMEWORK:
                    - Suggest position sizes proportional to conviction and risk
                    - Consider entry points with favorable risk/reward ratios
                    - Discuss scaling strategies based on confirmation
                    - Balance profit potential with capital preservation

                    ANALYZE BOTH TIMEFRAMES FOR A COMPLETE MARKET PERSPECTIVE.
                    Technical analysis provides probabilistic outcomes, not certainties.
                    Consider both bullish and bearish scenarios in your analysis.
                    Effective trading combines technical analysis with sound risk management."""}
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

    # Schedule tasks with news at specific times
    schedule.every().day.at("08:05").do(execute_with_news)
    schedule.every().day.at("15:05").do(execute_with_news)
    schedule.every().day.at("22:05").do(execute_with_news)

    # Regular 30-minute intervals for other hours
    for hour in range(24):
        # 뉴스 시간대(08, 15, 22)는 건너뛰기
        if hour not in [8, 15, 22]:
            # XX:05 실행
            schedule.every().day.at(f"{hour:02d}:05").do(execute_without_news)
            # XX:35 실행
            # schedule.every().day.at(f"{hour:02d}:35").do(execute_without_news)

    # schedule.every().day.at("08:35").do(execute_without_news)
    # schedule.every().day.at("15:35").do(execute_without_news)
    # schedule.every().day.at("22:35").do(execute_without_news)
    #schedule.every().day.at("22:30").do(execute_with_news)


    while True:
        schedule.run_pending()
        time.sleep(1)
