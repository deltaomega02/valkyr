import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import pyupbit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
import schedule
import time
from openai import OpenAI
import plotly.express as px

# dotenv 호출
load_dotenv()

# Upbit 객체 생성
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))

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
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        conn.commit()

# 세션 상태 초기화 함수
def initialize_session_state():
    if 'translated_reasons' not in st.session_state:
        st.session_state.translated_reasons = {}

# 현재 총 보유자산 가치 계산
def get_current_portfolio_value():
    try:
        balances = upbit.get_balances()
        xrp_balance = 0
        krw_balance = 0
        
        for b in balances:
            if b['currency'] == "XRP":
                xrp_balance = float(b['balance'])
            elif b['currency'] == "KRW":
                krw_balance = float(b['balance'])
        
        # 현재 xrp 가격 조회
        current_price = pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"]
        
        # 총 보유자산 가치 계산
        xrp_value = xrp_balance * current_price
        total_value = xrp_value + krw_balance
        
        return total_value, xrp_value, krw_balance, current_price
    except Exception as e:
        st.error(f"보유자산 조회 중 오류가 발생했습니다: {str(e)}")
        return 0, 0, 0, 0

# 실시간 잔고 조회
def get_real_time_balance():
    try:
        balances = upbit.get_balances()
        xrp_balance = 0
        krw_balance = 0
        xrp_avg_buy_price = 0
        
        for b in balances:
            if b['currency'] == "XRP":
                xrp_balance = float(b['balance'])
                xrp_avg_buy_price = float(b['avg_buy_price'])
            elif b['currency'] == "KRW":
                krw_balance = float(b['balance'])
        
        return xrp_balance, krw_balance, xrp_avg_buy_price
    except Exception as e:
        st.error(f"잔고 조회 중 오류가 발생했습니다: {str(e)}")
        return 0, 0, 0
    
# 거래 기록 로드
def load_data():
    db_path = 'ripple_trading_decisions.sqlite'
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(decisions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            required_columns = {
                'timestamp', 'decision', 'percentage', 'reason', 
                'xrp_balance', 'krw_balance', 'xrp_krw_price', 
                'xrp_avg_buy_price', 'performance',
                'fee', 'settlement_amount'
            }
            existing_columns = set(columns)
            
            select_columns = ', '.join(required_columns & existing_columns)
            query = f"SELECT {select_columns} FROM decisions ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn)
            
            for col in required_columns - existing_columns:
                df[col] = None
            
            # 결측값 처리
            df['fee'] = df['fee'].fillna(0)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
        return pd.DataFrame()

# 목표가 정보 로드
def load_target_data():
    db_path = 'ripple_trading_decisions.sqlite'
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT * FROM decision_targets 
            ORDER BY last_updated DESC 
            LIMIT 1
            """
            df = pd.read_sql_query(query, conn)
            
            if len(df) > 0:
                return df.iloc[0]
            else:
                return None
    except Exception as e:
        st.error(f"목표가 데이터 로드 중 오류가 발생했습니다: {str(e)}")
        return None

# 큰 숫자를 읽기 쉽게 포맷팅
def format_large_number(number):
    try:
        # NaN 체크
        if pd.isna(number):
            return "0원"
            
        # 절대값으로 변환하여 처리
        abs_number = abs(number)
        
        if abs_number < 10000:
            return f"{number:,.0f}원"
        elif abs_number < 100000000:  # 1만 이상, 1억 미만
            return f"{number/10000:.1f}만원"
        else:  # 1억 이상
            return f"{number/100000000:.1f}억원"
    except:
        return "0원"
    
# 정기 분석 시간과 뉴스 분석 시간의 다음 실행 시간과 남은 시간 계산
def get_next_execution_time():
    now = datetime.now()
    
    # 뉴스 포함 분석 시간
    news_times = ["08:05", "15:05", "22:05"]
    
    # 30분 간격 정기 분석 시간
    regular_times = []
    for hour in range(24):
        hour_str = str(hour).zfill(2)
        if f"{hour_str}:05" not in news_times:
            regular_times.append(f"{hour_str}:05")
        regular_times.append(f"{hour_str}:35")
    
    # 모든 분석 시간 통합
    all_times = news_times + regular_times
    
    # 시간 변환
    today_times = [
        datetime.strptime(time, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day
        )
        for time in all_times
    ]
    
    # 다음 실행 시간 계산
    upcoming_times = [
        time if time > now else time + timedelta(days=1) 
        for time in today_times
    ]
    next_time = min(upcoming_times)
    
    # 남은 시간 계산
    remaining_time = next_time - now
    hours, remainder = divmod(remaining_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return (
        next_time.strftime("%Y-%m-%d %H:%M"),
        f"{hours:02d}:{minutes:02d}",
        sorted(all_times)  # 모든 분석 시간을 정렬하여 반환
    )

# 상단에 판단 요약 표시
def display_collapsible_schedule_summary():
    st.markdown("""
        <style>
        .countdown-container {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .countdown-time {
            font-size: 2em;
            font-weight: bold;
            color: #1976d2;
        }
        .next-analysis {
            text-align: right;
        }
        .next-analysis-label {
            color: #666;
            font-size: 0.9em;
        }
        .next-analysis-time {
            color: #1976d2;
            font-size: 1.2em;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    next_time, remaining, _ = get_next_execution_time()
    
    # 기본 타이머 표시
    st.markdown(f"""
        <div class="countdown-container">
            <div class="countdown-time">{remaining}</div>
            <div class="next-analysis">
                <div class="next-analysis-label">다음 판단</div>
                <div class="next-analysis-time">{next_time}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# 목표가 카드 표시
def display_target_goals():
    target_data = load_target_data()
    if target_data is None:
        st.info("목표가 정보가 아직 설정되지 않았습니다.")
        return

    st.markdown("### 🎯 거래 목표")
    
    # 현재 가격 조회
    _, _, _, current_price = get_current_portfolio_value()
    
    # 단기 목표와 현재 가격 사이의 거리 계산
    short_term_distance = ((target_data['short_term_target'] - current_price) / current_price) * 100
    
    # 단기 목표 정보를 보여주기 위한 컨테이너
    with st.container():
        # CSS 클래스 대신 직접 스타일 적용
        st.markdown("""
            <style>
            .target-card {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 20px;
                border-top: 5px solid #42a5f5;
                margin-bottom: 20px;
            }
            .target-info {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .target-info-label {
                width: 120px;
                color: #666;
                font-size: 0.9rem;
            }
            .target-info-value {
                font-size: 1.1rem;
                font-weight: 500;
            }
            .target-info-value.positive {
                color: #4caf50;
            }
            .target-info-value.negative {
                color: #f44336;
            }
            .confidence-bar-container {
                background-color: #f1f1f1;
                height: 20px;
                border-radius: 10px;
                margin-top: 15px;
                position: relative;
            }
            .confidence-bar {
                height: 100%;
                border-radius: 10px;
                background-image: linear-gradient(to right, #42a5f5, #1976d2);
            }
            .confidence-text {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 500;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }
            </style>
        """, unsafe_allow_html=True)

        # 직접 필요한 정보 렌더링
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### 🚀 단기 목표")
        
        with col2:
            # 목표가, 손절가, 예상 기간, 목표까지 표시
            cols = st.columns(4)
            with cols[0]:
                st.markdown("##### 목표가")
                st.markdown(f"<span style='color:#4caf50; font-weight:bold; font-size:1.2rem;'>{format_large_number(target_data['short_term_target'])}</span>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown("##### 손절가")
                st.markdown(f"<span style='color:#f44336; font-weight:bold; font-size:1.2rem;'>{format_large_number(target_data['short_term_stop_loss'])}</span>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown("##### 도달 시간")
                st.markdown(f"<span style='font-weight:bold; font-size:1.2rem;'>{target_data['short_term_target_time']}</span>", unsafe_allow_html=True)
            with cols[3]:
                st.markdown("##### 목표까지")
                color = "#4caf50" if short_term_distance > 0 else "#f44336"
                st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:1.2rem;'>{short_term_distance:.2f}%</span>", unsafe_allow_html=True)
        
        # 신뢰도 바 표시
        st.markdown("##### 신뢰도")
        confidence = target_data['short_term_confidence']
        st.progress(confidence/100, text=f"{confidence}%")

# 리플 가격 차트 생성
# 리플 가격 차트 생성
def create_ripple_price_chart():
    df_price = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", count=24)
    
    # 실시간 평균 매수가 조회
    _, _, xrp_avg_buy_price = get_real_time_balance()
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_price.index,
            y=df_price['close'],
            mode='lines',
            name="XRP/KRW",
            line=dict(color='#2196F3', width=2)
        )
    )
    
    # 목표가 정보 가져오기
    target_data = load_target_data()
    if target_data is not None:
        # 단기 목표가 수평선 추가
        fig.add_shape(
            type="line",
            x0=df_price.index[0],
            y0=target_data['short_term_target'],
            x1=df_price.index[-1],
            y1=target_data['short_term_target'],
            line=dict(
                color="#4CAF50",
                width=2,
                dash="dash",
            ),
        )
        
        # 단기 손절가 수평선 추가
        fig.add_shape(
            type="line",
            x0=df_price.index[0],
            y0=target_data['short_term_stop_loss'],
            x1=df_price.index[-1],
            y1=target_data['short_term_stop_loss'],
            line=dict(
                color="#F44336",
                width=2,
                dash="dash",
            ),
        )
        
        # 평균 매수가 수평선 추가
        fig.add_shape(
            type="line",
            x0=df_price.index[0],
            y0=xrp_avg_buy_price,
            x1=df_price.index[-1],
            y1=xrp_avg_buy_price,
            line=dict(
                color="#FF9800",  # 주황색으로 설정
                width=2,
                dash="dot",  # 점선으로 설정하여 구분
            ),
        )
        
        # 주석 추가
        fig.add_annotation(
            x=df_price.index[-1],
            y=target_data['short_term_target'],
            text="단기 목표가",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#4CAF50",
            arrowsize=1,
            arrowwidth=2,
            ax=-80,
            ay=-30,
            font=dict(
                size=10,
                color="#4CAF50"
            ),
            bgcolor="white",
            bordercolor="#4CAF50",
            borderwidth=1,
            borderpad=4,
        )
        
        fig.add_annotation(
            x=df_price.index[-1],
            y=target_data['short_term_stop_loss'],
            text="단기 손절가",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#F44336",
            arrowsize=1,
            arrowwidth=2,
            ax=-80,
            ay=30,
            font=dict(
                size=10,
                color="#F44336"
            ),
            bgcolor="white",
            bordercolor="#F44336",
            borderwidth=1,
            borderpad=4,
        )
        
        # 평균 매수가 주석 추가
        fig.add_annotation(
            x=df_price.index[-1],
            y=xrp_avg_buy_price,
            text="평균 매수가",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#FF9800",
            arrowsize=1,
            arrowwidth=2,
            ax=-80,
            ay=0,  # 중간 위치로 설정
            font=dict(
                size=10,
                color="#FF9800"
            ),
            bgcolor="white",
            bordercolor="#FF9800",
            borderwidth=1,
            borderpad=4,
        )
    else:
        # 목표가 데이터가 없을 경우에도 평균 매수가는 표시
        fig.add_shape(
            type="line",
            x0=df_price.index[0],
            y0=xrp_avg_buy_price,
            x1=df_price.index[-1],
            y1=xrp_avg_buy_price,
            line=dict(
                color="#FF9800",  # 주황색으로 설정
                width=2,
                dash="dot",  # 점선으로 설정하여 구분
            ),
        )
        
        fig.add_annotation(
            x=df_price.index[-1],
            y=xrp_avg_buy_price,
            text="평균 매수가",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#FF9800",
            arrowsize=1,
            arrowwidth=2,
            ax=-80,
            ay=0,
            font=dict(
                size=10,
                color="#FF9800"
            ),
            bgcolor="white",
            bordercolor="#FF9800",
            borderwidth=1,
            borderpad=4,
        )
    
    fig.update_layout(
        title={
            'text': '리플 가격 추이 (24시간)',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20)
        },
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor='#f0f0f0',
            zeroline=False,
            title='시간'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f0f0f0',
            zeroline=False,
            title='가격 (KRW)',
            tickformat=','
        )
    )
    
    return fig

# 자산 현황 보기
def create_asset_overview(xrp_balance, krw_balance, current_price, xrp_avg_buy_price):
    xrp_value = xrp_balance * current_price
    total_value = xrp_value + krw_balance
    
    # CSS 스타일 정의
    st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background-color: #2196F3;
        }
        .stProgress {
            height: 35px !important;
            margin-bottom: 1rem !important;
        }
        .stProgress > div > div > div {
            height: 35px !important;
        }
        .stProgress > div > div > div > div:first-child {
            height: 35px !important;
            line-height: 35px !important;
            padding-left: 12px !important;
            font-size: 1.3rem !important;
            font-weight: 600 !important;
        }
        .asset-percentage {
            font-size: 1.5rem !important;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # 첫 번째 행: 자산 분배
    st.markdown("### 💰 자산 분배")
    xrp_percentage = (xrp_value / total_value) * 100 if total_value > 0 else 0
    krw_percentage = (krw_balance / total_value) * 100 if total_value > 0 else 0
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<p class="asset-percentage">리플 {xrp_percentage:.1f}%</p>', unsafe_allow_html=True)
            st.progress(xrp_percentage / 100, text="")
        with col2:
            st.markdown(f'<p class="asset-percentage">현금 {krw_percentage:.1f}%</p>', unsafe_allow_html=True)
            st.progress(krw_percentage / 100, text="")
    
    # 두 번째 행: 투자 현황 요약
    st.markdown("### 📊 투자 현황")
    
    # 업비트 스타일 수익률 계산
    upbit_profit_percentage = calculate_upbit_style_profit_percentage(
        xrp_balance, current_price, xrp_avg_buy_price)
    profit_amount = calculate_profit_amount(
        xrp_balance, current_price, xrp_avg_buy_price)
    
    # 투자 현황 지표들
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">코인 수익률</div>
                <div class="metric-value {'negative-delta' if upbit_profit_percentage < 0 else 'positive-delta'}">{upbit_profit_percentage:+.2f}%</div>
                <div class="metric-delta {'negative-delta' if profit_amount < 0 else 'positive-delta'}">{format_large_number(round(profit_amount))}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">총 보유자산</div>
                <div class="metric-value">{format_large_number(total_value)}</div>
                <div class="metric-delta">XRP: {format_large_number(xrp_value)}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        price_change = current_price - xrp_avg_buy_price
        price_change_percentage = ((current_price - xrp_avg_buy_price) / xrp_avg_buy_price * 100) if xrp_avg_buy_price > 0 else 0
        
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">XRP 현재가</div>
                <div class="metric-value">{format_large_number(current_price)}</div>
                <div class="metric-delta {'negative-delta' if price_change < 0 else 'positive-delta'}">{price_change_percentage:+.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

# 수익률 계산
def calculate_upbit_style_profit_percentage(xrp_balance, current_price, xrp_avg_buy_price):
    if xrp_balance <= 0 or xrp_avg_buy_price <= 0:
        return 0
    
    return ((current_price - xrp_avg_buy_price) / xrp_avg_buy_price) * 100

# 실현/미실현 손익 계산
def calculate_profit_amount(xrp_balance, current_price, xrp_avg_buy_price):
    if xrp_balance <= 0:
        return 0
    
    total_current_value = xrp_balance * current_price
    total_bought_value = xrp_balance * xrp_avg_buy_price
    return total_current_value - total_bought_value

# 거래 기록에서 누적 수수료 계산
def calculate_total_fees(df):
    if 'fee' in df.columns:
        return df['fee'].fillna(0).sum()  # 결측값 처리 후 합산
    return 0

# 총 누적 수익 계산
def calculate_total_profit(df):
    realized_profit = 0
    unrealized_profit = 0
    current_holdings = {
        'amount': 0,  # XRP 보유량
        'total_cost': 0  # 총 매수 비용
    }
    
    df = df.sort_values('timestamp')
    
    for _, trade in df.iterrows():
        price = float(trade['xrp_krw_price'])
        settlement = float(trade.get('settlement_amount', 0))
        percentage = float(trade.get('percentage', 100)) / 100
        
        if trade['decision'] == 'buy':
            if settlement > 0:
                # 매수 시: 보유량과 총 비용 증가
                amount = settlement / price
                current_holdings['amount'] += amount
                current_holdings['total_cost'] += settlement
                
        elif trade['decision'] == 'sell':
            if current_holdings['amount'] > 0:
                # 매도 시: 현재 평균 매수가 기준으로 수익 계산
                sell_amount = current_holdings['amount'] * percentage
                if sell_amount > 0:
                    avg_buy_price = current_holdings['total_cost'] / current_holdings['amount']
                    sell_value = sell_amount * price
                    cost_basis = sell_amount * avg_buy_price
                    
                    # 실현 수익 계산
                    trade_profit = sell_value - cost_basis
                    realized_profit += trade_profit
                    
                    # 남은 보유량과 비용 갱신
                    current_holdings['amount'] -= sell_amount
                    current_holdings['total_cost'] -= cost_basis
    
    # 미실현 수익 계산
    if current_holdings['amount'] > 0:
        current_price = pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"]
        avg_buy_price = current_holdings['total_cost'] / current_holdings['amount']
        unrealized_profit = current_holdings['amount'] * (current_price - avg_buy_price)
    
    total_profit = realized_profit + unrealized_profit
    total_fees = df['fee'].fillna(0).sum()
    
    return total_profit, realized_profit, unrealized_profit, total_fees

# 수익 지표 표시 - 현재 보유자산 기준
def display_profit_metrics(df):
    total_profit, realized_profit, unrealized_profit, total_fees = calculate_total_profit(df)
    current_value, xrp_value, krw_balance, _ = get_current_portfolio_value()

    # 누적 수수료 계산
    total_fees = calculate_total_fees(df)

    st.markdown("### 💰 총 수익 현황")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">총 보유자산</div>
                <div class="metric-value">{format_large_number(round(current_value))}</div>
                <div class="metric-delta">XRP: {format_large_number(round(xrp_value))} / KRW: {format_large_number(round(krw_balance))}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">실현 수익</div>
                <div class="metric-value {'negative-delta' if realized_profit < 0 else 'positive-delta'}">{format_large_number(round(realized_profit))}</div>
                <div class="metric-delta">{'수익 실현 완료' if realized_profit != 0 else '실현 수익 없음'}</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">미실현 수익</div>
                <div class="metric-value {'negative-delta' if unrealized_profit < 0 else 'positive-delta'}">{format_large_number(round(unrealized_profit))}</div>
                <div class="metric-delta">현재 보유분 평가손익</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">총 누적 수수료</div>
                <div class="metric-value">{format_large_number(round(total_fees))}</div>
                <div class="metric-delta">전체 거래 수수료 합계</div>
            </div>
        """, unsafe_allow_html=True)

# 시간당/일평균 수익 표시 전환
def toggle_profit_view():
    st.session_state.profit_view_type = 'daily' if st.session_state.profit_view_type == 'hourly' else 'hourly'

# 거래 내역 표시
def display_transaction_history():
    st.markdown("### 📝 거래 내역")
    
    df = load_data()
    
    if not df.empty:
        # 거래 내역 정렬 및 필터링 옵션 (가로로 배치)
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            sort_order = st.radio(
                "정렬 순서",
                options=["최신순", "오래된순"],
                horizontal=True
            )
        with col2:
            transaction_types = ["전체", "매수", "매도", "홀딩"]
            selected_type = st.selectbox("거래 유형", transaction_types)
        with col3:
            max_items = st.slider("표시할 항목 수", 5, 50, 10)
        
        if sort_order == "최신순":
            df = df.sort_values('timestamp', ascending=False)
        else:
            df = df.sort_values('timestamp', ascending=True)
        
        if selected_type != "전체":
            type_mapping = {"매수": "buy", "매도": "sell", "홀딩": "hold"}
            df = df[df["decision"] == type_mapping[selected_type]]
        
        # 최대 항목 수에 맞게 데이터프레임 자르기
        df_display = df.head(max_items)
        
        # 거래 내역 표시 - 탭 형태로 구성
        for idx, row in df_display.iterrows():
            decision_type = row['decision']
            
            # 결정 타입별 색상 및 아이콘 설정
            if decision_type == 'buy':
                card_color = "#e3f2fd"
                border_color = "#2196F3"
                icon = "📈"
                decision_text = "매수"
            elif decision_type == 'sell':
                card_color = "#ffebee"
                border_color = "#f44336"
                icon = "📉"
                decision_text = "매도"
            else:  # hold
                card_color = "#f1f8e9"
                border_color = "#8bc34a"
                icon = "⏸️"
                decision_text = "홀딩"
            
            # 타임스탬프 문자열 형식으로 명시적 변환
            timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
            
            # 카드 헤더와 탭 구성 - 모든 거래 타입에 동일한 방식 적용
            with st.container():
                if decision_type in ['buy', 'sell'] and "percentage" in row and row["percentage"] < 100:
                    percentage_text = f' {row["percentage"]}%'
                else:
                    percentage_text = ''

                # 카드 헤더 스타일
                header_html = f"""
                    <div style="
                        background-color: {card_color};
                        border-left: 5px solid {border_color};
                        border-radius: 5px 5px 0 0;
                        padding: 12px 15px;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <div style="font-weight: bold; font-size: 16px;">
                            {icon} {decision_text}{percentage_text}
                        </div>
                        <div style="color: #666; font-size: 14px;">
                            {timestamp_str}
                        </div>
                    </div>
                """
                st.markdown(header_html, unsafe_allow_html=True)
                
                # 탭 구성
                tab1, tab2 = st.tabs(["요약", "상세 내용"])
                
                # 탭 1: 요약 정보
                with tab1:
                    # 거래 정보 간략히 표시
                    if decision_type in ['buy', 'sell']:
                        settlement_amount = row.get('settlement_amount', 0)
                        
                        if decision_type == 'buy':
                            xrp_amount = settlement_amount / row['xrp_krw_price'] if row['xrp_krw_price'] > 0 else 0
                            st.markdown(f"""
                                <div style="padding: 10px; background-color: #f8f9fa; border-radius: 4px; margin-top: 5px;">
                                    <div style="font-weight: bold; margin-bottom: 5px;">거래 요약</div>
                                    <div style="display: flex; justify-content: space-between;">
                                        <div>매수가: {format_large_number(row['xrp_krw_price'])}</div>
                                        <div>수량: {xrp_amount:.2f} XRP</div>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                                        <div>정산금액: {format_large_number(settlement_amount)}</div>
                                        <div>수수료: {format_large_number(row.get('fee', 0))}</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                        else:  # sell
                            xrp_amount = (row['xrp_balance'] * row['percentage'] / 100) if 'percentage' in row else 0
                            st.markdown(f"""
                                <div style="padding: 10px; background-color: #f8f9fa; border-radius: 4px; margin-top: 5px;">
                                    <div style="font-weight: bold; margin-bottom: 5px;">거래 요약</div>
                                    <div style="display: flex; justify-content: space-between;">
                                        <div>매도가: {format_large_number(row['xrp_krw_price'])}</div>
                                        <div>수량: {xrp_amount:.2f} XRP</div>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                                        <div>정산금액: {format_large_number(settlement_amount)}</div>
                                        <div>수수료: {format_large_number(row.get('fee', 0))}</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:  # hold
                        # 홀딩 결정 표시
                        xrp_balance_str = f"{row.get('xrp_balance', 0):.2f}"
                        xrp_price_str = format_large_number(row.get('xrp_krw_price', 0))
                        krw_balance_str = format_large_number(row.get('krw_balance', 0))
                        avg_buy_price_str = format_large_number(row.get('xrp_avg_buy_price', 0))
                        
                        # 테이블 대신 div 기반 레이아웃 사용 (매수/매도와 같은 구조)
                        hold_summary_html = f"""
                            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 4px; margin-top: 5px;">
                                <div style="font-weight: bold; margin-bottom: 5px;">현황 요약</div>
                                <div style="display: flex; justify-content: space-between;">
                                    <div>보유 XRP: {xrp_balance_str} XRP</div>
                                    <div>현재가: {xrp_price_str}</div>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                                    <div>현금: {krw_balance_str}</div>
                                    <div>평균매수가: {avg_buy_price_str}</div>
                                </div>
                            </div>
                        """
                        
                        st.markdown(hold_summary_html, unsafe_allow_html=True)
                
                # 탭 2: 상세 내용
                with tab2:
                    # 결정 이유를 expander로 표시 (접혀있다가 클릭하면 펼쳐짐)
                    if 'reason' in row and row['reason']:
                        with st.expander("**📋 판단 근거 (클릭하여 펼치기)**"):
                            st.markdown(f"""
                                <div style="
                                    background-color: white;
                                    border: 1px solid #e0e0e0;
                                    border-radius: 4px;
                                    padding: 15px;
                                    font-size: 14px;
                                    line-height: 1.5;
                                ">
                                    {row['reason']}
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # 거래 데이터를 expander로 표시
                    with st.expander("**📊 상세 거래 정보 (클릭하여 펼치기)**"):
                        col1, col2 = st.columns(2)
                        with col1:
                            if decision_type in ['buy', 'sell']:
                                info = [
                                    ("결정 유형", f"{decision_text} ({row['percentage']}%)" if row['percentage'] < 100 else decision_text),
                                    ("거래 시간", row['timestamp'].strftime('%Y-%m-%d %H:%M')),
                                    ("XRP 가격", format_large_number(row['xrp_krw_price'])),
                                    ("정산 금액", format_large_number(row.get('settlement_amount', 0))),
                                    ("수수료", format_large_number(row.get('fee', 0)))
                                ]
                            else:
                                info = [
                                    ("결정 유형", decision_text),
                                    ("거래 시간", row['timestamp'].strftime('%Y-%m-%d %H:%M')),
                                    ("XRP 가격", format_large_number(row['xrp_krw_price'])),
                                    ("수익률", f"{row.get('performance', 0):.2f}%")
                                ]
                            
                            for label, value in info:
                                st.markdown(f"""
                                    <div style="display: flex; margin-bottom: 8px;">
                                        <div style="width: 100px; color: #666;">{label}</div>
                                        <div style="font-weight: 500;">{value}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            info2 = [
                                ("보유 XRP", f"{row['xrp_balance']:.4f} XRP"),
                                ("보유 현금", format_large_number(row['krw_balance'])),
                                ("평균 매수가", format_large_number(row['xrp_avg_buy_price'])),
                                ("총 자산 가치", format_large_number(row['xrp_balance'] * row['xrp_krw_price'] + row['krw_balance']))
                            ]
                            
                            for label, value in info2:
                                st.markdown(f"""
                                    <div style="display: flex; margin-bottom: 8px;">
                                        <div style="width: 100px; color: #666;">{label}</div>
                                        <div style="font-weight: 500;">{value}</div>
                                    </div>
                                """, unsafe_allow_html=True)
    else:
        st.info("거래 내역이 없습니다.")

def main():
    st.set_page_config(
        layout="wide",
        page_title="V.A.L.K.Y.R."
    )
    
    # 데이터베이스 초기화
    initialize_db()

    # 세션 상태 초기화 추가
    initialize_session_state()
    
    # 전역 스타일 정의
    st.markdown("""
        <style>
        /* 타이틀 스타일 */
        .title-container {
            position: relative;
            display: inline-block;
        }
        .title-tooltip {
            visibility: hidden;
            background-color: rgba(0, 0, 0, 0.9);
            color: white;
            text-align: left;
            padding: 15px;
            border-radius: 8px;
            position: absolute;
            z-index: 1;
            top: 100%;
            left: 0;
            margin-top: 10px;
            width: 400px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .title-container:hover .title-tooltip {
            visibility: visible;
            opacity: 1;
        }
        .tooltip-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 8px;
            color: #4CAF50;
        }
        .tooltip-subtitle {
            font-size: 1em;
            color: #90caf9;
            margin-bottom: 8px;
        }
        .tooltip-description {
            font-size: 0.9em;
            line-height: 1.4;
        }

        /* 메트릭 카드 스타일 */
        .metric-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
            text-align: center;
            margin: 0.5rem 0;
        }
        .metric-title {
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            font-weight: 500;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0.8rem 0;
        }
        .metric-delta {
            color: #666;
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        .positive-delta {
            color: #2E7D32;
            font-weight: 500;
        }
        .negative-delta {
            color: #C62828;
            font-weight: 500;
        }
        </style>
        <div class="title-container">
            <h1>🧬 V.A.L.K.Y.R.</h1>
            <div class="title-tooltip">
                <div class="tooltip-title">V.A.L.K.Y.R.</div>
                <div class="tooltip-subtitle">Virtual AI-based Leverage & Yield Keeper Rebalancer</div>
                <div class="tooltip-description">
                    가상자산 AI 기반 레버리지 & 수익 관리 리밸런서<br><br>
                    GPT를 활용하여 매시간 XRP 코인의 차트를 분석하고, 시장 상황에 따라 최적의 거래 전략을 도출하여 수익을 창출하는 자동화된 트레이딩 시스템입니다.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    display_collapsible_schedule_summary()
    
    # 첫 번째 섹션: 운영 현황 요약
    df = load_data()
    if not df.empty:
        total_profit, realized_profit, unrealized_profit, total_fees = calculate_total_profit(df)
        net_profit = realized_profit + unrealized_profit - total_fees
        
        # 운영 시간 계산
        first_trade_time = df['timestamp'].min()
        current_time = datetime.now()
        time_diff = current_time - first_trade_time
        days = time_diff.days
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds % 3600) // 60
        
        # 세션 상태 초기화
        if 'profit_view_type' not in st.session_state:
            st.session_state.profit_view_type = 'daily'
        
        total_hours = time_diff.total_seconds() / 3600
        total_days = time_diff.total_seconds() / (24 * 60 * 60)
        
        # 시간당 수익은 항상 계산
        hourly_profit = net_profit / total_hours if total_hours > 0 else 0
        
        if st.session_state.profit_view_type == 'hourly':
            avg_profit = hourly_profit
            profit_text = f"시간당 {format_large_number(round(avg_profit))}"
            profit_description = "시간당 순이익 (클릭하여 일평균 확인)"
        else:
            if total_hours < 24:
                profit_text = "일평균 계산불가"
                profit_description = "24시간 이상 운영 필요 (클릭하여 시간당 확인)"
                avg_profit = 0
            else:
                avg_profit = net_profit / total_days
                profit_text = f"일평균 {format_large_number(round(avg_profit))}"
                profit_description = "일평균 순이익 (클릭하여 시간당 확인)"
        
        # 운영 현황 표시
        st.markdown("### 🧪 운영 현황")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">운영 시간</div>
                    <div class="metric-value">{days}일 {hours}시간</div>
                    <div class="metric-delta">{minutes}분</div>
                </div>
            """, unsafe_allow_html=True)
            
        with status_col2:
            # 클릭 가능한 버튼으로 변경
            if st.button(
                label="평균 수익",
                key="profit_toggle",
                help="클릭하여 시간당/일평균 전환",
                use_container_width=True
            ):
                toggle_profit_view()
                st.rerun()
            
            profit_color = "positive-delta" if avg_profit >= 0 else "negative-delta"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">평균 수익</div>
                    <div class="metric-value {profit_color}">{profit_text}</div>
                    <div class="metric-delta">{profit_description}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with status_col3:
            net_profit_color = "positive-delta" if net_profit >= 0 else "negative-delta"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">총 순이익</div>
                    <div class="metric-value {net_profit_color}">{format_large_number(round(net_profit))}</div>
                    <div class="metric-delta">실현+미실현-수수료</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    xrp_balance, krw_balance, xrp_avg_buy_price = get_real_time_balance()
    current_price = pyupbit.get_orderbook(ticker="KRW-XRP")['orderbook_units'][0]["ask_price"]
    current_value = xrp_balance * current_price + krw_balance
    
    # 목표가 정보 표시 (단기 목표만)
    display_target_goals()
    
    df = load_data()
    
    if not df.empty:
        time_diff = datetime.now() - df.iloc[0]['timestamp']
        days = time_diff.days
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds % 3600) // 60
        
        display_profit_metrics(df)

        total_trades = len(df)
        trades_by_type = df['decision'].value_counts()
        
        st.markdown("### 🔬 거래 통계")
        stat_col1, stat_col2 = st.columns([1, 3])  # 1:3 비율로 컬럼 분할

        # 왼쪽 컬럼: 파이 차트
        with stat_col1:
            # 파이 차트 데이터 준비
            labels = {'buy': '매수', 'sell': '매도', 'hold': '홀딩'}
            values = [trades_by_type.get(key, 0) for key in ['buy', 'sell', 'hold']]
            
            # 더 세련된 색상 팔레트
            colors = ['#4CAF50', '#FF5252', '#2196F3']  # 부드러운 녹색, 산호색, 하늘색
            
            fig = go.Figure(data=[go.Pie(
                labels=[labels[k] for k in ['buy', 'sell', 'hold']],
                values=values,
                marker=dict(
                    colors=colors,
                    line=dict(color='#ffffff', width=2)  # 흰색 테두리 추가
                ),
                hole=0.6,  # 도넛 홀 크기 증가
                textinfo='label+percent',
                textfont=dict(size=14, family="Arial, sans-serif"),
                hovertemplate="<b>%{label}</b><br>%{value}회<br>비율: %{percent}<extra></extra>",
                rotation=90,  # 회전 각도 조정
                pull=[0.05, 0.05, 0.05]  # 각 섹션을 살짝 분리
            )])
            
            fig.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                height=250,
                width=250,
                paper_bgcolor='rgba(0,0,0,0)',  # 투명 배경
                plot_bgcolor='rgba(0,0,0,0)',
                annotations=[
                    dict(
                        text=f'총 {sum(values)}회',
                        x=0.5, y=0.5,
                        font=dict(size=16, family='Arial, sans-serif', color='#333333'),
                        showarrow=False
                    )
                ]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        # 오른쪽 컬럼: 기존 통계
        with stat_col2:
            cols = st.columns(4)
            with cols[0]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">총 거래 횟수</div>
                        <div class="metric-value">{total_trades}회</div>
                    </div>
                """, unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">매수 횟수</div>
                        <div class="metric-value">{trades_by_type.get('buy', 0)}회</div>
                    </div>
                """, unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">매도 횟수</div>
                        <div class="metric-value">{trades_by_type.get('sell', 0)}회</div>
                    </div>
                """, unsafe_allow_html=True)
            with cols[3]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">홀딩 횟수</div>
                        <div class="metric-value">{trades_by_type.get('hold', 0)}회</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # 자산 현황 시각화
        create_asset_overview(
            xrp_balance,
            krw_balance,
            current_price,
            xrp_avg_buy_price
        )
        
        # 포트폴리오 현황
        st.markdown("### 📊 포트폴리오 상세")
        portfolio_col1, portfolio_col2, portfolio_col3, portfolio_col4 = st.columns(4)

        with portfolio_col1:
            # 업비트 스타일 수익률 계산
            upbit_profit_percentage = calculate_upbit_style_profit_percentage(
                xrp_balance, current_price, xrp_avg_buy_price)
            profit_amount = calculate_profit_amount(
                xrp_balance, current_price, xrp_avg_buy_price)
            
            formatted_profit = format_large_number(round(profit_amount))
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">코인 수익률</div>
                    <div class="metric-value {'negative-delta' if upbit_profit_percentage < 0 else 'positive-delta'}">{upbit_profit_percentage:+.2f}%</div>
                    <div class="metric-delta {'negative-delta' if profit_amount < 0 else 'positive-delta'}">{formatted_profit}</div>
                </div>
            """, unsafe_allow_html=True)

        with portfolio_col2:
            xrp_value = xrp_balance * current_price
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">총 보유자산</div>
                    <div class="metric-value">{format_large_number(current_value)}</div>
                    <div class="metric-delta">XRP: {format_large_number(xrp_value)}</div>
                </div>
            """, unsafe_allow_html=True)

        with portfolio_col3:
            price_change = current_price - xrp_avg_buy_price
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">XRP 현재가</div>
                    <div class="metric-value">{format_large_number(current_price)}</div>
                    <div class="metric-delta {'negative-delta' if price_change < 0 else 'positive-delta'}">{format_large_number(price_change)}</div>
                </div>
            """, unsafe_allow_html=True)

        with portfolio_col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">투자 기간</div>
                    <div class="metric-value">{days}일 {hours}시간</div>
                    <div class="metric-delta">{minutes}분</div>
                </div>
            """, unsafe_allow_html=True)
        
        # 리플 가격 차트
        st.markdown("### 📈 리플 가격 차트")
        price_chart = create_ripple_price_chart()
        st.plotly_chart(price_chart, use_container_width=True)
        
        # 상세 정보
        st.markdown("### 💎 보유 자산 정보")
        detail_col1, detail_col2, detail_col3 = st.columns(3)

        with detail_col1:
            xrp_value_krw = xrp_balance * current_price
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">보유 XRP</div>
                    <div class="metric-value">{xrp_balance:.8f} XRP</div>
                    <div class="metric-delta">{format_large_number(xrp_value_krw)}</div>
                </div>
            """, unsafe_allow_html=True)

        with detail_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">보유 현금</div>
                    <div class="metric-value">{format_large_number(krw_balance)}</div>
                    <div class="metric-delta"></div>
                </div>
            """, unsafe_allow_html=True)

        with detail_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">XRP 평균 매수가</div>
                    <div class="metric-value">{format_large_number(xrp_avg_buy_price)}</div>
                    <div class="metric-delta"></div>
                </div>
            """, unsafe_allow_html=True)
        
        # 거래 내역을 메인 영역 하단에 표시
        display_transaction_history()
    else:
        st.info("거래 기록이 없습니다.")

if __name__ == '__main__':
    main()