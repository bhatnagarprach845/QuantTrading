import os

# Silence Hugging Face caching/symlink alerts on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import smtplib
from email.mime.text import MIMEText

# Alpaca Trading & Execution Infrastructure
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# Alpaca Historical Data, Snapshots, News, & Screener Infrastructure
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import StockSnapshotRequest, StockBarsRequest, NewsRequest, MostActivesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import MostActivesBy

# =====================================================================
# AGGRESSIVE ENGINE CONFIGURATION MATRIX (HIGH RISK / HIGH REWARD)
# =====================================================================
API_KEY = "PKZDEM6OFD7W4S7IPL6PKWX4I4"
SECRET_KEY = "B9csJGtky9cpgrebkyScvYZyCarMhiYCm7zeFevnsF4B"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)
screener_client = ScreenerClient(API_KEY, SECRET_KEY)

# Hyper-Responsive Exponential Moving Averages (EMA)
FAST_EMA_PERIOD = 3
SLOW_EMA_PERIOD = 8
MAX_OPEN_POSITIONS = 15
TRADE_VALUE_USD = 1200

# Base Bracket Constraints for Mid-Caps (Penny stocks have wider dynamic brackets)
TAKE_PROFIT_PERCENT = 0.05
STOP_LOSS_PERCENT = 0.02

# =====================================================================
# EMAIL NOTIFICATION ENGINE CONFIGURATION
# =====================================================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "bhatnagar.prachi845@gmail.com"  # Your sending Gmail account
SENDER_PASSWORD = "coic qleo lybh ngcl"  # Your 16-digit Google App Password
TARGET_EMAIL_INBOX = "bhatnagar.prachi845@gmail.com"  # Your receiving personal email inbox


def send_email_alert(subject: str, message_body: str):
    """Transmits instantaneous trade alerts directly to your personal email inbox."""
    try:
        msg = MIMEText(message_body)
        msg['From'] = SENDER_EMAIL
        msg['To'] = TARGET_EMAIL_INBOX
        msg['Subject'] = subject

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, TARGET_EMAIL_INBOX, msg.as_string())
        server.quit()
        print(f"   [Email] Trade notification routed to {TARGET_EMAIL_INBOX} successfully.")
    except Exception as e:
        print(f"   [!] Failed to route email notification alert: {e}")


# Global vocabulary vectors for Lexical Scanner
BEARISH_KEYWORDS = ["downgrade", "misses", "lawsuit", "deficit", "investigation", "fraud", "slashes", "bearish", "drop",
                    "fall", "plummet", "sink", "lower", "decline", "loss", "negative", "warns", "weak", "cut", "fail",
                    "failed", "warn"]
BULLISH_KEYWORDS = ["upgrade", "beats", "partnership", "growth", "surge", "acquisition", "profit", "bullish", "rise",
                    "gain", "rally", "higher", "climb", "jump", "positive", "win", "strong", "grew", "expand",
                    "succeed", "soars"]


# =====================================================================
# TECHNICAL & PREDICTIVE INDICATOR MATRIX COMPUTERS
# =====================================================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bandwidth(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = series.rolling(window=window).mean()
    rstd = series.rolling(window=window).std()
    upper_band = sma + (num_std * rstd)
    lower_band = sma - (num_std * rstd)
    bandwidth = (upper_band - lower_band) / (sma + 1e-9)
    return upper_band, lower_band, bandwidth


# =====================================================================
# QUANTITATIVE HIGH-VOLATILITY PENNY FILTER MODULE
# =====================================================================
def screen_high_risk_penny_stocks(candidate_symbols: list) -> list:
    if not candidate_symbols:
        return []
    aggressive_targets = []
    try:
        snapshots = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=candidate_symbols))
        start_time = datetime.now(timezone.utc) - timedelta(days=45)
        historical_bars = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=candidate_symbols, timeframe=TimeFrame.Day, start=start_time
        ))
        historical_df = historical_bars.df
    except Exception as e:
        print(f"   [!] Filter Fetch Exception: {e}")
        return []

    for symbol in candidate_symbols:
        try:
            if symbol not in snapshots or symbol not in historical_df.index.levels[0]:
                continue
            latest_price = snapshots[symbol].latest_trade.price
            if not (1.00 <= latest_price <= 15.00):
                continue
            df_daily = historical_df.loc[symbol].copy()
            current_daily_volume = df_daily['volume'].iloc[-1]
            avg_20day_volume = df_daily['volume'].rolling(window=20).mean().iloc[-1]
            if current_daily_volume < 800000:
                continue
            relative_volume = current_daily_volume / avg_20day_volume
            if relative_volume < 2.0:
                continue
            sma_20 = df_daily['close'].rolling(window=20).mean().iloc[-1]
            if latest_price < sma_20:
                continue
            print(
                f"   💥 [Aggressive Setup Detected] {symbol} | Price: ${latest_price:.2f} | RelVol: {relative_volume:.2f}x")
            aggressive_targets.append(symbol)
        except Exception:
            continue
    return aggressive_targets


# =====================================================================
# MACRO NEWS EVALUATION MODULE
# =====================================================================
def analyze_asset_sentiment(symbol: str) -> float:
    request_params = NewsRequest(symbols=symbol, limit=5)
    try:
        news_response = news_client.get_news(request_params)
        if not news_response:
            return 0.0
        if isinstance(news_response, dict):
            articles = news_response.get('news', [])
        else:
            articles = getattr(news_response, 'news', [])
    except Exception:
        return 0.0

    net_score, scored_articles_count = 0.0, 0
    for article in articles:
        if isinstance(article, str): continue
        headline_text = article.get('headline', '') if isinstance(article, dict) else getattr(article, 'headline', '')
        summary_text = article.get('summary', '') if isinstance(article, dict) else getattr(article, 'summary', '')
        text_vector = f"{headline_text} {summary_text}".lower().strip()
        if not text_vector: continue
        bearish_hits = sum(1 for word in BEARISH_KEYWORDS if word in text_vector)
        bullish_hits = sum(1 for word in BULLISH_KEYWORDS if word in text_vector)
        if bearish_hits > bullish_hits:
            net_score -= 0.4
            scored_articles_count += 1
        elif bullish_hits > bearish_hits:
            net_score += 0.4
            scored_articles_count += 1
        else:
            scored_articles_count += 1
    if scored_articles_count == 0: return 0.0
    return round(max(-1.0, min(1.0, net_score / scored_articles_count)), 2)


# =====================================================================
# AUTONOMOUS SYSTEM INITIALIZATION & MONITORING LOOP
# =====================================================================
def execute_autonomous_trading_agent():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Active Scanning Run Executing...")
    try:
        open_positions = trading_client.get_all_positions()
        current_position_count = len(open_positions)
        active_portfolio_symbols = [pos.symbol for pos in open_positions]
    except Exception:
        current_position_count, open_positions, active_portfolio_symbols = 0, [], []

    # Emergency News Liquidation
    for position in open_positions:
        symbol = position.symbol
        if analyze_asset_sentiment(symbol) <= -0.4:
            try:
                trading_client.cancel_all_orders()
                from alpaca.trading.requests import MarketOrderRequest
                trading_client.submit_order(
                    MarketOrderRequest(symbol=symbol, qty=float(position.qty), side=OrderSide.SELL,
                                       time_in_force=TimeInForce.DAY))

                subject = f"🚨 EMERGENCY LIQUIDATION: {symbol}"
                body = f"The trading engine has forcefully closed all allocations in {symbol} due to a critical bearish drop in real-time sentiment metrics."
                send_email_alert(subject, body)

                if symbol in active_portfolio_symbols: active_portfolio_symbols.remove(symbol)
                current_position_count -= 1
            except Exception:
                pass

    if current_position_count >= MAX_OPEN_POSITIONS: return

    try:
        actives_response = screener_client.get_most_actives(MostActivesRequest(top=50, by=MostActivesBy.VOLUME))
        dynamic_candidates = [item['symbol'] if isinstance(item, dict) else item.symbol for item in (
            actives_response.get('most_actives', []) if isinstance(actives_response,
                                                                   dict) else actives_response.most_actives)]
    except Exception:
        dynamic_candidates = ["AAPL", "MSFT", "NVDA", "AMD", "SOFI", "SPY", "QQQ", "NFLX", "AMZN", "TSLA"]

    unowned_candidates = [ticker for ticker in dynamic_candidates if ticker not in active_portfolio_symbols]
    scan_targets = screen_high_risk_penny_stocks(unowned_candidates)
    if not scan_targets: return

    try:
        snapshots = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=scan_targets))
        start_time = datetime.now(timezone.utc) - timedelta(hours=4)
        historical_bars = data_client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=scan_targets, timeframe=TimeFrame.Minute, start=start_time))
        historical_df = historical_bars.df
    except Exception:
        return

    for symbol in scan_targets:
        try:
            if symbol not in historical_df.index.levels[0]: continue
            df = historical_df.loc[symbol].copy()
            latest_price = snapshots[symbol].latest_trade.price
            df.loc[datetime.now(timezone.utc)] = [latest_price, latest_price, latest_price, latest_price, 0, 0, 0]

            df['fast_ema'] = df['close'].ewm(span=FAST_EMA_PERIOD, adjust=False).mean()
            df['slow_ema'] = df['close'].ewm(span=SLOW_EMA_PERIOD, adjust=False).mean()
            is_bullish_crossover = (df['fast_ema'].iloc[-2] <= df['slow_ema'].iloc[-2]) and (
                        df['fast_ema'].iloc[-1] > df['slow_ema'].iloc[-1])

            df['rsi_14'] = calculate_rsi(df['close'], period=14)
            current_rsi = df['rsi_14'].iloc[-1]
            is_rsi_surging = 60.0 <= current_rsi <= 85.0

            df['bb_upper'], _, _ = calculate_bollinger_bandwidth(df['close'], window=20)
            is_volatility_breaking_out = latest_price >= df['bb_upper'].iloc[-1]

            _, _, df['macd_hist'] = calculate_macd(df['close'])
            is_macd_accelerating = (df['macd_hist'].iloc[-1] > df['macd_hist'].iloc[-2] > df['macd_hist'].iloc[-3])

            # 5. Immediate Volume Acceleration Gate
            # REDUCED THRESHOLD: Changed from 1.2 to 0.95 to make entries easier
            # and prevent volume bottlenecks on valid crossovers.
            avg_volume_5m = df['volume'].rolling(window=5).mean().iloc[-1]
            is_volume_accelerating = df['volume'].iloc[-1] >= (avg_volume_5m * 0.95)
            sentiment = analyze_asset_sentiment(symbol)

            print(
                f" -> {symbol} | Price: ${latest_price:.2f} | EMA Cross: {'YES' if is_bullish_crossover else 'NO'} | BB Breakout: {'YES' if is_volatility_breaking_out else 'NO'} | MACD Accel: {'YES' if is_macd_accelerating else 'NO'} | RSI: {current_rsi:.1f}| Vol Accel: {'YES' if is_volume_accelerating else 'NO'} | Sent: {sentiment}")

            if is_bullish_crossover and is_rsi_surging and is_volatility_breaking_out and is_macd_accelerating and is_volume_accelerating and (
                    sentiment >= -0.2):
                shares_to_buy = int(TRADE_VALUE_USD // latest_price)
                if shares_to_buy == 0: continue

                effective_take_profit = 0.10 if latest_price < 5.00 else TAKE_PROFIT_PERCENT
                effective_stop_loss = 0.04 if latest_price < 5.00 else STOP_LOSS_PERCENT

                entry_limit_price = round(latest_price + 0.02, 2)
                target_profit_price = round(entry_limit_price * (1.0 + effective_take_profit), 2)
                stop_loss_trigger = round(entry_limit_price * (1.0 - effective_stop_loss), 2)

                if stop_loss_trigger >= entry_limit_price:
                    stop_loss_trigger = round(entry_limit_price - 0.03, 2)

                bracket_order = LimitOrderRequest(
                    symbol=symbol, qty=shares_to_buy, limit_price=entry_limit_price, side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=target_profit_price),
                    stop_loss=StopLossRequest(stop_price=stop_loss_trigger)
                )

                submitted_order = trading_client.submit_order(bracket_order)
                print(
                    f"[+] High-Risk Bracket Order ID {submitted_order.id} transmitted. Status: {submitted_order.status}")

                # TRIGGER IMMEDIATE EMAIL NOTIFICATION
                email_subject = f"🚀 QUANT ALGORITHM: BUY ORDER EXECUTED ({symbol})"
                email_body = (
                    f"Predictive Technical Convergence Triggered Successfully.\n\n"
                    f"Asset: {symbol}\n"
                    f"Action: Limit Buy Entered\n"
                    f"Execution Limit Price: ${entry_limit_price:.2f}\n"
                    f"Position Volume: {shares_to_buy} shares\n"
                    f"Capital Allocated: ${shares_to_buy * entry_limit_price:.2f} USD\n\n"
                    f"--- RISK CONTROL BOUNDARIES ---\n"
                    f"Target Profit Price: ${target_profit_price:.2f} (+{effective_take_profit * 100:.1f}%)\n"
                    f"Stop-Loss Protection Price: ${stop_loss_trigger:.2f} (-{effective_stop_loss * 100:.1f}%)\n\n"
                    f"Status: Bracket Transmitted Successfully."
                )
                send_email_alert(email_subject, email_body)

                current_position_count += 1
                if current_position_count >= MAX_OPEN_POSITIONS: break
        except Exception as e:
            print(f" [!] Order entry failed for {symbol} | Error Details: {e}")
            continue


if __name__ == "__main__":
    print("=" * 60)
    print("LAUNCHING HIGH-VELOCITY PREDICTIVE MOMENTUM ENGINE WITH EMAIL")
    print("=" * 60)

    while True:
        start_time = time.time()
        execute_autonomous_trading_agent()
        elapsed = time.time() - start_time
        print(f"[*] Block cycle completed in {elapsed:.2f} seconds.\nAwaiting 15-second scanning block window...\n")
        time.sleep(15)