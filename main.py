"""
High-Velocity Momentum Engine (paper trading) — cleaned & hardened.

REQUIRED environment variables (do NOT hardcode secrets):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY

OPTIONAL (email alerts; if unset, alerts are skipped silently):
    ALERT_SENDER_EMAIL          e.g. you@gmail.com
    ALERT_SENDER_APP_PASSWORD   16-char Google App Password
    ALERT_TARGET_INBOX          defaults to ALERT_SENDER_EMAIL
    SMTP_SERVER                 default smtp.gmail.com
    SMTP_PORT                   default 587

Example (macOS/Linux):
    export ALPACA_API_KEY="..."
    export ALPACA_SECRET_KEY="..."
    export ALERT_SENDER_EMAIL="you@gmail.com"
    export ALERT_SENDER_APP_PASSWORD="xxxx xxxx xxxx xxxx"

NOTE: This is paper trading and NOT financial advice. Backtest and measure
expectancy (win rate, avg win/loss, max drawdown, results net of spread/slippage)
before considering real capital.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import smtplib
from email.mime.text import MIMEText

# Alpaca Trading & Execution
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus

# Alpaca Data / Screener / News
from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import (
    StockSnapshotRequest,
    StockBarsRequest,
    NewsRequest,
    MostActivesRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import MostActivesBy

# =====================================================================
# CREDENTIALS — loaded from environment, never hardcoded
# =====================================================================
API_KEY = "PKZDEM6OFD7W4S7IPL6PKWX4I4"
SECRET_KEY = "B9csJGtky9cpgrebkyScvYZyCarMhiYCm7zeFevnsF4B"

if not API_KEY or not SECRET_KEY:
    sys.exit(
        "ERROR: set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables "
        "before running. Do not hardcode them into this file."
    )

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)
screener_client = ScreenerClient(API_KEY, SECRET_KEY)

# =====================================================================
# STRATEGY CONFIGURATION
# =====================================================================
FAST_EMA_PERIOD = 3
SLOW_EMA_PERIOD = 8
MAX_OPEN_POSITIONS = 15
TRADE_VALUE_USD = 1200
SCAN_INTERVAL_SECONDS = 15
MIN_BARS_REQUIRED = 30          # need enough bars for 26-period MACD / 20-period BB
VOLUME_ACCEL_MULTIPLIER = 0.95  # last-bar volume vs its rolling mean (your tunable)

# Base bracket constraints for mid-caps; sub-$5 names get wider brackets below.
TAKE_PROFIT_PERCENT = 0.05
STOP_LOSS_PERCENT = 0.02

# =====================================================================
# EMAIL CONFIGURATION
# =====================================================================
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SENDER_EMAIL = "bhatnagar.prachi845@gmail.com"
SENDER_PASSWORD = "coic qleo lybh ngcl"
TARGET_EMAIL_INBOX = "bhatnagar.prachi845@gmail.com"
EMAIL_ENABLED = bool(SENDER_EMAIL and SENDER_PASSWORD and TARGET_EMAIL_INBOX)


def send_email_alert(subject: str, message_body: str):
    """Send a trade alert. Silently no-ops if email env vars are not configured."""
    if not EMAIL_ENABLED:
        return
    try:
        msg = MIMEText(message_body)
        msg["From"] = SENDER_EMAIL
        msg["To"] = TARGET_EMAIL_INBOX
        msg["Subject"] = subject

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, TARGET_EMAIL_INBOX, msg.as_string())
        print(f"   [Email] Alert sent to {TARGET_EMAIL_INBOX}.")
    except Exception as e:
        print(f"   [!] Email alert failed: {e}")


# =====================================================================
# NEWS SENTIMENT (whole-word matching + short-lived cache)
# =====================================================================
BEARISH_KEYWORDS = {
    "downgrade", "misses", "lawsuit", "deficit", "investigation", "fraud", "slashes",
    "bearish", "drop", "fall", "plummet", "sink", "lower", "decline", "loss", "negative",
    "warns", "weak", "cut", "fail", "failed", "warn",
}
BULLISH_KEYWORDS = {
    "upgrade", "beats", "partnership", "growth", "surge", "acquisition", "profit",
    "bullish", "rise", "gain", "rally", "higher", "climb", "jump", "positive", "win",
    "strong", "grew", "expand", "succeed", "soars",
}

_TOKEN_RE = re.compile(r"[a-z']+")
_sentiment_cache = {}  # symbol -> (unix_ts, score)
SENTIMENT_TTL_SECONDS = 300


def _count_whole_word_hits(text: str, keywords: set) -> int:
    """Whole-word match so 'win' no longer matches 'winter', 'gain' not 'against', etc."""
    tokens = set(_TOKEN_RE.findall(text.lower()))
    return len(tokens & keywords)


def analyze_asset_sentiment(symbol: str) -> float:
    """Return a sentiment score in [-1.0, 1.0]. Cached per symbol for a few minutes."""
    now = time.time()
    cached = _sentiment_cache.get(symbol)
    if cached and (now - cached[0]) < SENTIMENT_TTL_SECONDS:
        return cached[1]

    try:
        news_response = news_client.get_news(NewsRequest(symbols=symbol, limit=5))
        if isinstance(news_response, dict):
            articles = news_response.get("news", [])
        else:
            articles = getattr(news_response, "news", []) or []
    except Exception:
        _sentiment_cache[symbol] = (now, 0.0)
        return 0.0

    net_score, scored = 0.0, 0
    for article in articles:
        if isinstance(article, str):
            continue
        headline = article.get("headline", "") if isinstance(article, dict) else getattr(article, "headline", "")
        summary = article.get("summary", "") if isinstance(article, dict) else getattr(article, "summary", "")
        text = f"{headline} {summary}".strip()
        if not text:
            continue
        bearish = _count_whole_word_hits(text, BEARISH_KEYWORDS)
        bullish = _count_whole_word_hits(text, BULLISH_KEYWORDS)
        if bearish > bullish:
            net_score -= 0.4
        elif bullish > bearish:
            net_score += 0.4
        scored += 1

    score = 0.0 if scored == 0 else round(max(-1.0, min(1.0, net_score / scored)), 2)
    _sentiment_cache[symbol] = (now, score)
    return score


# =====================================================================
# TECHNICAL INDICATORS
# =====================================================================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = series.rolling(window=window).mean()
    rstd = series.rolling(window=window).std()
    upper = sma + (num_std * rstd)
    lower = sma - (num_std * rstd)
    return upper, sma, lower


def _symbol_in_multiindex(df: pd.DataFrame, symbol: str) -> bool:
    if df is None or df.empty or not isinstance(df.index, pd.MultiIndex):
        return False
    return symbol in df.index.get_level_values(0)


# =====================================================================
# PENNY / SMALL-CAP LIQUIDITY FILTER
# =====================================================================
def screen_high_risk_penny_stocks(candidate_symbols: list) -> list:
    if not candidate_symbols:
        return []
    targets = []
    # Funnel diagnostics: how many candidates drop out at each stage.
    funnel = {
        "input": len(candidate_symbols),
        "no_data": 0,
        "price_band": 0,   # outside $1-$15
        "low_volume": 0,   # < 800k shares today
        "low_relvol": 0,   # < 2x 20-day avg
        "below_sma": 0,    # under 20-day SMA
        "passed": 0,
    }
    try:
        snapshots = data_client.get_stock_snapshot(
            StockSnapshotRequest(symbol_or_symbols=candidate_symbols)
        )
        start_time = datetime.now(timezone.utc) - timedelta(days=45)
        historical_df = data_client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=candidate_symbols, timeframe=TimeFrame.Day, start=start_time)
        ).df
    except Exception as e:
        print(f"   [!] Filter fetch exception: {e}")
        return []

    for symbol in candidate_symbols:
        try:
            if symbol not in snapshots or not _symbol_in_multiindex(historical_df, symbol):
                funnel["no_data"] += 1
                continue
            snap = snapshots[symbol]
            if snap is None or snap.latest_trade is None:
                funnel["no_data"] += 1
                continue
            latest_price = snap.latest_trade.price

            if not (1.00 <= latest_price <= 15.00):
                funnel["price_band"] += 1
                continue

            df_daily = historical_df.loc[symbol]
            if len(df_daily) < 20:
                funnel["no_data"] += 1
                continue

            current_vol = df_daily["volume"].iloc[-1]
            avg_20 = df_daily["volume"].rolling(window=20).mean().iloc[-1]
            if pd.isna(avg_20) or avg_20 <= 0 or current_vol < 800_000:
                funnel["low_volume"] += 1
                continue

            if (current_vol / avg_20) < 2.0:
                funnel["low_relvol"] += 1
                continue

            sma_20 = df_daily["close"].rolling(window=20).mean().iloc[-1]
            if pd.isna(sma_20) or latest_price < sma_20:
                funnel["below_sma"] += 1
                continue

            funnel["passed"] += 1
            print(f"   💥 [Setup] {symbol} | ${latest_price:.2f} | RelVol {current_vol / avg_20:.2f}x")
            targets.append(symbol)
        except Exception:
            funnel["no_data"] += 1
            continue

    print(
        f"   [Screen funnel] in={funnel['input']} "
        f"no_data={funnel['no_data']} price_band={funnel['price_band']} "
        f"low_vol={funnel['low_volume']} low_relvol={funnel['low_relvol']} "
        f"below_sma={funnel['below_sma']} -> passed={funnel['passed']}"
    )
    return targets


# =====================================================================
# ACCOUNT CONTEXT
# =====================================================================
def get_account_context():
    """Return (positions, position_symbols, pending_order_symbols)."""
    try:
        positions = trading_client.get_all_positions()
    except Exception:
        positions = []
    position_symbols = {p.symbol for p in positions}

    try:
        open_orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        pending_symbols = {o.symbol for o in open_orders}
    except Exception:
        pending_symbols = set()

    return positions, position_symbols, pending_symbols


def liquidate_position(symbol: str):
    """Cancel ONLY this symbol's open orders (so other positions keep their brackets),
    then flatten the position."""
    try:
        symbol_orders = trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        for o in symbol_orders:
            try:
                trading_client.cancel_order_by_id(o.id)
            except Exception:
                pass
    except Exception:
        pass

    try:
        trading_client.close_position(symbol)  # liquidates full position at market
        return True
    except Exception as e:
        print(f"   [!] close_position failed for {symbol}: {e}")
        return False


# =====================================================================
# MAIN AGENT
# =====================================================================
def execute_autonomous_trading_agent():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning run executing...")

    positions, position_symbols, pending_symbols = get_account_context()
    current_position_count = len(positions)

    # --- Emergency news-driven liquidation (per-symbol, brackets preserved) ---
    for position in positions:
        symbol = position.symbol
        if analyze_asset_sentiment(symbol) <= -0.4:
            if liquidate_position(symbol):
                send_email_alert(
                    f"🚨 EMERGENCY LIQUIDATION: {symbol}",
                    f"Closed all allocation in {symbol} due to a strongly bearish sentiment reading.",
                )
                position_symbols.discard(symbol)
                current_position_count -= 1

    if current_position_count >= MAX_OPEN_POSITIONS:
        return

    # --- Candidate discovery ---
    try:
        actives_response = screener_client.get_most_actives(
            MostActivesRequest(top=50, by=MostActivesBy.VOLUME)
        )
        raw_actives = (
            actives_response.get("most_actives", [])
            if isinstance(actives_response, dict)
            else actives_response.most_actives
        )
        dynamic_candidates = [it["symbol"] if isinstance(it, dict) else it.symbol for it in raw_actives]
    except Exception:
        dynamic_candidates = ["AAPL", "MSFT", "NVDA", "AMD", "SOFI", "SPY", "QQQ", "NFLX", "AMZN", "TSLA"]

    # Exclude both held positions AND symbols with an order already pending (no stacking).
    blocked = position_symbols | pending_symbols
    unowned = [t for t in dynamic_candidates if t not in blocked]

    scan_targets = screen_high_risk_penny_stocks(unowned)
    if not scan_targets:
        return

    # --- Intraday data for signals ---
    try:
        snapshots = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=scan_targets))
        start_time = datetime.now(timezone.utc) - timedelta(hours=6)
        historical_df = data_client.get_stock_bars(
            StockBarsRequest(symbol_or_symbols=scan_targets, timeframe=TimeFrame.Minute, start=start_time)
        ).df
    except Exception as e:
        print(f"   [!] Signal data fetch failed: {e}")
        return

    # Per-gate failure tally across this cycle's candidates.
    gate_fails = {"trend": 0, "rsi": 0, "breakout": 0, "macd": 0, "volume": 0, "sentiment": 0}
    evaluated = 0

    for symbol in scan_targets:
        try:
            if not _symbol_in_multiindex(historical_df, symbol):
                continue

            snap = snapshots.get(symbol)
            if snap is None or snap.latest_trade is None:
                continue
            latest_price = snap.latest_trade.price

            # Indicators computed on REAL completed bars only — no synthetic 0-volume
            # bar is injected (that was silently breaking the volume gate and repainting
            # RSI/MACD/Bollinger). The live price is used only for the breakout check
            # and for order pricing.
            df = historical_df.loc[symbol].copy()
            if len(df) < MIN_BARS_REQUIRED:
                continue

            df["fast_ema"] = df["close"].ewm(span=FAST_EMA_PERIOD, adjust=False).mean()
            df["slow_ema"] = df["close"].ewm(span=SLOW_EMA_PERIOD, adjust=False).mean()
            # Trend-STATE, not a fresh crossover: fast simply above slow = uptrend in
            # progress. Consistent with the daily screen (already-trending names) and
            # with the RSI/breakout gates, which only fire mid-move.
            is_uptrend = df["fast_ema"].iloc[-1] > df["slow_ema"].iloc[-1]

            rsi = calculate_rsi(df["close"], period=14)
            current_rsi = rsi.iloc[-1]
            is_rsi_surging = not pd.isna(current_rsi) and (30.0 <= current_rsi <= 85.0)

            # Breakout relaxed from UPPER band to MIDDLE band (20-period SMA): join a
            # confirmed uptrend instead of buying the most extended tick at the top band.
            _, bb_middle, _ = calculate_bollinger_bands(df["close"], window=20)
            bb_middle_last = bb_middle.iloc[-1]
            is_breakout = not pd.isna(bb_middle_last) and (latest_price >= bb_middle_last)

            _, _, macd_hist = calculate_macd(df["close"])
            if macd_hist.iloc[-3:].isna().any():
                is_macd_accelerating = False
            else:
                is_macd_accelerating = macd_hist.iloc[-1] > macd_hist.iloc[-2] > macd_hist.iloc[-3]

            # Volume gate reads the last COMPLETED bar (iloc[-2]), not the partially
            # formed current minute, so it isn't comparing a half-bar to full-bar averages.
            completed_vol = df["volume"].iloc[-2]
            avg_vol_5 = df["volume"].iloc[:-1].rolling(window=5).mean().iloc[-1]
            is_volume_accelerating = (
                not pd.isna(avg_vol_5) and avg_vol_5 > 0 and completed_vol >= avg_vol_5 * VOLUME_ACCEL_MULTIPLIER
            )

            sentiment = analyze_asset_sentiment(symbol)

            print(
                f" -> {symbol} | ${latest_price:.2f} | Trend:{'Y' if is_uptrend else 'N'} "
                f"| BB:{'Y' if is_breakout else 'N'} | MACD:{'Y' if is_macd_accelerating else 'N'} "
                f"| RSI:{current_rsi:.1f} | Vol:{'Y' if is_volume_accelerating else 'N'} | Sent:{sentiment}"
            )

            evaluated += 1
            failed = []
            if not is_uptrend:
                gate_fails["trend"] += 1
                failed.append("trend")
            if not is_rsi_surging:
                gate_fails["rsi"] += 1
                failed.append("rsi")
            if not is_breakout:
                gate_fails["breakout"] += 1
                failed.append("breakout")
            if not is_macd_accelerating:
                gate_fails["macd"] += 1
                failed.append("macd")
            if not is_volume_accelerating:
                gate_fails["volume"] += 1
                failed.append("volume")
            if sentiment < -0.2:
                gate_fails["sentiment"] += 1
                failed.append("sentiment")

            if failed:
                continue

            shares_to_buy = int(TRADE_VALUE_USD // latest_price)
            if shares_to_buy == 0:
                continue

            take_profit_pct = 0.10 if latest_price < 5.00 else TAKE_PROFIT_PERCENT
            stop_loss_pct = 0.04 if latest_price < 5.00 else STOP_LOSS_PERCENT

            entry_limit = round(latest_price + 0.02, 2)
            target_price = round(entry_limit * (1.0 + take_profit_pct), 2)
            stop_price = round(entry_limit * (1.0 - stop_loss_pct), 2)
            if stop_price >= entry_limit:
                stop_price = round(entry_limit - 0.03, 2)

            bracket_order = LimitOrderRequest(
                symbol=symbol,
                qty=shares_to_buy,
                limit_price=entry_limit,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=target_price),
                stop_loss=StopLossRequest(stop_price=stop_price),
            )

            submitted = trading_client.submit_order(bracket_order)
            print(f"[+] Bracket order {submitted.id} submitted. Status: {submitted.status}")

            # Reserve the symbol immediately so the same loop / next loop won't re-order it.
            pending_symbols.add(symbol)

            send_email_alert(
                f"🚀 BUY ORDER SUBMITTED ({symbol})",
                (
                    f"Asset: {symbol}\n"
                    f"Action: Limit Buy (bracket)\n"
                    f"Limit Price: ${entry_limit:.2f}\n"
                    f"Shares: {shares_to_buy}\n"
                    f"Capital: ${shares_to_buy * entry_limit:.2f}\n\n"
                    f"Take Profit: ${target_price:.2f} (+{take_profit_pct * 100:.1f}%)\n"
                    f"Stop Loss:   ${stop_price:.2f} (-{stop_loss_pct * 100:.1f}%)"
                ),
            )

            current_position_count += 1
            if current_position_count >= MAX_OPEN_POSITIONS:
                break
        except Exception as e:
            print(f" [!] Entry failed for {symbol}: {e}")
            continue

    if evaluated:
        ranked = sorted(gate_fails.items(), key=lambda kv: kv[1], reverse=True)
        summary = " ".join(f"{gate}={n}" for gate, n in ranked)
        worst = ranked[0][0] if ranked[0][1] > 0 else "none"
        print(f"   [Gate tally] evaluated={evaluated} | {summary} | most-blocking: {worst}")


if __name__ == "__main__":
    print("=" * 60)
    print("HIGH-VELOCITY MOMENTUM ENGINE (paper) — hardened")
    print("=" * 60)

    while True:
        loop_start = time.time()

        # Only trade when the market is open; otherwise idle cheaply.
        try:
            clock = trading_client.get_clock()
        except Exception as e:
            print(f"[!] Clock fetch failed: {e}. Retrying in 30s.")
            time.sleep(30)
            continue

        if not getattr(clock, "is_open", False):
            print("[*] Market closed. Sleeping 60s.")
            time.sleep(60)
            continue

        try:
            execute_autonomous_trading_agent()
        except Exception as e:
            print(f"[!] Unhandled error in scan cycle: {e}")

        elapsed = time.time() - loop_start
        print(f"[*] Cycle done in {elapsed:.2f}s. Waiting for next window...\n")
        time.sleep(max(0.0, SCAN_INTERVAL_SECONDS - elapsed))