# spot_price.py
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest

def get_spy_spot(stock_client: StockHistoricalDataClient) -> float:
    snap = stock_client.get_stock_snapshot(
        StockSnapshotRequest(symbol_or_symbols="SPY")
    )["SPY"]

    if snap.latest_trade:
        return float(snap.latest_trade.price)
    if snap.latest_quote:
        return (snap.latest_quote.bid_price + snap.latest_quote.ask_price) / 2

    raise RuntimeError("SPY price unavailable")
