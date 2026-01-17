import hashlib
import hmac
import time
from decimal import Decimal, ROUND_DOWN
import requests
from urllib.parse import urlencode
from config import TESTNET_BASE_URL, API_KEY, API_SECRET


class BinanceFuturesClient:
    """币安期货测试网API客户端"""

    def __init__(self):
        self.base_url = TESTNET_BASE_URL
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key
        })
        self.time_offset = 0
        self._symbol_filters = {}
        self._sync_time()

    def _sync_time(self):
        """同步服务器时间"""
        try:
            server_time = self.get_server_time()["serverTime"]
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
            # 不打印，避免编码问题
        except Exception as e:
            # 不打印，避免编码问题
            self.time_offset = 0

    def _sign(self, params: dict) -> str:
        """生成签名"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False, max_retries: int = 3):
        """发送请求（带重试）"""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        if signed:
            params["timestamp"] = int(time.time() * 1000) + self.time_offset
            params["signature"] = self._sign(params)

        last_error = None
        for attempt in range(max_retries):
            try:
                if method == "GET":
                    response = self.session.get(url, params=params, timeout=30)
                elif method == "POST":
                    response = self.session.post(url, params=params, timeout=30)
                elif method == "DELETE":
                    response = self.session.delete(url, params=params, timeout=30)
                else:
                    raise ValueError(f"不支持的HTTP方法: {method}")
                
                # 成功，返回结果
                if not response.ok:
                    try:
                        error_data = response.json()
                        raise Exception(f"API错误 {error_data.get('code')}: {error_data.get('msg')}")
                    except ValueError:
                        response.raise_for_status()
                
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_error = f"请求超时: {endpoint}"
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2秒, 4秒, 6秒
                    print(f"Timeout, retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    # 重新签名（时间戳会过期）
                    if signed:
                        params["timestamp"] = int(time.time() * 1000) + self.time_offset
                        params["signature"] = self._sign(params)
                else:
                    raise Exception(last_error)
                    
            except requests.exceptions.RequestException as e:
                last_error = f"网络错误: {str(e)}"
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Network error, retrying in {wait_time}s ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    if signed:
                        params["timestamp"] = int(time.time() * 1000) + self.time_offset
                        params["signature"] = self._sign(params)
                else:
                    raise Exception(last_error)
        
        raise Exception(last_error or "请求失败")

    # ========== 公共接口 ==========
    def get_server_time(self):
        """获取服务器时间"""
        return self._request("GET", "/fapi/v1/time")

    def get_exchange_info(self):
        """获取交易规则和交易对信息"""
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def get_ticker_price(self, symbol: str = None):
        """获取最新价格"""
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/ticker/price", params)

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 100):
        """获取K线数据"""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return self._request("GET", "/fapi/v1/klines", params)

    # ========== 账户接口 ==========
    def get_account(self):
        """获取账户信息"""
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_balance(self):
        """获取账户余额"""
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def get_positions(self):
        """获取持仓信息"""
        return self._request("GET", "/fapi/v2/positionRisk", signed=True)

    def set_leverage(self, symbol: str, leverage: int):
        """设置杠杆倍数"""
        params = {"symbol": symbol, "leverage": leverage}
        return self._request("POST", "/fapi/v1/leverage", params, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED"):
        """设置保证金模式 (ISOLATED/CROSSED)"""
        params = {"symbol": symbol, "marginType": margin_type}
        return self._request("POST", "/fapi/v1/marginType", params, signed=True)

    # ========== 交易接口 ==========
    def get_symbol_info(self, symbol: str):
        """获取交易对信息（精度、最小数量等）"""
        info = self.get_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                return s
        return None

    def get_symbol_filters(self, symbol: str) -> dict:
        """获取交易对过滤参数（tick/step/minNotional）"""
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]
        info = self.get_symbol_info(symbol) or {}
        filters = {f.get("filterType"): f for f in info.get("filters", [])}
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        min_notional = filters.get("MIN_NOTIONAL", {})
        data = {
            "tick_size": float(price_filter.get("tickSize", 0) or 0),
            "step_size": float(lot_filter.get("stepSize", 0) or 0),
            "min_qty": float(lot_filter.get("minQty", 0) or 0),
            "min_notional": float(min_notional.get("notional", min_notional.get("minNotional", 0)) or 0),
        }
        self._symbol_filters[symbol] = data
        return data

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                    price: float = None, stop_price: float = None, time_in_force: str = None,
                    reduce_only: bool = False):
        """下单
        
        reduce_only:
            True  -> 只减仓，不会反向开新仓（用于平仓，避免保证金不足）
        """
        def _round_to_step(value: float, step: float) -> float:
            if not step or step <= 0:
                return value
            d_value = Decimal(str(value))
            d_step = Decimal(str(step))
            return float((d_value // d_step) * d_step)

        # 获取精度信息
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info:
            qty_precision = symbol_info.get("quantityPrecision", 3)
            price_precision = symbol_info.get("pricePrecision", 2)
            filters = {f.get("filterType"): f for f in symbol_info.get("filters", [])}
            tick_size = float(filters.get("PRICE_FILTER", {}).get("tickSize", 0) or 0)
            step_size = float(filters.get("LOT_SIZE", {}).get("stepSize", 0) or 0)

            if step_size > 0:
                quantity = _round_to_step(quantity, step_size)
            else:
                quantity = round(quantity, qty_precision)
            if price is not None:
                if tick_size > 0:
                    price = _round_to_step(price, tick_size)
                else:
                    price = round(price, price_precision)
            if stop_price is not None:
                if tick_size > 0:
                    stop_price = _round_to_step(stop_price, tick_size)
                else:
                    stop_price = round(stop_price, price_precision)

        # Basic safety checks after rounding
        if quantity is None or quantity <= 0:
            raise Exception("订单数量过小，未满足最小步进")
        if price is not None and price <= 0:
            raise Exception("订单价格过小，未满足最小跳动")
        if stop_price is not None and stop_price <= 0:
            raise Exception("触发价过小，未满足最小跳动")

        params = {
            "symbol": symbol,
            "side": side,  # BUY / SELL
            "type": order_type,  # LIMIT / MARKET / STOP / TAKE_PROFIT 等
            "quantity": quantity
        }
        if reduce_only:
            params["reduceOnly"] = True
        if price:
            params["price"] = price
        if stop_price:
            params["stopPrice"] = stop_price
        if time_in_force:
            params["timeInForce"] = time_in_force
        elif order_type == "LIMIT":
            params["timeInForce"] = "GTC"

        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_order(self, symbol: str, order_id: int = None, client_order_id: str = None):
        """取消订单"""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._request("DELETE", "/fapi/v1/order", params, signed=True)

    def cancel_all_orders(self, symbol: str):
        """取消所有订单"""
        params = {"symbol": symbol}
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params, signed=True)

    def get_open_orders(self, symbol: str = None):
        """获取当前挂单"""
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    def get_order(self, symbol: str, order_id: int = None, client_order_id: str = None):
        """查询订单"""
        params = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        return self._request("GET", "/fapi/v1/order", params, signed=True)

    def get_all_orders(self, symbol: str, limit: int = 50):
        """获取所有订单历史"""
        params = {"symbol": symbol, "limit": limit}
        return self._request("GET", "/fapi/v1/allOrders", params, signed=True)

    def get_account_trades(self, symbol: str, limit: int = 50):
        """获取成交历史（包含手续费）"""
        params = {"symbol": symbol, "limit": limit}
        return self._request("GET", "/fapi/v1/userTrades", params, signed=True)

    def get_income_history(self, income_type: str = None, limit: int = 100):
        """获取收益历史（包含手续费、资金费率等）"""
        params = {"limit": limit}
        if income_type:
            params["incomeType"] = income_type  # COMMISSION, FUNDING_FEE, REALIZED_PNL, etc.
        return self._request("GET", "/fapi/v1/income", params, signed=True)
