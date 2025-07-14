import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from enum import Enum

# Define order types
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"

# Define order
@dataclass
class Order:
    asset: str
    size: float
    price: Optional[float]
    order_type: OrderType
    timestamp: pd.Timestamp

# Define position
@dataclass
class Position:
    asset: str
    size: float
    entry_price: float

    @property
    def value(self, current_price: float) -> float:
        if not isinstance(current_price, (int, float)) or np.isnan(current_price) or np.isinf(current_price):
            raise ValueError(f"Invalid current_price for {self.asset}: {current_price}")
        return self.size * current_price

# Data feed class to hold OHLCV data for one asset
class DataFeed:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataFeed with a pandas DataFrame.
        Expected columns: 'Open', 'High', 'Low', 'Close', 'Volume'
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        self.df = df.copy()
        # Ensure no NaN or inf in price columns
        for col in ['Open', 'High', 'Low', 'Close']:
            if self.df[col].isna().any() or np.isinf(self.df[col]).any():
                raise ValueError(f"DataFrame column {col} contains NaN or inf values")

    def get_data_up_to(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Return data up to the given timestamp."""
        return self.df.loc[:timestamp]

# Base strategy class
class Strategy:
    def __init__(self):
        self.data: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.cash: float = 0.0
        self.indicators: Dict[str, Dict[str, Any]] = {}

    def init(self):
        """Initialize indicators and other setup."""
        pass

    def next(self):
        """Define trading logic for each timestamp."""
        pass

    def buy(self, asset: str, size: float, price: Optional[float] = None, order_type: str = 'market'):
        """Place a buy order."""
        if asset not in self.data:
            raise ValueError(f"Asset {asset} not found in data")
        order = Order(
            asset=asset,
            size=size,
            price=price,
            order_type=OrderType(order_type),
            timestamp=self.data[asset].index[-1]
        )
        self.orders.append(order)

    def sell(self, asset: str, size: float, price: Optional[float] = None, order_type: str = 'market'):
        """Place a sell order."""
        if asset not in self.data:
            raise ValueError(f"Asset {asset} not found in data")
        order = Order(
            asset=asset,
            size=-size,
            price=price,
            order_type=OrderType(order_type),
            timestamp=self.data[asset].index[-1]
        )
        self.orders.append(order)

    def close(self, asset: str):
        """Close position for the specified asset."""
        if asset in self.positions and self.positions[asset].size != 0:
            size = abs(self.positions[asset].size)
            if self.positions[asset].size > 0:
                self.sell(asset, size)
            else:
                self.buy(asset, size)

    def I(self, func: Callable, asset: str, *args, **kwargs) -> np.ndarray:
        """Compute an indicator for the specified asset."""
        result = func(self.data[asset], *args, **kwargs)
        if asset not in self.indicators:
            self.indicators[asset] = {}
        indicator_name = f"{func.__name__}_{len(self.indicators[asset])}"
        self.indicators[asset][indicator_name] = result
        return result

# Dual Momentum Strategy
class DualMomentumStrategy(Strategy):
    def __init__(self, assets, safe_asset, lookback_days=252, rebalance_frequency='monthly', top_k=1):
        super().__init__()
        self.assets = assets
        self.safe_asset = safe_asset
        self.lookback_days = lookback_days
        self.rebalance_frequency = rebalance_frequency.lower()
        if self.rebalance_frequency not in ['weekly', 'monthly']:
            raise ValueError("rebalance_frequency must be 'weekly' or 'monthly'")
        self.top_k = top_k
        self.calculation_days = None

    def set_calculation_days(self, calculation_days):
        self.calculation_days = calculation_days

    def next(self):
        current_timestamp = self.data[self.assets[0]].index[-1]
        if current_timestamp not in self.calculation_days:
            return

        # Calculate returns over lookback period
        returns = {}
        for asset in self.assets:
            if len(self.data[asset]) > self.lookback_days:
                try:
                    past_close = self.data[asset].iloc[-self.lookback_days - 1]['Close']
                    current_close = self.data[asset].ilocs[-1]['Close']
                    ret = (current_close / past_close) - 1
                    returns[asset] = ret
                except (IndexError, KeyError):
                    returns[asset] = np.nan
            else:
                returns[asset] = np.nan

        # Filter assets with positive absolute momentum
        positive_assets = [asset for asset in self.assets if returns.get(asset, np.nan) > 0 and not np.isnan(returns[asset])]

        if len(positive_assets) > 0:
            # Select top_k assets with highest relative momentum
            sorted_assets = sorted(positive_assets, key=lambda x: returns[x], reverse=True)
            selected_assets = sorted_assets[:min(self.top_k, len(sorted_assets))]
            weight = 1.0 / max(1, len(selected_assets))
            target_allocation = {asset: weight for asset in selected_assets}
        else:
            target_allocation = {self.safe_asset: 1.0}

        # Compute current equity
        equity = self.cash
        for asset, position in self.positions.items():
            if asset in self.data and len(self.data[asset]) > 0:
                try:
                    current_price = self.data[asset].iloc[-1]['Close']
                    equity += position.value(current_price)
                except (IndexError, KeyError, ValueError):
                    continue

        if equity <= 0:
            return

        # Place orders to adjust to target allocation
        all_assets = set(self.assets + [self.safe_asset])
        for asset in all_assets:
            if asset in target_allocation:
                weight = target_allocation[asset]
                desired_value = equity * weight
                try:
                    current_price = self.data[asset].iloc[-1]['Close']
                    desired_size = desired_value / current_price
                except (IndexError, KeyError):
                    continue
            else:
                desired_size = 0

            # Adjust position
            current_size = self.positions.get(asset, Position(asset=asset, size=0, entry_price=0)).size
            size_diff = desired_size - current_size
            if abs(size_diff) > 1e-6:
                if size_diff > 0:
                    self.buy(asset=asset, size=size_diff)
                else:
                    self.sell(asset=asset, size=-size_diff)

# Backtest class
class Backtest:
    def __init__(
        self,
        data: Dict[str, DataFeed],
        strategy_cls: type,
        strategy_params: dict = {},
        cash: float = 10000.0,
        commission: float = 0.0,
        trade_on_close: bool = False
    ):
        """Initialize backtest with data, strategy, and parameters."""
        self.data = data
        self.strategy = strategy_cls(**strategy_params)
        self.strategy.cash = cash
        self.commission = commission
        self.trade_on_close = trade_on_close
        self.equity_curve = []
        self.trades = []

    def run(self):
        """Run the backtest and return statistics."""
        # Align timestamps across all assets
        timestamps = sorted(set.union(*[set(df.df.index) for df in self.data.values()]))
        if not timestamps:
            raise ValueError("No valid timestamps found in data")

        # Set calculation days for strategies that need it
        if hasattr(self.strategy, 'set_calculation_days'):
            df_timestamps = pd.Series(timestamps, index=timestamps)
            if hasattr(self.strategy, 'rebalance_frequency'):
                if self.strategy.rebalance_frequency == 'monthly':
                    calculation_days = df_timestamps.resample('ME').last().dropna().index
                elif self.strategy.rebalance_frequency == 'weekly':
                    calculation_days = df_timestamps.resample('W').last().dropna().index
                else:
                    raise ValueError("Invalid rebalance_frequency")
                self.strategy.set_calculation_days(calculation_days)

        # Initialize strategy
        self.strategy.data = {asset: df.get_data_up_to(timestamps[0]) for asset, df in self.data.items()}
        self.strategy.init()

        # Main backtest loop
        for i, t in enumerate(timestamps[:-1]):
            # Update data up to current timestamp
            self.strategy.data = {asset: df.get_data_up_to(t) for asset, df in self.data.items()}
            
            # Call strategy's next method
            self.strategy.next()

            # Process orders
            next_timestamp = timestamps[i + 1]
            for order in self.strategy.orders[:]:
                self._process_order(order, next_timestamp)

            # Update equity
            equity = self._compute_equity(next_timestamp)
            self.equity_curve.append((t, equity))

            # Clear orders
            self.strategy.orders = []

        # Compute final statistics
        stats = self._compute_stats()
        return stats

    def _process_order(self, order: Order, next_timestamp: pd.Timestamp):
        """Process an order and update positions and cash."""
        asset = order.asset
        df = self.data[asset].df
        if next_timestamp not in df.index:
            return

        if order.order_type == OrderType.MARKET:
            fill_price = df.loc[next_timestamp]['Close' if self.trade_on_close else 'Open']
        elif order.order_type == OrderType.LIMIT and order.price is not None:
            next_bar = df.loc[next_timestamp]
            if order.size > 0:
                fill_price = order.price if next_bar['Low'] <= order.price <= next_bar['High'] else None
            else:
                fill_price = order.price if next_bar['Low'] <= order.price <= next_bar['High'] else None
            if fill_price is None:
                return
        else:
            return

        # Calculate cost including commission
        cost = abs(order.size) * fill_price * (1 + self.commission)
        if order.size > 0 and cost > self.strategy.cash:
            return

        # Update cash
        self.strategy.cash -= order.size * fill_price * (1 + self.commission)

        # Update position
        if asset not in self.strategy.positions:
            self.strategy.positions[asset] = Position(asset=asset, size=0, entry_price=0)
        
        position = self.strategy.positions[asset]
        if position.size * order.size < 0:
            close_size = min(abs(position.size), abs(order.size))
            if position.size > 0:
                profit = close_size * (fill_price - position.entry_price)
            else:
                profit = close_size * (position.entry_price - fill_price)
            self.strategy.cash += profit
            position.size += order.size
        else:
            if position.size == 0:
                position.entry_price = fill_price
            else:
                total_size = position.size + order.size
                if total_size != 0:
                    position.entry_price = (position.size * position.entry_price + order.size * fill_price) / total_size
            position.size += order.size

        # Record trade
        self.trades.append({
            'asset': asset,
            'size': order.size,
            'price': fill_price,
            'timestamp': next_timestamp
        })

    def _compute_equity(self, timestamp: pd.Timestamp) -> float:
        """Compute total equity at the given timestamp."""
        equity = self.strategy.cash
        for asset, position in self.strategy.positions.items():
            if position.size != 0 and timestamp in self.data[asset].df.index:
                try:
                    current_price = self.data[asset].df.loc[timestamp]['Close']
                    if not isinstance(current_price, (int, float)) or np.isnan(current_price) or np.isinf(current_price):
                        print(f"Warning: Invalid price for {asset} at {timestamp}: {current_price}")
                        continue
                    equity += position.value(current_price)
                except (KeyError, IndexError, ValueError) as e:
                    print(f"Error computing equity for {asset} at {timestamp}: {e}")
                    continue
        return max(equity, 0)

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute performance statistics."""
        if not self.equity_curve:
            return {'Final Equity': self.strategy.cash, 'Total Return': 0.0, 'Annualized Return': 0.0,
                    'Max Drawdown': 0.0, 'Sharpe Ratio': np.nan, 'Number of Trades': 0}

        equity_series = pd.Series([eq for _, eq in self.equity_curve], index=[t for t, _ in self.equity_curve])
        returns = equity_series.pct_change().dropna()
        
        stats = {
            'Final Equity': equity_series.iloc[-1],
            'Total Return': (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100 if equity_series.iloc[0] != 0 else 0.0,
            'Annualized Return': ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / len(equity_series)) - 1) * 100 if equity_series.iloc[0] != 0 else 0.0,
            'Max Drawdown': self._compute_max_drawdown(equity_series),
            'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else np.nan,
            'Number of Trades': len(self.trades)
        }
        return stats

    def _compute_max_drawdown(self, equity_series: pd.Series) -> float:
        """Compute maximum drawdown as a percentage."""
        if len(equity_series) == 0 or equity_series.iloc[0] == 0:
            return 0.0
        rolling_max = equity_series.cummax()
        drawdowns = (rolling_max - equity_series) / rolling_max
        return drawdowns.max() * 100

    def optimize(self, maximize: str = 'Final Equity', **param_ranges):
        """Optimize strategy parameters."""
        from itertools import product
        param_combinations = list(product(*[param_ranges[param] for param in param_ranges]))
        best_stats = None
        best_params = None
        best_value = float('-inf')

        for params in param_combinations:
            param_dict = dict(zip(param_ranges.keys(), params))
            for param, value in param_dict.items():
                setattr(self.strategy, param, value)
            stats = self.run()
            value = stats[maximize]
            if value > best_value:
                best_value = value
                best_stats = stats
                best_params = param_dict

        return best_stats, best_params

# Generate realistic sample data
def generate_sample_data(start_date: str, end_date: str, assets: List[str], seed: int = 42) -> Dict[str, pd.DataFrame]:
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    data = {}
    for asset in assets:
        returns = np.random.normal(0, 0.01, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        prices = np.maximum(prices, 1e-6)
        prices = np.where(np.isfinite(prices), prices, 100.0)
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'High': prices * (1 + np.random.normal(0.01, 0.005, len(dates))),
            'Low': prices * (1 - np.random.normal(0.01, 0.005, len(dates))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = np.maximum(df[col], 1e-6)
            df[col] = np.where(np.isfinite(df[col]), df[col], 100.0)
        data[asset] = df
    return data

# Example usage
if __name__ == "__main__":
    sample_data = generate_sample_data(
        start_date='2019-01-01',
        end_date='2020-12-31',
        assets=['SPY', 'EFA', 'AGG']
    )

    bt = Backtest(
        data={asset: DataFeed(df) for asset, df in sample_data.items()},
        strategy_cls=DualMomentumStrategy,
        strategy_params={
            'assets': ['SPY', 'EFA'],
            'safe_asset': 'AGG',
            'lookback_days': 252,
            'rebalance_frequency': 'monthly',
            'top_k': 1
        },
        cash=10000,
        commission=0.001
    )
    stats = bt.run()
    print("Backtest Results:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
