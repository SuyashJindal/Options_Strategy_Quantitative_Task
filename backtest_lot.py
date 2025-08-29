import pandas as pd
import numpy as np
from datetime import datetime, time
from strategy_ml import df_options_data
from strategy_ml import spot_with_signals
class Backtest:
    def __init__(self, starting_capital=200000, stop_loss=0.015, take_profit=0.03, force_exit_time="15:15"):
        """
        Initializes the backtesting framework.
        """
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.force_exit_time = time.fromisoformat(force_exit_time)
        self.trades = []
        self.current_positions = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        
    def prepare_data(self, df_options_data):
        """
        Pre-processes options data for faster lookups.
        """
        df_options_data = df_options_data.copy()
        df_options_data['datetime'] = pd.to_datetime(df_options_data['datetime'])
        df_options_data['expiry_date'] = pd.to_datetime(df_options_data['expiry_date'])
        df_options_data['datetime_key'] = df_options_data['datetime'].dt.floor('min')
        df_options_data['expiry_key'] = df_options_data['expiry_date'].dt.date
        df_options_data = df_options_data.set_index(['datetime_key', 'expiry_key', 'option_type'])
        df_options_data = df_options_data.sort_index()
        return df_options_data
    
    def find_atm_option_fast(self, df_options_indexed, target_datetime, spot_price, expiry_date, option_type):
        """
        Fast ATM option lookup that ensures only a single row (Series) is returned.
        """
        try:
            target_key = (target_datetime.floor('min'), expiry_date, option_type)
            if target_key in df_options_indexed.index:
                options_subset = df_options_indexed.loc[target_key]
                
                if isinstance(options_subset, pd.Series):
                    return options_subset
                else:
                    strike_diff = abs(options_subset['strike_price'] - spot_price)
                    min_idx = strike_diff.idxmin()
                    closest_option = options_subset.loc[min_idx]
                    
                    # If .loc returns a DataFrame (due to duplicate index), take the first row
                    if isinstance(closest_option, pd.DataFrame):
                        return closest_option.iloc[0]
                    else:
                        return closest_option
            else:
                return None
        except (KeyError, IndexError):
            return None
    
    def check_exit_conditions(self, entry_price, current_price, entry_time, current_time):
        """
        Checks if any exit conditions are met.
        """
        if entry_price == 0: return False, None # Avoid division by zero
        pnl_pct = (entry_price - current_price) / entry_price
        if pnl_pct >= self.take_profit:
            return True, "TAKE_PROFIT"
        if pnl_pct <= -self.stop_loss:
            return True, "STOP_LOSS"
        if current_time.time() >= self.force_exit_time:
            return True, "FORCE_EXIT"
        if current_time.time() >= time(15, 30):
            return True, "EOD"
        return False, None
    
    def execute_trade(self, signal, spot_price, option_data, entry_datetime):
        """
        Executes a trade based on the signal.
        """
        if option_data is None:
            return
            
        option_type = 'PE' if signal == 'Buy' else 'CE'
            
        trade = {
            'entry_datetime': entry_datetime,
            'signal': signal,
            'option_type': option_type,
            'strike_price': float(option_data['strike_price']),
            'entry_price': float(option_data['close']),
            'spot_price': spot_price,
            'expiry_date': option_data['expiry_date'].date(),
            'status': 'OPEN'
        }
        
        self.current_positions.append(trade)
        
    def update_positions_fast(self, df_options_indexed, current_datetime):
        """
        Fast position updates using vectorized operations.
        """
        if not self.current_positions:
            return
            
        positions_to_close = []
        
        for i, position in enumerate(self.current_positions):
            current_option_data = self.find_atm_option_fast(
                df_options_indexed,
                current_datetime,
                position['spot_price'],
                position['expiry_date'],
                position['option_type']
            )
            
            if current_option_data is not None:
                current_price = float(current_option_data['close'])
                
                should_exit, exit_reason = self.check_exit_conditions(
                    position['entry_price'],
                    current_price,
                    position['entry_datetime'],
                    current_datetime
                )
                
                if should_exit:
                    lot_size = 75
                    pnl = (position['entry_price'] - current_price)*lot_size
                    pnl_pct = (pnl / position['entry_price']) if position['entry_price'] != 0 else 0
                    
                    closed_trade = position.copy()
                    closed_trade.update({
                        'exit_datetime': current_datetime,
                        'exit_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'status': 'CLOSED'
                    })
                    
                    self.trades.append(closed_trade)
                    positions_to_close.append(i)
                    
                    self.total_trades += 1
                    self.total_pnl += pnl
                    self.current_capital += pnl
                    
                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
        
        for i in reversed(positions_to_close):
            self.current_positions.pop(i)
    
    def validate_dataframes(self, spot_with_signals, df_options_data):
        print("DATAFRAME VALIDATION")
        print("="*50)
        print(f"spot_with_signals shape: {spot_with_signals.shape}")
        print(f"df_options_data shape: {df_options_data.shape}")
        required_spot_cols = ['datetime', 'close', 'composite_signal', 'closest_expiry']
        if not all(col in spot_with_signals.columns for col in required_spot_cols):
             raise ValueError("spot_with_signals is missing required columns.")
        print("\n✅ Required columns validation passed!")

    def run_backtest(self, spot_with_signals, df_options_data):
        """
        Runs the complete backtest.
        """
        self.validate_dataframes(spot_with_signals, df_options_data)
        
        print("\nStarting backtest...")
        print("Preparing and indexing options data for fast lookups...")
        df_options_indexed = self.prepare_data(df_options_data)
        print("Indexing complete.")
        print(f"Initial Capital: ₹{self.starting_capital:,.2f}")
        print("-" * 50)
        
        for index, row in spot_with_signals.iterrows():
            current_datetime = pd.to_datetime(row['datetime'])
            spot_price = row['close']
            signal = row['composite_signal']
            closest_expiry = pd.to_datetime(row['closest_expiry']).date()

            self.update_positions_fast(df_options_indexed, current_datetime)
            
            if signal in ['Buy', 'Sell']:
                option_type = 'PE' if signal == 'Buy' else 'CE'
                
                atm_option = self.find_atm_option_fast(
                    df_options_indexed,
                    current_datetime,
                    spot_price,
                    closest_expiry,
                    option_type
                )
                
                if atm_option is not None:
                    self.execute_trade(signal, spot_price, atm_option, current_datetime)
                    strike_price = float(atm_option['strike_price'])
                    close_price = float(atm_option['close'])
                    print(f"{current_datetime}: {signal} signal - Sold {option_type} {strike_price} at ₹{close_price:.2f}")
                else:
                    print(f"{current_datetime}: {signal} signal - No matching option found")
        
        print("\nClosing remaining positions at end of backtest...")
        if self.current_positions:
            last_datetime = pd.to_datetime(spot_with_signals.iloc[-1]['datetime'])
            for position in self.current_positions:
                closed_trade = position.copy()
                closed_trade.update({
                    'exit_datetime': last_datetime,
                    'exit_price': position['entry_price'],
                    'pnl': 0, 'pnl_pct': 0,
                    'exit_reason': 'END_OF_BACKTEST',
                    'status': 'CLOSED'
                })
                self.trades.append(closed_trade)
            self.current_positions.clear()

    def get_performance_summary(self):
        """
        Generates performance summary.
        """
        if self.total_trades == 0:
            return "No trades executed during backtest period"
            
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        avg_win = np.mean([trade['pnl'] for trade in self.trades if trade['pnl'] > 0]) if self.winning_trades > 0 else 0
        avg_loss = np.mean([trade['pnl'] for trade in self.trades if trade['pnl'] < 0]) if self.losing_trades > 0 else 0
        final_capital = self.current_capital
        total_return = ((final_capital - self.starting_capital) / self.starting_capital) * 100
        
        summary = f"""
        BACKTEST PERFORMANCE SUMMARY
        {'='*50}
        Starting Capital: ₹{self.starting_capital:,.2f}
        Final Capital:    ₹{final_capital:,.2f}
        Total Return:     {total_return:.2f}%
        Total P&L:        ₹{self.total_pnl:,.2f}
        
        TRADE STATISTICS
        {'='*50}
        Total Trades:   {self.total_trades}
        Winning Trades: {self.winning_trades}
        Losing Trades:  {self.losing_trades}
        Win Rate:       {win_rate:.1f}%
        
        Average Win:  ₹{avg_win:.2f}
        Average Loss: ₹{avg_loss:.2f}
        """
        return summary
    
    def get_trades_dataframe(self):
        """
        Returns trades as a pandas DataFrame.
        """
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
backtest = Backtest(starting_capital=200000, stop_loss=0.015, take_profit=0.03)

# Run backtest
backtest.run_backtest(spot_with_signals, df_options_data)

# Get results
print(backtest.get_performance_summary())
trades_df = backtest.get_trades_dataframe()
trades_df.to_csv("final_backtest_trades.csv", index=False)
# if not trades_df.empty:
#     print(trades_df)
#     trades_df.to_csv("final_backtest_trades.csv", index=False)
