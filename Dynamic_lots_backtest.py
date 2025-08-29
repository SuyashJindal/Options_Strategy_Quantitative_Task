import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')
from strategy_ml import spot_with_signals
from strategy_ml import df_options_data
class OptionsBacktest:
    def __init__(self, initial_capital=200000, stop_loss_pct=1.5, take_profit_pct=3.0):
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.force_exit_time = time(15, 15)  # 15:15 IST
        
        # Track portfolio metrics
        self.capital = initial_capital
        self.trades = []
        self.portfolio_value = []
        
    def find_atm_option(self, df_options, datetime_val, expiry_date, spot_price, option_type):

        # Filter options for the specific datetime and expiry
        mask = (
            (df_options['datetime'] == datetime_val) & 
            (df_options['expiry_date'] == expiry_date) &
            (df_options['option_type'] == option_type)
        )
        filtered_options = df_options[mask]
        
        if filtered_options.empty:
            return None
            
        # Find the strike closest to spot price (ATM)
        filtered_options['strike_diff'] = abs(filtered_options['strike_price'] - spot_price)
        atm_option = filtered_options.loc[filtered_options['strike_diff'].idxmin()]
        
        return atm_option
    
    def calculate_pnl(self, entry_price, exit_price, option_type, lot_size=1):
     
        # For selling options: profit when option premium decreases
        pnl = (entry_price - exit_price) * lot_size
        return pnl
    
    def check_exit_conditions(self, entry_price, current_price, current_time):
        # Calculate current P&L percentage
        # For selling: profit when price goes down
        pnl_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Check stop loss (loss when option price increases)
        if pnl_pct <= -self.stop_loss_pct:
            return True, 'stop_loss'
        
        # Check take profit (profit when option price decreases)
        if pnl_pct >= self.take_profit_pct:
            return True, 'take_profit'
        
        # Check force exit time
        if current_time.time() >= self.force_exit_time:
            return True, 'force_exit'
        
        return False, None
    
    def run_backtest(self, spot_with_signals, df_options_data):
        print(f"Starting Backtest with Initial Capital: ₹{self.initial_capital:,.0f}")
        print(f"Stop Loss: {self.stop_loss_pct}% | Take Profit: {self.take_profit_pct}%")
        print("-" * 80)
        
        active_position = None
        
        for idx, row in spot_with_signals.iterrows():
            current_datetime = row['datetime']
            spot_price = row['close']
            signal = row.get('composite_signal', None)  # Assuming signal column name
            closest_expiry = row['closest_expiry']
            
            # Skip if no signal
            if pd.isna(signal) or signal not in ['Buy', 'Sell']:
                # Check for exit if position is active
                if active_position:
                    exit_needed, exit_reason = self.check_exit_conditions(
                        active_position['entry_price'],
                        active_position['current_price'],
                        current_datetime
                    )
                    
                    if exit_needed:
                        # Close position
                        self.close_position(active_position, exit_reason, current_datetime)
                        active_position = None
                continue
            
            # Close existing position if signal changes
            if active_position and active_position['signal'] != signal:
                self.close_position(active_position, 'signal_change', current_datetime)
                active_position = None
            
            # Open new position if no active position
            if not active_position:
                # Determine option type to sell based on signal
                if signal == 'Buy':
                    option_type = 'PE'  # Sell PUT when bullish
                elif signal == 'Sell':
                    option_type = 'CE'  # Sell CALL when bearish
                else:
                    continue
                
                # Find ATM option
                atm_option = self.find_atm_option(
                    df_options_data,
                    current_datetime,
                    closest_expiry,
                    spot_price,
                    option_type
                )
                
                if atm_option is not None:
                    # Calculate position size (using fixed lot size for simplicity)
                    lot_size = max(1, int(self.capital * 0.1 / atm_option['close']))  # 10% of capital
                    
                    active_position = {
                        'entry_datetime': current_datetime,
                        'signal': signal,
                        'option_type': option_type,
                        'strike_price': atm_option['strike_price'],
                        'expiry': closest_expiry,
                        'entry_price': atm_option['close'],
                        'current_price': atm_option['close'],
                        'spot_entry': spot_price,
                        'lot_size': lot_size
                    }
                    
                    print(f"\n{current_datetime} - NEW POSITION:")
                    print(f"  Signal: {signal} | Selling {option_type} @ Strike: {atm_option['strike_price']}")
                    print(f"  Entry Premium: ₹{atm_option['close']:.2f} | Lots: {lot_size}")
            
            # Update current price for active position
            elif active_position:
                current_option = self.find_atm_option(
                    df_options_data,
                    current_datetime,
                    active_position['expiry'],
                    active_position['strike_price'],
                    active_position['option_type']
                )
                
                if current_option is not None:
                    active_position['current_price'] = current_option['close']
                    
                    # Check exit conditions
                    exit_needed, exit_reason = self.check_exit_conditions(
                        active_position['entry_price'],
                        active_position['current_price'],
                        current_datetime
                    )
                    
                    if exit_needed:
                        self.close_position(active_position, exit_reason, current_datetime)
                        active_position = None
        
        # Close any remaining position at end
        if active_position:
            self.close_position(active_position, 'backtest_end', spot_with_signals.iloc[-1]['datetime'])
        
        return self.generate_report()
    
    def close_position(self, position, exit_reason, exit_datetime):
        pnl = self.calculate_pnl(
            position['entry_price'],
            position['current_price'],
            position['option_type'],
            position['lot_size']
        )
        
        pnl_pct = (pnl / (position['entry_price'] * position['lot_size'])) * 100
        
        trade_record = {
            'entry_datetime': position['entry_datetime'],
            'exit_datetime': exit_datetime,
            'signal': position['signal'],
            'option_type': position['option_type'],
            'strike': position['strike_price'],
            'entry_price': position['entry_price'],
            'exit_price': position['current_price'],
            'lot_size': position['lot_size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason
        }
        
        self.trades.append(trade_record)
        self.capital += pnl
        self.portfolio_value.append({'datetime': exit_datetime, 'capital': self.capital})
        
        print(f"\n{exit_datetime} - POSITION CLOSED ({exit_reason}):")
        print(f"  Exit Premium: ₹{position['current_price']:.2f}")
        print(f"  P&L: ₹{pnl:,.2f} ({pnl_pct:.2f}%)")
        print(f"  New Capital: ₹{self.capital:,.0f}")
    
    def generate_report(self):
        if not self.trades:
            return {
                'total_trades': 0,
                'message': 'No trades executed during backtest'
            }
        
        trades_df = pd.DataFrame(self.trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        max_profit = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        final_capital = self.capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        report = {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'exit_reasons': exit_reasons,
            'trades_df': trades_df
        }
        print("\n" + "=" * 80)
        print("BACKTEST SUMMARY")
        print("=" * 80)
        print(f"Initial Capital:    ₹{self.initial_capital:,.0f}")
        print(f"Final Capital:      ₹{final_capital:,.0f}")
        print(f"Total P&L:          ₹{total_pnl:,.2f}")
        print(f"Total Return:       {total_return:.2f}%")
        print(f"\nTotal Trades:       {total_trades}")
        print(f"Win Rate:           {win_rate:.2f}%")
        print(f"Profit Factor:      {profit_factor:.2f}")
        print(f"\nMax Profit:         ₹{max_profit:,.2f}")
        print(f"Max Loss:           ₹{max_loss:,.2f}")
        print(f"\nExit Reasons:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count}")
        
        return report
if __name__ == "__main__":
    spot_with_signals['datetime'] = pd.to_datetime(spot_with_signals['datetime'])
    spot_with_signals['closest_expiry'] = pd.to_datetime(spot_with_signals['closest_expiry'])
  
    df_options_data['datetime'] = pd.to_datetime(df_options_data['datetime'])
    df_options_data['expiry_date'] = pd.to_datetime(df_options_data['expiry_date'])
    print("Data preparation complete.")
    backtest = OptionsBacktest(
        initial_capital=200000,
        stop_loss_pct=1.5,
        take_profit_pct=3.0
    )

    results = backtest.run_backtest(spot_with_signals, df_options_data)
    print(results)
    if results and 'trades_df' in results and not results['trades_df'].empty:
        trades_df = results['trades_df']
        print("\nSaving trades to backtest_trades.csv")
        trades_df.to_csv('backtest_trades.csv', index=False) 
