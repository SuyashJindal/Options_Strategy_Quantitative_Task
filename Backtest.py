import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class Backtest:
    def __init__(self, spot_data, options_data, starting_capital=200000, sl_pct=0.015, tp_pct=0.03, force_exit_time='15:15',lot_size =75):

        self.spot_df = spot_data
        self.options_df = options_data
        self.capital = starting_capital
        self.initial_capital = starting_capital
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.force_exit_time = pd.to_datetime(force_exit_time).time()
        self.trade_log = []
        self.position = None
        self.lot_size = lot_size
        self.equity_curve = []
        self._prepare_data()

    def _prepare_data(self):
        self.spot_df['datetime'] = pd.to_datetime(self.spot_df['datetime'])
        self.spot_df['closest_expiry'] = pd.to_datetime(self.spot_df['closest_expiry']).dt.date
        self.options_df['datetime'] = pd.to_datetime(self.options_df['datetime'])
        self.options_df['expiry_date'] = pd.to_datetime(self.options_df['expiry_date']).dt.date

        # set index but also keep datetime as a column
        self.options_df = self.options_df.set_index('datetime', drop=False).sort_index()

    def run(self):
        for index, spot_row in self.spot_df.iterrows():
            current_time = spot_row['datetime']

            # --- 1. Check for Exit Conditions for the current open position ---
            if self.position:
                self._check_exit_conditions(current_time)

            # --- 2. Check for Entry Conditions if no position is open ---
            if not self.position:
                signal = spot_row['signal']
                if signal in ['Buy', 'Sell']:
                    self._enter_position(spot_row)
            self.equity_curve.append((current_time, self.capital))
        
        if self.position:
            last_spot_row = self.spot_df.iloc[-1]
            self._close_position(last_spot_row['datetime'], self.position['entry_price'], "End of Data")

    def _check_exit_conditions(self, current_time):
        """Checks if the stop-loss, take-profit, or EOD exit has been triggered."""
        try:
            # Find the current market price of the option we are holding
            option_market_data = self.options_df.loc[current_time]
            # Handle cases where multiple options match by taking the first one
            current_option_row = option_market_data[option_market_data['ticker'] == self.position['ticker']]

            if not current_option_row.empty:
                current_price = current_option_row.iloc[0]['close']
                entry_price = self.position['entry_price']

                # PnL is calculated from the perspective of a seller. Profit occurs when the option price drops.
                pnl_percentage = (entry_price - current_price) / entry_price

                # Define exit conditions
                stop_loss_hit = pnl_percentage <= -self.sl_pct
                take_profit_hit = pnl_percentage >= self.tp_pct
                eod_exit = current_time.time() >= self.force_exit_time

                if stop_loss_hit:
                    self._close_position(current_time, current_price, "Stop-Loss")
                elif take_profit_hit:
                    self._close_position(current_time, current_price, "Take-Profit")
                elif eod_exit:
                    self._close_position(current_time, current_price, "EOD Exit")

        except KeyError:
            # This handles timestamps present in spot_df but not in options_df.
            pass


    def _enter_position(self, spot_row):
        """Enters a new trade by selling an ATM option based on the signal."""
        spot_close = spot_row['close']
        current_time = spot_row['datetime']
        expiry = spot_row['closest_expiry']
        signal = spot_row['signal']

        # Determine option type ('PE' for Buy signal -> Sell Put) and find ATM strike
        option_type = 'PE' if signal == 'Buy' else 'CE'
        atm_strike = round(spot_close / 50) * 50

        try:
            # Find the specific option contract to sell
            options_at_time = self.options_df.loc[current_time]
            target_option = options_at_time[
                (options_at_time['expiry_date'] == expiry) &
                (options_at_time['strike_price'] == atm_strike) &
                (options_at_time['option_type'] == option_type)
            ]

            if not target_option.empty:
                option_to_sell = target_option.iloc[0]
                self.position = {
                    'entry_time': current_time,
                    'ticker': option_to_sell['ticker'],
                    'entry_price': option_to_sell['close'],
                    'option_type': option_type,
                    'strike': atm_strike,
                    'signal': signal
                }
        except KeyError:
            # This handles timestamps present in spot_df but not in options_df.
            pass

    def _close_position(self, exit_time, exit_price, reason):
        """Closes the current position, calculates PnL, and logs the trade."""
        # PnL = Entry Price - Exit Price for a short (sell) position
       #pnl = self.position['entry_price'] - exit_price
        pnl = (self.position['entry_price'] - exit_price) * self.lot_size
        self.capital += pnl
        
        self.trade_log.append({
            'Entry Time': self.position['entry_time'],
            'Exit Time': exit_time,
            'Ticker': self.position['ticker'],
            'Signal': self.position['signal'],
            'Option Type': self.position['option_type'],
            'Strike': self.position['strike'],
            'Entry Price': self.position['entry_price'],
            'Exit Price': exit_price,
            'PnL': pnl,
            'Exit Reason': reason,
            'Lot Size': self.lot_size   
        })
        # Reset position to None to allow for new entries
        self.position = None

    def results(self):
        print("--- Backtest Results ---")
        if not self.trade_log:
            print("No trades were executed during the simulation.")
            return

        log_df = pd.DataFrame(self.trade_log)
        # Display all columns without truncation
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(log_df)

        print("\n--- Performance Summary ---")
        print(f"Starting Capital: ₹{self.initial_capital:,.2f}")
        print(f"Ending Capital:   ₹{self.capital:,.2f}")
        
        total_pnl = self.capital - self.initial_capital
        pnl_color = "\033[92m" if total_pnl > 0 else "\033[91m" # Green for profit, Red for loss
        print(f"Total PnL:        {pnl_color}₹{total_pnl:,.2f}\033[0m")

        if not log_df.empty:
            wins = log_df[log_df['PnL'] > 0]
            losses = log_df[log_df['PnL'] <= 0]
            win_rate = len(wins) / len(log_df)
            
            print(f"\nTotal Trades: {len(log_df)}")
            print(f"Win Rate:     {win_rate:.2%}")
            if not wins.empty:
                print(f"Avg. Profit:  ₹{wins['PnL'].mean():,.2f}")
            if not losses.empty:
                print(f"Avg. Loss:    ₹{losses['PnL'].mean():,.2f}")
        log_df.to_csv("trades.csv", index=False)

        # --- Performance metrics ---
        equity_df = pd.DataFrame(self.equity_curve, columns=["datetime", "equity"]).set_index("datetime")
        equity_df.to_csv("equity_curve.csv")

        equity_df["cummax"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = equity_df["equity"] / equity_df["cummax"] - 1

        total_return = (self.capital - self.initial_capital) / self.initial_capital
        returns = equity_df["equity"].pct_change().dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if not returns.empty else 0
        max_dd = equity_df["drawdown"].min()

        metrics = {
            "Starting Capital": self.initial_capital,
            "Ending Capital": self.capital,
            "Total Return": total_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd
        }
        pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)

        # --- Plots ---
        plt.figure(figsize=(10, 5))
        equity_df["equity"].plot(title="Equity Curve")
        plt.ylabel("Portfolio Value")
        plt.savefig("equity_curve.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        equity_df["drawdown"].plot(title="Drawdown")
        plt.ylabel("Drawdown")
        plt.savefig("drawdown.png")
        plt.close()

        print("\n--- Metrics ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# --- Execution ---
if __name__ == '__main__':
        backtest = Backtest(
            spot_data=spot_with_signals,
            options_data=df_options_data,
            starting_capital=200000,
            sl_pct=0.015,
            tp_pct=0.03,
            lot_size=750
        )
        backtest.run()

        # Display the final results
        backtest.results()
