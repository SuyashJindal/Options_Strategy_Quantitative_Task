### Applied XGBoost on features(technical indicators columns) + owned features+ signal(own house)
### Backtested on different lots size(first backtesting framework :- doing dynamic lots size trade based on capital fraction & premium value)
### Also tested on different lot size to check performance for risk assesment scenario:-( 750,75,50) 
### Mainly done feature engineering on technical indicators like ml and technical voting giving weights . 
### Getting around 80 percent accuracy on composite signals on validating frame and test date .
### Mainly on different lot sizs(dynamic) getting around  2500 percent return.
### While fixed lot size and using 1 lot size :- making 80,60,and 30 percent capital on different lot size while backtesting showing how much capital we deployed per trade to access returns.

Initial Capital:    ₹200,000
Final Capital:      ₹5,223,126
Total P&L:          ₹5,023,126.40

**Total Return:2511.56%**
The Composite Signal is an ensemble-based trading decision.
It blends three sources:

Core signal (50%) – baseline strategy rule

Machine learning vote (30%) – predictive model

Technical indicator vote (20%) – RSI & crossover rules

The weighted score decides the final action:

Buy if score ≥ 0.4

Sell if score ≤ -0.4

Hold otherwise

This approach reduces reliance on any single model/indicator and balances rule-based logic + data-driven learning + technical confirmation
