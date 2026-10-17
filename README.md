# FX_26
FX Adaptive Algorithmic Bot

## Requirements: 
Python with numpy, pandas and scipy. It doesn’t need matplotlib or your period_converter.

## Data: 
your histdata.com 1-minute MT files. Put several months or years of one pair in a folder, since one year is thin for walk-forward testing.

- Run it from the command line:
  - python fx_adaptive_bot.py --data "C:/data/AUDJPY/**/DAT_MT_AUDJPY_M1_*.csv" --tf 30min --cost 0.000035
  - Optional flags are --train 4000, --step 1000 and --fix-rsi. --synthetic runs a built-in random-walk test. --train and --step are in bars: at 30min, 1000 bars is about one month, so the defaults mean “optimize on about 4 months, then trade the next month”.
- Or from Python:
python
import fx_adaptive_bot as f
df = f.load_histdata("C:/data/AUDJPY/**/DAT_MT_AUDJPY_M1_*.csv", "30min")
out = f.run(df, "30min", 0.000035, train_bars=4000, step_bars=1000)
print(out["table"])

## Output: 
a table of each strategy’s out-of-sample performance next to EQUAL-WEIGHT and ADAPTIVE, plus the average weights. It also writes chosen_parameters.csv (the parameters chosen at each fold) and weights_history.csv.
