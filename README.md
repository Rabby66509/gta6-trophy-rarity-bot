# GTA VI Trophy Rarity Forecaster

Entry for **gta6-trophy-rarity-forecast-01**: forecast every PlayStation 5
trophy's Exophase earn rate at 2026-11-26T00:00Z, as five quantiles per
trophy. Score is mean pinball loss on the logit scale; lower is better.
Prize needs a loss strictly below the best house entry.

## The model (`main.py`, Python 3.13 stdlib only, no network)

An earn rate is **earned / tracked**, and the history gives both numbers,
so each is modelled separately:

- **Structure over trend.** `log(earned)` and `log(tracked)` are each fit
  by least squares against `ln(1 + hours since release)`, and the median
  forecast is `100 * E_target / N_target`. When tracked players grow
  faster than earned counts the rate falls even while players keep
  earning — a plain rate trend cannot see that dilution.
- **Flat-history refresh prior.** Launch night's reads were all identical
  (638 players tracked) until Exophase refreshed to 3,868 players and
  every rate but the 100% one fell. A case with no movement in its
  history has no trend to fit, so the median assumes one such refresh
  with earned counts unchanged: `rate * 638 / 3868`. The 100% trophy
  stays at 100%.
- **Honest spread.** Quantiles are the median shifted on the logit scale.
  The spread combines the trend fit's residual noise, a random-walk term
  growing with the horizon, and the uncertainty of tracked-player growth;
  a flat history gets a wide spread (0.7 logit) covering "no refresh"
  and "refresh plus some earning" around the dilution median.
- The forecast blends toward the last read when the history is short,
  the median's move is capped, and the output is fully deterministic.

## The board

Practice losses of the current top entries (dev phase, 2026-11-19):
#1 vivivexa 0.114, #2 amber-comet-97 0.122, #3 amber-lynx-35 0.148.
House baselines score ~1.51 on the practice cases, so beating the top
means predicting the launch-night refresh (the dilution prior) rather
than just persisting the last read.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per
case (55 trophies, five quantiles each, at most 65,536 bytes). Two runs
on the same input are byte-identical.
