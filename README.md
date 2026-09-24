# Portfolio Optimization: Does Trailing Return Signal Actually Beat Diversification?

## 1. Overview

This project builds and backtests a factor-based equity strategy ("My Strategy") and compares it against three benchmarks — **Equal Weight**, **Risk Parity**, and the **S&P 500** — across a universe of large-cap U.S. stocks from 2000 onward.

The core question it set out to answer: **can a systematic strategy built from trailing return and volatility data actually beat simple diversification?**

The short answer, after several iterations and a proper train/test split: **no — not reliably, for this universe, at this horizon.** The strategy's own final design (Section 8) ends up confirming this analytically, not just empirically. This README documents the full path to that conclusion, including the wrong turns, since the wrong turns are most of what makes the final result trustworthy.

---

## 2. Setup / How to Run

**Dependencies:** `numpy`, `pandas`, `yfinance`, `scipy`, `matplotlib`

```bash
pip install numpy pandas yfinance scipy matplotlib
python portfolio.py
```

**Key parameters** (set in `backtest()` / `one_dol_growth()`):
- `lookback` — trading days of history used to estimate returns/covariance (default 756, ≈3 years)
- `rebalance_freq` — trading days between rebalances (default 63, ≈1 quarter)

**A note on the ticker universe:** several large, well-known tickers were deliberately excluded or required care, because their price history doesn't actually extend back to 2000 under their current symbol — `GM` (the pre-2009 company was fully liquidated in bankruptcy; the current `GM` only re-IPO'd in Nov 2010), `V` and `MA` (IPO'd 2008 and 2006 respectively), and `GOOGL` specifically (the dual-class ticker didn't exist until 2014; `GOOG` is the older single-class symbol). Including these would have silently truncated the entire backtest's start date to whenever the latest-listed ticker began trading, since the data pipeline drops any date where *any* ticker is missing.

---

## 3. Attempt 1 — Classical Mean-Variance Optimization, and Why It Broke

The starting point was a standard textbook approach: estimate each stock's trailing mean return and covariance matrix, then use `scipy.optimize` to find the max-Sharpe portfolio (`max_sharpe_scipy`).

**The problem, quantified:** over a 756-day (3-year) window, the *statistical noise* in an estimated mean return is large relative to any realistic true skill edge:

```
standard error of a 3-year sample mean:  ≈ 13.75% annualized
a genuinely strong true skill edge:       3% annualized
noise-to-signal ratio:                    ≈ 4.6x
```

In other words, the noise in the input is several times larger than the differences the optimizer is trying to detect. Mean-variance optimizers are known in the literature as "estimation-error maximizers" (Michaud, 1989) for exactly this reason — they don't distinguish a real edge from a lucky draw, and the math (the tangency portfolio is proportional to `Σ⁻¹(μ − r_f)`) actively *amplifies* noise in `μ` when assets are correlated.

**Demonstrated directly:** in a synthetic test with 26 assets given *identical* true expected returns (i.e., zero real skill differences by construction), the uncapped optimizer still confidently put **100% of the portfolio into a single stock** — whichever one happened to draw favorable noise in the estimation window. This is the corner-solution problem, and it's the reason the entire mean-variance approach was abandoned in favor of something more robust.

*(`max_sharpe_scipy`, `min_volatility_for_target`, and `efficient_frontier` remain in the code as the original approach, superseded by everything below.)*

---

## 4. Attempt 2 — A Custom Composite-Score Strategy

Rather than continue patching the optimizer, `backtest()` was rebuilt from scratch around a composite factor score instead of direct optimization.

**The factors:**
- **Return** — trailing mean daily return (`mean_returns`)
- **Volatility** — trailing daily volatility, pulled from the diagonal of the covariance matrix (`volatility_list`)

**Combining them:** both factors are standardized into cross-sectional z-scores (how many standard deviations a stock sits from that quarter's *group* average), then combined:

```python
composite = (z_score_mean - z_score_std) / 2
```

The minus sign is deliberate — high volatility should count against a stock, not for it. Combining two only-partially-correlated signals is a form of noise reduction on its own: a stock that looks mediocre on trailing return but has notably low volatility can still score well overall, which a single-factor return-only ranking would miss entirely.

**Turning the score into weights — and a concentration risk found along the way:** the first design weighted stocks proportional to their raw composite score among those with a positive score. Testing this against a null case (again, no real skill differences) showed it concentrated dangerously — **average effective number of holdings dropped to 8.7 out of 26** (a standard diversification metric, where 26 would mean fully diversified), and **5.5 in the worst 5% of periods**. The fix: switch to **rank-weighting** (weight proportional to rank, not raw score, among positive-scoring stocks) — this keeps the "exclude weak stocks" behavior while removing the raw-score's sensitivity to outliers.

---

## 5. Making the Signal Statistically Honest — Shrinkage and a Self-Tuning Cap

Two further refinements, both designed to avoid introducing arbitrary tuned parameters.

**Shrinkage (`shrink_mean_returns`):** applies James-Stein shrinkage to pull each stock's noisy trailing mean return toward the cross-sectional grand mean, by an amount (`λ`, "shrinkage") computed directly from the data — how much apparent spread between stocks is explainable by pure sampling noise alone:

```python
shrinkage = clip((n - 3) * avg_sampling_variance / dispersion, 0, 1)
shrunk_return = grand_mean + (1 - shrinkage) * (raw_mean - grand_mean)
```

`λ` near 1 means "these differences look like noise, don't trust them." `λ` near 0 means "these differences look real."

**A cap tied to that same number, not a guess:** rather than pick an arbitrary concentration limit, the cap is derived from `λ`:

```python
C_t = 1 + (1 - shrinkage)      # ranges from 1 (fully distrustful) to 2 (fully trusting)
cap = C_t / k                  # k = number of stocks with a positive composite score that quarter
```

This isn't arbitrary: for rank-weighting, the *natural, uncapped* maximum weight any single stock can reach is exactly `2/(k+1)` — a closed-form fact, not an estimate. That pins the *meaningful* range for `C_t` to exactly `(1, 2)`: below 1 is undefined, at 1 the cap forces pure equal-weighting, and at 2 or above the cap stops doing anything at all (rank-weighting was never going to exceed that on its own). So `C_t` moving with `λ` inside that fixed window is a genuinely self-tuning cap, not a second hidden knob.

---

## 6. Testing It Properly — Train/Test Methodology

Every design choice above was made while looking at how it performed on the full history — which creates a real risk that the strategy was shaped to fit the noise of *that specific stretch* rather than finding anything genuine.

**The split used:**
- **Train (2000–2016):** free to inspect, reason about, and iterate on
- **Test (2016–2024):** held out, checked once. A short buffer before 2016 was included in the raw data pull purely to fill the lookback window for the first test-period rebalance — the reported results themselves only start once real out-of-sample data begins.

The discipline that matters: once the test period is checked and a conclusion drawn, it stops being a valid test if you go back, change something, and check it again — that just turns it into more training data. (This discipline slipped partway through the process — documented honestly in Section 10.)

---

## 7. Diagnosing Underperformance — From "Is It NVDA?" to a Bigger Discovery

**The initial result:** "My Strategy" underperformed *both* Equal Weight and Risk Parity, in both the train and test periods — and even trailed plain S&P 500 on a risk-adjusted basis.

**First hypothesis:** the composite score's volatility penalty was excluding NVDA — simultaneously one of the highest-return and highest-volatility stocks in the universe — too often, missing its gains. A diagnostic function (`diagnose_ticker_exclusion`) was built to test this directly: track a specific ticker's composite score and its actual subsequent-quarter return at every rebalance.

**A bug in the diagnostic itself, caught before trusting the result:** the exclusion check (`composite <= 0`) silently mishandled `NaN` values — any comparison with `NaN` evaluates to `False` in IEEE floating point, so `NaN` cases were being miscounted as "included." Once corrected, the NVDA-specific hypothesis actually got *weaker*: genuinely-included quarters had a *higher* average subsequent return (9.5%) than excluded quarters (4.4%) — the opposite of what the hypothesis needed.

**The real discovery, hiding in those same NaN values:** `composite` only becomes `NaN` when *every single stock* in the universe ends up with an identical shrunk return that quarter — which happens whenever `shrinkage` clips to exactly `1.0`. When every stock ties, z-scoring a constant array is a 0/0 division. And critically, this isn't about NVDA at all — when it happens, the entire portfolio scores `NaN`, nothing gets picked, and the strategy sits in **100% cash**.

This happened in **11 of 31 test-period quarters — over a third of the entire test window** spent completely out of the market, while every benchmark stayed fully invested. That's a far bigger structural driver of underperformance than any single-stock story.

---

## 8. The Final Design — Blending With Equal Weight by Confidence

**First fix attempted (and it made things worse):** guard the degenerate case by setting the return z-score to zero instead of `NaN`, letting the ranking fall back to volatility alone during those quarters. Tested against the same held-out period: **Sharpe dropped from 0.740 to 0.600**, and NVDA's exclusion rate went *up*, from 45% to 87% — because volatility-only ranking systematically favors low-vol stocks, and NVDA is reliably one of the highest-vol names in the universe. Holding cash, it turned out, was less bad than this particular fallback.

**The actual fix — blend the whole portfolio with Equal Weight, using `λ` directly:**

```python
weights_new = shrinkage * (equal_weight_vector) + (1 - shrinkage) * weights_new
weights_new = weights_new / weights_new.sum()
```

At `λ ≈ 1` (no trustworthy signal), the portfolio *is* Equal Weight — never cash. At `λ ≈ 0`, it's fully the strategy's own picks. No new free parameters — it reuses `λ`, which was already being computed and justified for a different purpose.

**A subtle bug this introduced, caught before trusting the result (again):** when `k = 0` (nobody had a positive composite score) *and* `shrinkage` wasn't exactly `1.0` — a case that, after the fix above, could occur for perfectly ordinary reasons, not just the degenerate tie case — the blend formula only summed to `shrinkage`, not `1`. A meaningful fraction of the portfolio was silently going untracked. Fixed with an explicit renormalization (`weights_new / weights_new.sum()`) right after the blend — a no-op on every normal rebalance, and a real fix on the ones that weren't.

---

## 9. Final Result and What It Means

With the blend in place, the headline diagnostic number:

```
average shrinkage across all rebalances: ≈ 0.91–0.93
```

**In plain terms: roughly 91–93% of the portfolio, on average, every single quarter, was Equal Weight.** Only 7–9% was ever the strategy's own view actually asserting itself.

That's not a disappointing result — it's the strategy's own purpose-built confidence measure directly confirming, analytically, what every earlier empirical attempt (Sections 3, 4, 7, 8) had already pointed to independently: **trailing 3-year mean returns don't carry reliable signal for this universe, at this horizon.**

This is also a well-established, respected pattern in real portfolio theory — the Black-Litterman model works on exactly this principle, blending a manager's own view with a stable market-based prior in proportion to confidence in that view. A model that ends up leaning heavily on the stable prior isn't failing; it's doing its job honestly given what the data actually supports.

---

## 10. Robustness Checks and a Methodology Lesson

Two further tests, run after the "final" result above, worth including for what they reveal about process as much as results:

- **A 35-stock universe** (up from 26, adding names across sectors not previously represented, e.g. utilities) showed the strategy tracking much closer to the benchmarks than any earlier run — a genuinely interesting result, but confounded, since NVDA was also removed from the universe at the same time. No clean way to attribute the improvement to either change alone.
- **A shortened return lookback** (252 days instead of 756) was tested, but the covariance window and rebalance frequency were inadvertently changed at the same time — again, multiple simultaneous variables, no clean attribution.

**The honest process note:** both of these tests, along with an earlier rerun, ended up checking against the 2016–2024 test window more than once — which, per Section 6's own stated rule, means that window should no longer be treated as a clean, untouched test. Any future claim resting on it carries less weight than it would have the first time. This is included deliberately, not glossed over — a real methodology lesson from doing the work, not just a footnote.

---

## 11. What's Next

- **A properly isolated version of the lookback-window test** — change *only* the return-side lookback (momentum research typically uses 3–12 month windows, not 3 years), holding the covariance window and rebalance frequency fixed, tested against data not yet spent.
- **Real walk-forward validation** — multiple rolling train/test windows across the full history, rather than one static split, for a more robust read than a single boundary date can give.
- **Fundamentals-based factors (value, quality)** — deliberately not pursued here, because `yfinance`'s free financial-statement data only covers roughly the last 3–4 years/quarters, nowhere near enough for point-in-time history across a 20+ year backtest. Would require a different data source.
- Once real time passes, 2024–2026 becomes a genuinely fresh, never-touched window again — the natural point to properly revisit any of the above.