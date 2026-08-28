"""Run both A-share harnesses unchanged on the S&P 500 panel.

Step 1 reuses debug/eval_baseline/evaluate_factors.py (single-factor RankIC +
quintile backtest); step 2 reuses debug/lgbm_walkforward/run.py (LightGBM
walk-forward with pre-registered pass bar). Only module-level constants are
overridden: input/output paths and trading costs (US round-trip ~0.1% vs the
0.2% assumed for A-shares; one-way ~5bp vs 15bp).

Usage: uv run python debug/us_baseline/run_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def main() -> int:
    RESULTS.mkdir(exist_ok=True)

    import debug.eval_baseline.evaluate_factors as ev
    ev.KLINE_DIR = HERE / "data" / "kline"
    ev.RESULTS_DIR = RESULTS
    ev.COST_PER_TURNOVER = 0.0005
    ev.ANNUAL_DAYS = 252
    ev.main()

    import debug.lgbm_walkforward.run as wf
    wf.PANEL = RESULTS / "factor_panel.parquet"
    wf.OUT = RESULTS
    wf.COST = 0.001
    wf.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
