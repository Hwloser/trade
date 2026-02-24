#!/usr/bin/env python3
"""CLI: run daily sentiment pipeline.

Usage:
    python -m scripts.run_sentiment --date 2026-02-24
    python -m scripts.run_sentiment --date 2026-02-24 --dry-run
    python -m scripts.run_sentiment --start 2026-02-01 --end 2026-02-24
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_date(s: str) -> date:
    from datetime import date as dt
    return dt.fromisoformat(s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sentiment pipeline")
    parser.add_argument("--date", help="Single date (YYYY-MM-DD)")
    parser.add_argument("--start", help="Start date for range")
    parser.add_argument("--end", help="End date for range")
    parser.add_argument("--data-root", default="data", help="Data directory")
    parser.add_argument("--source", default="rss", help="Data source (rss)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch articles but skip Claude API")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (default: ANTHROPIC_API_KEY env)")
    args = parser.parse_args()

    # Determine date range
    if args.date:
        dates = [parse_date(args.date)]
    elif args.start:
        start = parse_date(args.start)
        end = parse_date(args.end) if args.end else date.today()
        dates = []
        d = start
        while d <= end:
            dates.append(d)
            d += timedelta(days=1)
    else:
        dates = [date.today()]

    # Ensure we can import from project root
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root / "python"))

    from trade_py.sentiment_pipeline import run

    total_stats = []
    for target_date in dates:
        print(f"\n=== {target_date} ===")
        stats = run(
            target_date=target_date,
            data_root=args.data_root,
            sources=[args.source],
            api_key=args.api_key,
            dry_run=args.dry_run,
        )
        total_stats.append(stats)
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
