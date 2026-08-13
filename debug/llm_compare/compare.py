"""Head-to-head: Claude Haiku vs local Qwen on real Bronze articles.

Samples N articles deterministically, runs both clients, dumps raw results to
JSON so agreement metrics can be recomputed without re-paying for the API.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

from trade_py.intelligence.clients.anthropic import AnthropicClient
from trade_py.intelligence.clients.ollama import OllamaClient

BRONZE = "data/sentiment/bronze/rss/**/*.parquet"
OUT = Path(__file__).parent / "results.json"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def sample(n: int) -> list[dict]:
    con = duckdb.connect()
    rows = con.execute(
        f"""
        select title, text from read_parquet('{BRONZE}')
        where title is not null and length(coalesce(text, '')) > 50
        order by content_hash
        limit {n}
        """
    ).fetchall()
    return [{"title": t, "text": x} for t, x in rows]


def run(client, articles: list[dict], tag: str) -> list[dict]:
    out = []
    for i, a in enumerate(articles, 1):
        t0 = time.time()
        r = client.analyze(a["title"], a["text"])
        d = r.to_dict()
        d["_elapsed"] = round(time.time() - t0, 2)
        out.append(d)
        print(f"  [{tag} {i}/{len(articles)}] {d['_elapsed']}s "
              f"{d['sentiment_label']:8} {d['event_type']}", file=sys.stderr)
    return out


def main() -> None:
    articles = sample(N)
    print(f"Sampled {len(articles)} articles", file=sys.stderr)

    qwen = run(OllamaClient(model="qwen2.5:14b-instruct"), articles, "qwen")
    claude = run(AnthropicClient(model="claude-haiku-4-5"), articles, "claude")

    OUT.write_text(json.dumps(
        {"articles": articles, "qwen": qwen, "claude": claude},
        ensure_ascii=False, indent=2,
    ))
    print(f"\nWrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
