"""RSS feed fetcher for Chinese financial news.

Fetches news from configured RSS feeds (财联社, 新浪财经, etc.).
Uses feedparser for parsing. No API key required for public RSS feeds.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """A single news article from an RSS feed."""
    title: str
    text: str                   # cleaned body text
    url: str
    source: str                 # feed name, e.g. "CLS", "Sina"
    published_at: datetime      # UTC
    content_hash: str = ""      # dedup key

    def __post_init__(self):
        if not self.content_hash:
            raw = f"{self.title}\n{self.text}"
            self.content_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def date(self) -> date:
        return self.published_at.date()


# Default RSS feeds (public, no auth required)
DEFAULT_FEEDS = [
    {"name": "CLS", "url": "https://rsshub.app/cls/telegraph"},
    {"name": "Sina", "url": "https://rsshub.app/sina/finance"},
    {"name": "EastMoney", "url": "https://rsshub.app/eastmoney/news"},
]


def _clean_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    import re
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_feed(feed_url: str, source_name: str,
               since: Optional[date] = None,
               timeout: int = 15) -> list[NewsArticle]:
    """Fetch articles from a single RSS/Atom feed.

    Args:
        feed_url: RSS feed URL
        source_name: Human-readable source name
        since: Only include articles published on or after this date
        timeout: HTTP timeout in seconds

    Returns:
        List of NewsArticle objects, most recent first
    """
    try:
        import feedparser
    except ImportError:
        raise ImportError("Install feedparser: pip install feedparser>=6.0")

    logger.debug("Fetching %s from %s", source_name, feed_url)
    try:
        feed = feedparser.parse(feed_url, request_headers={"User-Agent": "trade-bot/1.0"})
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", feed_url, e)
        return []

    articles = []
    for entry in feed.entries:
        # Parse published time
        pub_time = None
        for attr in ("published_parsed", "updated_parsed", "created_parsed"):
            t = getattr(entry, attr, None)
            if t:
                pub_time = datetime(*t[:6], tzinfo=timezone.utc)
                break
        if pub_time is None:
            pub_time = datetime.now(timezone.utc)

        if since and pub_time.date() < since:
            continue

        title = _clean_html(getattr(entry, "title", "") or "")
        # Prefer summary over content for brevity
        text_raw = (getattr(entry, "summary", "") or
                    getattr(entry, "description", "") or "")
        # Also append full content if available
        if hasattr(entry, "content"):
            for c in entry.content:
                text_raw += " " + (c.get("value", "") or "")

        text = _clean_html(text_raw)
        url = getattr(entry, "link", "") or ""

        articles.append(NewsArticle(
            title=title,
            text=text,
            url=url,
            source=source_name,
            published_at=pub_time,
        ))

    logger.info("Fetched %d articles from %s", len(articles), source_name)
    return articles


def fetch_all(feeds: Optional[list[dict]] = None,
              since: Optional[date] = None,
              deduplicate: bool = True) -> list[NewsArticle]:
    """Fetch from all configured feeds.

    Args:
        feeds: List of dicts with 'name' and 'url'. Defaults to DEFAULT_FEEDS.
        since: Only include articles on/after this date.
        deduplicate: Remove duplicate articles by content hash.

    Returns:
        All articles, sorted newest first.
    """
    if feeds is None:
        feeds = DEFAULT_FEEDS

    all_articles: list[NewsArticle] = []
    for feed_cfg in feeds:
        articles = fetch_feed(
            feed_url=feed_cfg["url"],
            source_name=feed_cfg["name"],
            since=since,
        )
        all_articles.extend(articles)

    # Sort newest first
    all_articles.sort(key=lambda a: a.published_at, reverse=True)

    if deduplicate:
        seen: set[str] = set()
        unique = []
        for a in all_articles:
            if a.content_hash not in seen:
                seen.add(a.content_hash)
                unique.append(a)
        all_articles = unique

    return all_articles
