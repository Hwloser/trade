"""Claude API client for financial news sentiment analysis.

Uses Claude Haiku (claude-haiku-4-5-20251001) for cost-effective,
high-quality structured sentiment extraction from Chinese financial news.

Cost: ~$0.001 per article at 200-500 tokens.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional
import hashlib
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是专业的A股市场金融情感分析助手。
分析新闻文本，提取结构化的情感和事件信息，只返回JSON，不要其他内容。"""

USER_TEMPLATE = """分析以下A股市场新闻：

标题：{title}
内容：{text}

返回JSON（只返回JSON对象，不要markdown）：
{{
  "sentiment_score": <float -1.0到1.0，-1.0极负面，1.0极正面>,
  "sentiment_label": <"positive"|"neutral"|"negative">,
  "event_type": <"policy"|"earnings"|"expansion"|"acquisition"|"regulation"|"macro"|"personnel"|"product"|"other">,
  "event_magnitude": <float 0.0到1.0，0.0微小影响，1.0重大影响>,
  "affected_sectors": <受影响行业列表，如["半导体","新能源"]>,
  "key_entities": <关键实体，公司/人物/政策名称列表>,
  "summary": <30字以内中文摘要>,
  "confidence": <float 0.0到1.0，分析置信度>
}}"""


@dataclass
class SentimentResult:
    sentiment_score: float = 0.0      # -1.0 to 1.0
    sentiment_label: str = "neutral"  # positive/neutral/negative
    event_type: str = "other"
    event_magnitude: float = 0.0      # 0.0 to 1.0
    affected_sectors: list = None     # will be set in __post_init__
    key_entities: list = None
    summary: str = ""
    confidence: float = 0.5
    # metadata
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self):
        if self.affected_sectors is None:
            self.affected_sectors = []
        if self.key_entities is None:
            self.key_entities = []

    def to_dict(self) -> dict:
        return asdict(self)


class ClaudeClient:
    """Thin wrapper around Anthropic API for batch news analysis."""

    MODEL = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 512
    RATE_LIMIT_DELAY = 0.5   # seconds between calls
    MAX_RETRIES = 3

    def __init__(self, api_key: Optional[str] = None):
        """Initialize client.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        """
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic>=0.40.0")

        self._last_call = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def analyze(self, title: str, text: str,
                max_text_chars: int = 800) -> SentimentResult:
        """Analyze a single news article.

        Args:
            title: Article headline
            text: Article body (will be truncated to max_text_chars)
            max_text_chars: Max body characters to send (cost control)

        Returns:
            SentimentResult with structured sentiment data
        """
        truncated = text[:max_text_chars] if len(text) > max_text_chars else text
        prompt = USER_TEMPLATE.format(title=title, text=truncated)

        # Rate limiting
        elapsed = time.time() - self._last_call
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                self._last_call = time.time()
                self._total_input_tokens += response.usage.input_tokens
                self._total_output_tokens += response.usage.output_tokens

                raw = response.content[0].text.strip()
                data = json.loads(raw)

                return SentimentResult(
                    sentiment_score=float(data.get("sentiment_score", 0.0)),
                    sentiment_label=str(data.get("sentiment_label", "neutral")),
                    event_type=str(data.get("event_type", "other")),
                    event_magnitude=float(data.get("event_magnitude", 0.0)),
                    affected_sectors=list(data.get("affected_sectors", [])),
                    key_entities=list(data.get("key_entities", [])),
                    summary=str(data.get("summary", "")),
                    confidence=float(data.get("confidence", 0.5)),
                    model=self.MODEL,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error (attempt %d): %s", attempt + 1, e)
                if attempt == self.MAX_RETRIES - 1:
                    return SentimentResult(summary="[parse error]")
            except Exception as e:
                logger.warning("API error (attempt %d): %s", attempt + 1, e)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    return SentimentResult(summary=f"[error: {e}]")

        return SentimentResult()

    def analyze_batch(self, articles: list[dict],
                      progress: bool = True) -> list[SentimentResult]:
        """Analyze a batch of articles.

        Args:
            articles: List of dicts with 'title' and 'text' keys
            progress: Show progress

        Returns:
            List of SentimentResult (same order as input)
        """
        results = []
        n = len(articles)
        for i, article in enumerate(articles):
            if progress:
                print(f"\r  [{i+1}/{n}] Analyzing... cost≈${self.estimated_cost:.3f}", end="")
            result = self.analyze(
                title=article.get("title", ""),
                text=article.get("text", ""),
            )
            results.append(result)
        if progress:
            print(f"\r  Done {n} articles. Cost≈${self.estimated_cost:.4f}  ")
        return results

    @property
    def estimated_cost(self) -> float:
        """Estimated API cost in USD (Haiku pricing)."""
        # Claude Haiku: $0.80/M input, $4.00/M output
        return (self._total_input_tokens * 0.80 + self._total_output_tokens * 4.00) / 1_000_000

    @property
    def token_usage(self) -> dict:
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "estimated_cost_usd": self.estimated_cost,
        }


def content_hash(title: str, text: str) -> str:
    """SHA-256 dedup key for an article."""
    return hashlib.sha256(f"{title}\n{text}".encode("utf-8")).hexdigest()[:16]
