#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace trade {

// ============================================================================
// TextCleaner: Chinese financial text cleaning utilities
// ============================================================================
// Provides static methods to preprocess raw text collected from RSS feeds,
// social media, and flash-news APIs before feeding into the NLP pipeline.
//
// Chinese-specific handling:
//   - Full-width to half-width character conversion (e.g. ',' -> ',')
//   - Chinese punctuation removal
//   - Common financial abbreviation normalization
//   - Chinese stopword removal
//
class TextCleaner {
public:
    // -- Full pipeline -------------------------------------------------------

    // Apply the complete cleaning pipeline:
    //   1. remove_html_tags
    //   2. fullwidth_to_halfwidth
    //   3. normalize_whitespace
    //   4. remove_punctuation
    //   5. remove_stopwords
    // Returns the cleaned text.
    static std::string clean(const std::string& raw_text);

    // -- Individual steps (can be used standalone) ----------------------------

    // Strip HTML/XML tags: "<p>foo</p>" -> "foo"
    static std::string remove_html_tags(const std::string& text);

    // Remove ASCII and Chinese punctuation characters.
    static std::string remove_punctuation(const std::string& text);

    // Collapse consecutive whitespace / newlines into a single space, trim.
    static std::string normalize_whitespace(const std::string& text);

    // Convert full-width characters to half-width equivalents.
    // e.g. U+FF0C (fullwidth comma) -> U+002C, U+FF10-FF19 -> '0'-'9'
    static std::string fullwidth_to_halfwidth(const std::string& text);

    // Remove Chinese stopwords (的, 了, 在, 是, 我, ...).
    // Uses the default built-in stopword list unless overridden.
    static std::string remove_stopwords(const std::string& text);

    // Remove stopwords using a custom set.
    static std::string remove_stopwords(const std::string& text,
                                        const std::unordered_set<std::string>& stopwords);

    // -- Stopword management -------------------------------------------------

    // Load a stopword file (one word per line, UTF-8).
    static std::unordered_set<std::string> load_stopwords(const std::string& path);

    // Get the default built-in Chinese financial stopword set.
    static const std::unordered_set<std::string>& default_stopwords();

    // -- Financial text helpers ----------------------------------------------

    // Normalize common financial abbreviations:
    //   "沪指" -> "上证指数", "深成指" -> "深证成指", etc.
    static std::string normalize_financial_abbrev(const std::string& text);

    // Compute SHA-256 content hash of text (used for dedup).
    static std::string content_hash(const std::string& text);
};

} // namespace trade
