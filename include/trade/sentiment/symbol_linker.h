#pragma once

#include "trade/common/types.h"
#include "trade/model/instrument.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trade {

// ============================================================================
// SymbolLinker: link text events to stock symbols
// ============================================================================
// Given a piece of text (title + body), identifies which A-share stocks are
// mentioned by matching against:
//   - Full stock names (e.g. "浦发银行")
//   - Stock codes (e.g. "600000", "600000.SH")
//   - Common aliases / abbreviations (e.g. "浦发", "PFYH")
//
// Internally builds a trie and hash map from the instrument list for fast
// multi-pattern matching (Aho-Corasick style).
//
// Ambiguity handling:
//   - When a short alias matches multiple instruments, all matches are returned
//     with a match_score indicating confidence.
//   - Callers can filter by score threshold or take the best match.
//

// Result of linking a single text to symbols
struct SymbolMatch {
    Symbol symbol;             // matched symbol, e.g. "600000.SH"
    std::string matched_text;  // the substring that matched
    double match_score = 1.0;  // confidence: 1.0 = exact name/code, <1.0 = alias
    enum class MatchType : uint8_t {
        kFullName = 0,         // matched on full stock name
        kStockCode = 1,        // matched on stock code
        kAlias = 2,            // matched on alias / abbreviation
    };
    MatchType type = MatchType::kFullName;
};

class SymbolLinker {
public:
    SymbolLinker();
    ~SymbolLinker();

    // -- Index building ------------------------------------------------------

    // Build the matching index from a list of instruments.
    // Call once (or when instrument list changes).
    void build_index(const std::vector<Instrument>& instruments);

    // Add a custom alias for a symbol (e.g. "浦发" -> "600000.SH").
    void add_alias(const Symbol& symbol, const std::string& alias);

    // Load aliases from a file (tab-separated: alias\tsymbol per line).
    void load_aliases(const std::string& path);

    // -- Matching ------------------------------------------------------------

    // Link a text to zero or more stock symbols.
    // Returns all matches sorted by match_score descending.
    std::vector<SymbolMatch> link(const std::string& text) const;

    // Link text against a specific subset of instruments.
    std::vector<SymbolMatch> link(const std::string& text,
                                  const std::vector<Symbol>& universe) const;

    // Convenience: return only symbols (no scores), deduplicated.
    std::vector<Symbol> link_symbols(const std::string& text) const;

    // -- Configuration -------------------------------------------------------

    // Minimum match score to include in results (default 0.5).
    void set_min_score(double score) { min_score_ = score; }
    double min_score() const { return min_score_; }

    // Whether to match bare 6-digit codes (may produce false positives
    // on numbers that are not stock codes).
    void set_match_bare_code(bool enable) { match_bare_code_ = enable; }
    bool match_bare_code() const { return match_bare_code_; }

    // -- Stats ---------------------------------------------------------------
    size_t index_size() const;  // number of patterns in the index

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    double min_score_ = 0.5;
    bool match_bare_code_ = true;
};

} // namespace trade
