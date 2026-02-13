#include "trade/sentiment/symbol_linker.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace trade {

// PIMPL definition
struct SymbolLinker::Impl {
    // Stores patterns for matching: full names, codes, aliases
    std::unordered_map<std::string, Symbol> name_to_symbol;
    std::unordered_map<std::string, Symbol> code_to_symbol;
    std::unordered_map<std::string, std::vector<std::pair<Symbol, double>>> alias_to_symbols;
    size_t pattern_count = 0;
};

SymbolLinker::SymbolLinker() : impl_(std::make_unique<Impl>()) {}

SymbolLinker::~SymbolLinker() = default;

void SymbolLinker::build_index(const std::vector<Instrument>& instruments) {
    impl_->name_to_symbol.clear();
    impl_->code_to_symbol.clear();
    impl_->pattern_count = 0;

    for (const auto& inst : instruments) {
        // Index by full name
        if (!inst.name.empty()) {
            impl_->name_to_symbol[inst.name] = inst.symbol;
            impl_->pattern_count++;
        }
        // Index by symbol code (e.g. "600000" extracted from "600000.SH")
        auto dot_pos = inst.symbol.find('.');
        if (dot_pos != std::string::npos) {
            std::string code = inst.symbol.substr(0, dot_pos);
            impl_->code_to_symbol[code] = inst.symbol;
            impl_->pattern_count++;
        }
        // Also index by full symbol
        impl_->code_to_symbol[inst.symbol] = inst.symbol;
        impl_->pattern_count++;
    }
}

void SymbolLinker::add_alias(const Symbol& symbol, const std::string& alias) {
    impl_->alias_to_symbols[alias].push_back({symbol, 0.8});
    impl_->pattern_count++;
}

void SymbolLinker::load_aliases(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        // Trim trailing \r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        // Tab-separated: alias\tsymbol
        auto tab_pos = line.find('\t');
        if (tab_pos == std::string::npos) continue;

        std::string alias = line.substr(0, tab_pos);
        std::string symbol = line.substr(tab_pos + 1);

        if (!alias.empty() && !symbol.empty()) {
            impl_->alias_to_symbols[alias].push_back({symbol, 0.8});
            impl_->pattern_count++;
        }
    }
}

std::vector<SymbolMatch> SymbolLinker::link(const std::string& text) const {
    std::vector<SymbolMatch> matches;

    // Simple substring matching against full names
    for (const auto& [name, symbol] : impl_->name_to_symbol) {
        if (text.find(name) != std::string::npos) {
            SymbolMatch m;
            m.symbol = symbol;
            m.matched_text = name;
            m.match_score = 1.0;
            m.type = SymbolMatch::MatchType::kFullName;
            matches.push_back(m);
        }
    }

    // Match codes if enabled
    if (match_bare_code_) {
        for (const auto& [code, symbol] : impl_->code_to_symbol) {
            if (code.size() == 6 && text.find(code) != std::string::npos) {
                SymbolMatch m;
                m.symbol = symbol;
                m.matched_text = code;
                m.match_score = 0.9;
                m.type = SymbolMatch::MatchType::kStockCode;
                matches.push_back(m);
            }
        }
    }

    // Match aliases
    for (const auto& [alias, symbol_list] : impl_->alias_to_symbols) {
        if (text.find(alias) != std::string::npos) {
            for (const auto& [sym, score] : symbol_list) {
                if (score >= min_score_) {
                    SymbolMatch m;
                    m.symbol = sym;
                    m.matched_text = alias;
                    m.match_score = score;
                    m.type = SymbolMatch::MatchType::kAlias;
                    matches.push_back(m);
                }
            }
        }
    }

    // Sort by score descending
    std::sort(matches.begin(), matches.end(),
              [](const SymbolMatch& a, const SymbolMatch& b) {
                  return a.match_score > b.match_score;
              });

    return matches;
}

std::vector<SymbolMatch> SymbolLinker::link(const std::string& text,
                                             const std::vector<Symbol>& universe) const {
    auto all_matches = link(text);
    // Filter to only symbols in the universe
    std::unordered_set<Symbol> universe_set(universe.begin(), universe.end());
    std::vector<SymbolMatch> filtered;
    for (const auto& m : all_matches) {
        if (universe_set.count(m.symbol)) {
            filtered.push_back(m);
        }
    }
    return filtered;
}

std::vector<Symbol> SymbolLinker::link_symbols(const std::string& text) const {
    auto matches = link(text);
    std::unordered_set<Symbol> seen;
    std::vector<Symbol> result;
    for (const auto& m : matches) {
        if (seen.insert(m.symbol).second) {
            result.push_back(m.symbol);
        }
    }
    return result;
}

size_t SymbolLinker::index_size() const {
    return impl_->pattern_count;
}

} // namespace trade
