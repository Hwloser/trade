#include "trade/sentiment/text_cleaner.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <sstream>

namespace trade {

std::string TextCleaner::clean(const std::string& raw_text) {
    auto text = remove_html_tags(raw_text);
    text = fullwidth_to_halfwidth(text);
    text = normalize_whitespace(text);
    text = remove_punctuation(text);
    text = remove_stopwords(text);
    return text;
}

// ---------------------------------------------------------------------------
// Helper: decode a single UTF-8 codepoint starting at data[pos].
// Returns the codepoint and advances pos past the consumed bytes.
// ---------------------------------------------------------------------------
static uint32_t decode_utf8(const std::string& s, size_t& pos) {
    uint32_t cp = 0;
    unsigned char c = static_cast<unsigned char>(s[pos]);
    if (c < 0x80) {
        cp = c;
        pos += 1;
    } else if ((c & 0xE0) == 0xC0) {
        cp = c & 0x1F;
        if (pos + 1 < s.size()) cp = (cp << 6) | (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
        pos += 2;
    } else if ((c & 0xF0) == 0xE0) {
        cp = c & 0x0F;
        if (pos + 1 < s.size()) cp = (cp << 6) | (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
        if (pos + 2 < s.size()) cp = (cp << 6) | (static_cast<unsigned char>(s[pos + 2]) & 0x3F);
        pos += 3;
    } else if ((c & 0xF8) == 0xF0) {
        cp = c & 0x07;
        if (pos + 1 < s.size()) cp = (cp << 6) | (static_cast<unsigned char>(s[pos + 1]) & 0x3F);
        if (pos + 2 < s.size()) cp = (cp << 6) | (static_cast<unsigned char>(s[pos + 2]) & 0x3F);
        if (pos + 3 < s.size()) cp = (cp << 6) | (static_cast<unsigned char>(s[pos + 3]) & 0x3F);
        pos += 4;
    } else {
        // Invalid leading byte -- skip one byte
        pos += 1;
        cp = 0xFFFD; // replacement character
    }
    return cp;
}

// ---------------------------------------------------------------------------
// Helper: encode a Unicode codepoint to UTF-8 and append to out.
// ---------------------------------------------------------------------------
static void encode_utf8(uint32_t cp, std::string& out) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ---------------------------------------------------------------------------
// strip_html: state-machine HTML tag removal + entity decoding
// ---------------------------------------------------------------------------
std::string TextCleaner::remove_html_tags(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    bool in_tag = false;
    size_t i = 0;
    while (i < text.size()) {
        char c = text[i];
        if (in_tag) {
            if (c == '>') {
                in_tag = false;
            }
            ++i;
        } else if (c == '<') {
            in_tag = true;
            ++i;
        } else if (c == '&') {
            // Try to decode common HTML entities
            if (text.compare(i, 5, "&amp;") == 0) {
                result.push_back('&');
                i += 5;
            } else if (text.compare(i, 4, "&lt;") == 0) {
                result.push_back('<');
                i += 4;
            } else if (text.compare(i, 4, "&gt;") == 0) {
                result.push_back('>');
                i += 4;
            } else if (text.compare(i, 6, "&quot;") == 0) {
                result.push_back('"');
                i += 6;
            } else if (text.compare(i, 6, "&nbsp;") == 0) {
                result.push_back(' ');
                i += 6;
            } else if (text.compare(i, 6, "&apos;") == 0) {
                result.push_back('\'');
                i += 6;
            } else {
                result.push_back(c);
                ++i;
            }
        } else {
            result.push_back(c);
            ++i;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// remove_punctuation: remove ASCII punctuation and common CJK punctuation
// ---------------------------------------------------------------------------
static bool is_cjk_punctuation(uint32_t cp) {
    // Chinese / CJK punctuation codepoints
    switch (cp) {
        case 0x3002: // 。
        case 0xFF0C: // ，
        case 0xFF01: // ！
        case 0xFF1F: // ？
        case 0x3001: // 、
        case 0xFF1B: // ；
        case 0xFF1A: // ：
        case 0xFF08: // （
        case 0xFF09: // ）
        case 0x3010: // 【
        case 0x3011: // 】
        case 0x300C: // 「
        case 0x300D: // 」
        case 0x300E: // 『
        case 0x300F: // 』
        case 0x201C: // "
        case 0x201D: // "
        case 0x2018: // '
        case 0x2019: // '
        case 0x2014: // —
        case 0x2026: // …
        case 0x00B7: // ·
        case 0x3008: // 〈
        case 0x3009: // 〉
        case 0x300A: // 《
        case 0x300B: // 》
            return true;
        default:
            return false;
    }
}

std::string TextCleaner::remove_punctuation(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = pos;
        uint32_t cp = decode_utf8(text, pos);

        // Skip ASCII punctuation
        if (cp < 0x80 && std::ispunct(static_cast<unsigned char>(cp))) {
            continue;
        }
        // Skip CJK punctuation
        if (is_cjk_punctuation(cp)) {
            continue;
        }
        // Keep the character (copy original bytes to preserve valid UTF-8)
        result.append(text, start, pos - start);
    }
    return result;
}

// ---------------------------------------------------------------------------
// normalize_whitespace: collapse runs of whitespace, trim edges
// ---------------------------------------------------------------------------
std::string TextCleaner::normalize_whitespace(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    bool in_space = false;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!in_space && !result.empty()) {
                result.push_back(' ');
                in_space = true;
            }
        } else {
            result.push_back(c);
            in_space = false;
        }
    }
    // Trim trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

// ---------------------------------------------------------------------------
// fullwidth_to_halfwidth: convert U+FF01..U+FF5E to ASCII (subtract 0xFEE0)
// Also convert fullwidth space U+3000 to ASCII space.
// ---------------------------------------------------------------------------
std::string TextCleaner::fullwidth_to_halfwidth(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    size_t pos = 0;
    while (pos < text.size()) {
        size_t start = pos;
        uint32_t cp = decode_utf8(text, pos);

        if (cp >= 0xFF01 && cp <= 0xFF5E) {
            // Fullwidth ASCII variant -> halfwidth
            encode_utf8(cp - 0xFEE0, result);
        } else if (cp == 0x3000) {
            // Fullwidth space -> ASCII space
            result.push_back(' ');
        } else {
            // Keep original bytes
            result.append(text, start, pos - start);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// remove_stopwords (default set)
// ---------------------------------------------------------------------------
std::string TextCleaner::remove_stopwords(const std::string& text) {
    return remove_stopwords(text, default_stopwords());
}

// ---------------------------------------------------------------------------
// remove_stopwords (custom set): split by whitespace, remove stopwords
// ---------------------------------------------------------------------------
std::string TextCleaner::remove_stopwords(const std::string& text,
                                           const std::unordered_set<std::string>& stopwords) {
    if (stopwords.empty()) return text;

    std::istringstream iss(text);
    std::string word;
    std::string result;
    bool first = true;

    while (iss >> word) {
        if (stopwords.find(word) == stopwords.end()) {
            if (!first) result.push_back(' ');
            result.append(word);
            first = false;
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// load_stopwords: read file line by line
// ---------------------------------------------------------------------------
std::unordered_set<std::string> TextCleaner::load_stopwords(const std::string& path) {
    std::unordered_set<std::string> stopwords;
    std::ifstream file(path);
    if (!file.is_open()) return stopwords;

    std::string line;
    while (std::getline(file, line)) {
        // Trim trailing \r (in case of Windows line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (!line.empty()) {
            stopwords.insert(line);
        }
    }
    return stopwords;
}

// ---------------------------------------------------------------------------
// default_stopwords: common Chinese financial stopwords
// ---------------------------------------------------------------------------
const std::unordered_set<std::string>& TextCleaner::default_stopwords() {
    static const std::unordered_set<std::string> stopwords = {
        "\xe7\x9a\x84",     // 的
        "\xe4\xba\x86",     // 了
        "\xe5\x9c\xa8",     // 在
        "\xe6\x98\xaf",     // 是
        "\xe5\x92\x8c",     // 和
        "\xe4\xb8\x8e",     // 与
        "\xe6\x88\x96",     // 或
        "\xe5\xaf\xb9",     // 对
        "\xe4\xbb\x8e",     // 从
        "\xe5\x88\xb0",     // 到
        "\xe4\xbb\xa5",     // 以
        "\xe5\xb0\x86",     // 将
        "\xe4\xb9\x9f",     // 也
        "\xe5\x8f\x88",     // 又
        "\xe5\xb0\xb1",     // 就
        "\xe9\x83\xbd",     // 都
        "\xe5\xb7\xb2",     // 已
        "\xe4\xb8\x8d",     // 不
        "\xe8\xa2\xab",     // 被
        "\xe6\x8a\x8a",     // 把
        "\xe6\x89\x80",     // 所
        "\xe7\xad\x89",     // 等
        "\xe8\xbf\x99",     // 这
        "\xe9\x82\xa3",     // 那
        "\xe6\x9c\x89",     // 有
        "\xe4\xb8\xba",     // 为
        "\xe4\xb8\xad",     // 中
        "\xe4\xb8\x8a",     // 上
        "\xe4\xb8\x8b",     // 下
    };
    return stopwords;
}

// ---------------------------------------------------------------------------
// normalize_financial_abbrev: replace common abbreviations
// ---------------------------------------------------------------------------
std::string TextCleaner::normalize_financial_abbrev(const std::string& text) {
    // Pairs of (pattern, replacement)
    static const std::vector<std::pair<std::string, std::string>> replacements = {
        {"\xe6\xb2\xaa\xe6\xb7\xb1\xe4\xb8\x89\xe7\x99\xbe",   // 沪深三百
         "\xe6\xb2\xaa\xe6\xb7\xb1\x33\x30\x30"},                // 沪深300
        {"\xe4\xb8\x8a\xe8\xaf\x81\xe4\xba\x94\xe5\x8d\x81",   // 上证五十
         "\xe4\xb8\x8a\xe8\xaf\x81\x35\x30"},                    // 上证50
        {"\xe6\xb2\xaa\xe6\x8c\x87",                              // 沪指
         "\xe4\xb8\x8a\xe8\xaf\x81\xe6\x8c\x87\xe6\x95\xb0"},   // 上证指数
        {"\xe6\xb7\xb1\xe6\x88\x90\xe6\x8c\x87",                 // 深成指
         "\xe6\xb7\xb1\xe8\xaf\x81\xe6\x88\x90\xe6\x8c\x87"},   // 深证成指
        {"\xe5\x88\x9b\xe4\xb8\x9a\xe6\x9d\xbf\xe6\x8c\x87",   // 创业板指
         "\xe5\x88\x9b\xe4\xb8\x9a\xe6\x9d\xbf\xe6\x8c\x87\xe6\x95\xb0"}, // 创业板指数
        {"\xe4\xb8\xad\xe5\xb0\x8f\xe6\x9d\xbf",                 // 中小板
         "\xe4\xb8\xad\xe5\xb0\x8f\xe6\x9d\xbf\xe6\x8c\x87\xe6\x95\xb0"}, // 中小板指数
        {"\xe7\xa7\x91\xe5\x88\x9b\xe6\x9d\xbf",                 // 科创板
         "\xe7\xa7\x91\xe5\x88\x9b\xe6\x9d\xbf\xe6\x8c\x87\xe6\x95\xb0"}, // 科创板指数
    };

    std::string result = text;
    for (const auto& [pattern, replacement] : replacements) {
        size_t pos = 0;
        while ((pos = result.find(pattern, pos)) != std::string::npos) {
            result.replace(pos, pattern.size(), replacement);
            pos += replacement.size();
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// content_hash: use std::hash<std::string> and return hex string
// ---------------------------------------------------------------------------
std::string TextCleaner::content_hash(const std::string& text) {
    std::size_t h = std::hash<std::string>{}(text);

    // Convert to hex string
    static const char hex_chars[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(sizeof(h) * 2);
    for (int i = static_cast<int>(sizeof(h)) - 1; i >= 0; --i) {
        unsigned char byte = static_cast<unsigned char>((h >> (i * 8)) & 0xFF);
        hex.push_back(hex_chars[byte >> 4]);
        hex.push_back(hex_chars[byte & 0x0F]);
    }
    return hex;
}

} // namespace trade
