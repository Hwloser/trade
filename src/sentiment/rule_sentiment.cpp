#include "trade/sentiment/rule_sentiment.h"

#include <cmath>
#include <fstream>
#include <sstream>

namespace trade {

RuleSentiment::RuleSentiment() {
    // Default negation words for Chinese text
    negation_words_ = {
        "\xe6\xb2\xa1\xe6\x9c\x89",  // 没有
        "\xe4\xb8\x8d",                // 不
        "\xe6\x9c\xaa",                // 未
        "\xe6\xb2\xa1",                // 没
        "\xe9\x9d\x9e",                // 非
        "\xe6\x97\xa0",                // 无
        "\xe5\x88\xab",                // 别
        "\xe8\x8e\xab",                // 莫
    };
}

RuleSentiment::~RuleSentiment() = default;

SentimentResult RuleSentiment::predict(const std::string& text) {
    double raw = compute_raw_score(text);
    return score_to_result(raw);
}

std::vector<SentimentResult> RuleSentiment::predict_batch(
    const std::vector<std::string>& texts) {
    std::vector<SentimentResult> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        results.push_back(predict(text));
    }
    return results;
}

bool RuleSentiment::is_ready() const {
    return !positive_words_.empty() || !negative_words_.empty();
}

size_t RuleSentiment::load_dict(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return 0;

    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        // Trim trailing \r
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;
        auto tab_pos = line.find('\t');
        std::string word;
        double weight = 0.0;
        if (tab_pos != std::string::npos) {
            word = line.substr(0, tab_pos);
            weight = std::stod(line.substr(tab_pos + 1));
        } else {
            word = line;
            weight = 1.0;  // default positive
        }
        if (weight > 0) {
            positive_words_[word] = weight;
        } else {
            negative_words_[word] = weight;
        }
        ++count;
    }
    return count;
}

void RuleSentiment::load_positive_dict(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        // Check for optional tab-separated weight
        auto tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string word = line.substr(0, tab_pos);
            double weight = std::stod(line.substr(tab_pos + 1));
            if (!word.empty()) {
                positive_words_[word] = (weight > 0) ? weight : 1.0;
            }
        } else {
            positive_words_[line] = 1.0;
        }
    }
}

void RuleSentiment::load_negative_dict(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;

        // Check for optional tab-separated weight
        auto tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string word = line.substr(0, tab_pos);
            double weight = std::stod(line.substr(tab_pos + 1));
            if (!word.empty()) {
                negative_words_[word] = (weight < 0) ? weight : -1.0;
            }
        } else {
            negative_words_[line] = -1.0;
        }
    }
}

void RuleSentiment::add_positive_word(const std::string& word, double weight) {
    positive_words_[word] = weight;
}

void RuleSentiment::add_negative_word(const std::string& word, double weight) {
    negative_words_[word] = weight;
}

void RuleSentiment::add_negation_word(const std::string& word) {
    negation_words_.insert(word);
}

// ---------------------------------------------------------------------------
// Helper: check if a negation word appears in the window before position pos
// in the text. We look for any negation word whose UTF-8 bytes end within
// negation_window_ bytes before pos.
// ---------------------------------------------------------------------------
static bool has_negation_in_window(const std::string& text,
                                   size_t pos,
                                   int window_bytes,
                                   const std::unordered_set<std::string>& negation_words) {
    // Look backward up to window_bytes before the match
    size_t start = (pos > static_cast<size_t>(window_bytes)) ? pos - window_bytes : 0;
    std::string window = text.substr(start, pos - start);

    for (const auto& neg : negation_words) {
        if (window.find(neg) != std::string::npos) {
            return true;
        }
    }
    return false;
}

double RuleSentiment::compute_raw_score(const std::string& text) const {
    double score_sum = 0.0;
    int sentiment_word_count = 0;

    // The negation window in bytes: Chinese characters are 3 bytes in UTF-8,
    // so negation_window_ characters * 3 bytes gives a reasonable byte window.
    int window_bytes = negation_window_ * 3;

    // Scan for positive words
    for (const auto& [word, weight] : positive_words_) {
        size_t pos = 0;
        while ((pos = text.find(word, pos)) != std::string::npos) {
            double w = weight;
            // Check for negation word before this match
            if (has_negation_in_window(text, pos, window_bytes, negation_words_)) {
                w = -w;  // Flip polarity
            }
            score_sum += w;
            sentiment_word_count++;
            pos += word.size();
        }
    }

    // Scan for negative words
    for (const auto& [word, weight] : negative_words_) {
        size_t pos = 0;
        while ((pos = text.find(word, pos)) != std::string::npos) {
            double w = weight;
            // Check for negation word before this match
            if (has_negation_in_window(text, pos, window_bytes, negation_words_)) {
                w = -w;  // Flip polarity
            }
            score_sum += w;
            sentiment_word_count++;
            pos += word.size();
        }
    }

    if (sentiment_word_count == 0) return 0.0;
    return score_sum / static_cast<double>(sentiment_word_count);
}

SentimentResult RuleSentiment::score_to_result(double raw_score) const {
    // Map raw score to probabilities via sigmoid
    SentimentResult result;
    double abs_score = std::abs(raw_score);

    if (abs_score < neutral_threshold_) {
        // Neutral zone
        result.neutral = 0.6;
        result.positive = 0.2;
        result.negative = 0.2;
    } else if (raw_score > 0) {
        // Positive
        double sigmoid = 1.0 / (1.0 + std::exp(-raw_score));
        result.positive = sigmoid;
        result.neutral = (1.0 - sigmoid) * 0.7;
        result.negative = (1.0 - sigmoid) * 0.3;
    } else {
        // Negative
        double sigmoid = 1.0 / (1.0 + std::exp(raw_score));
        result.negative = sigmoid;
        result.neutral = (1.0 - sigmoid) * 0.7;
        result.positive = (1.0 - sigmoid) * 0.3;
    }

    result.confidence = std::max({result.positive, result.neutral, result.negative});
    return result;
}

} // namespace trade
