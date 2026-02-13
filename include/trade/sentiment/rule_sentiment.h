#pragma once

#include "trade/sentiment/sentiment_model.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trade {

// ============================================================================
// RuleSentiment: dictionary-based sentiment engine (MVP / fallback)
// ============================================================================
// A zero-external-dependency sentiment classifier that uses financial
// sentiment dictionaries (positive / negative word lists) with negation
// handling.
//
// When the ONNX model is unavailable or during early development, this
// provides a baseline sentiment signal.
//
// Dictionary format (one word per line, optional tab-separated weight):
//   利好\t1.0
//   利空\t-1.0
//   下跌\t-0.8
//
// Negation handling:
//   Words like "没有", "不", "未" in a window before a sentiment word
//   flip its polarity.
//
// Scoring:
//   raw_score = sum(word_weight * negation_flip) / num_sentiment_words
//   Mapped to (positive, neutral, negative) probabilities via sigmoid.
//
class RuleSentiment : public ISentimentModel {
public:
    RuleSentiment();
    ~RuleSentiment() override;

    // -- ISentimentModel -----------------------------------------------------
    std::string name() const override { return "rule_dict"; }
    SentimentResult predict(const std::string& text) override;
    std::vector<SentimentResult> predict_batch(
        const std::vector<std::string>& texts) override;
    bool is_ready() const override;

    // -- Dictionary management -----------------------------------------------

    // Load a sentiment dictionary file.
    // Format: one word per line, optional tab-separated weight (default +1/-1).
    // Returns the number of words loaded.
    size_t load_dict(const std::string& path);

    // Load positive and negative word lists from separate files.
    void load_positive_dict(const std::string& path);
    void load_negative_dict(const std::string& path);

    // Add individual words programmatically.
    void add_positive_word(const std::string& word, double weight = 1.0);
    void add_negative_word(const std::string& word, double weight = -1.0);

    // Add negation words (default set is loaded automatically).
    void add_negation_word(const std::string& word);

    // -- Configuration -------------------------------------------------------

    // Negation window: how many characters before a sentiment word to look
    // for negation words (default: 4 characters for Chinese).
    void set_negation_window(int chars) { negation_window_ = chars; }
    int negation_window() const { return negation_window_; }

    // Neutral zone: abs(raw_score) < threshold is classified as neutral.
    void set_neutral_threshold(double t) { neutral_threshold_ = t; }
    double neutral_threshold() const { return neutral_threshold_; }

    // -- Stats ---------------------------------------------------------------
    size_t positive_dict_size() const { return positive_words_.size(); }
    size_t negative_dict_size() const { return negative_words_.size(); }

private:
    // Compute raw sentiment score for a text.
    double compute_raw_score(const std::string& text) const;

    // Convert raw score to SentimentResult probabilities via sigmoid mapping.
    SentimentResult score_to_result(double raw_score) const;

    // Positive words with weights (word -> weight, weight > 0)
    std::unordered_map<std::string, double> positive_words_;

    // Negative words with weights (word -> weight, weight < 0)
    std::unordered_map<std::string, double> negative_words_;

    // Negation words (没有, 不, 未, 没, 非, ...)
    std::unordered_set<std::string> negation_words_;

    int negation_window_ = 4;       // characters
    double neutral_threshold_ = 0.1;
};

} // namespace trade
