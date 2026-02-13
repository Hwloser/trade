#pragma once

#include "trade/common/types.h"

#include <string>
#include <vector>

namespace trade {

// ============================================================================
// SentimentResult: output of a sentiment model for a single text input
// ============================================================================
struct SentimentResult {
    double positive = 0.0;     // probability of positive sentiment
    double neutral = 0.0;      // probability of neutral sentiment
    double negative = 0.0;     // probability of negative sentiment
    double confidence = 0.0;   // max(positive, neutral, negative)

    // Dominant sentiment direction (argmax of the three probabilities).
    SentimentDirection direction() const {
        if (positive >= neutral && positive >= negative)
            return SentimentDirection::kPositive;
        if (negative >= positive && negative >= neutral)
            return SentimentDirection::kNegative;
        return SentimentDirection::kNeutral;
    }

    // Convenience: net sentiment score in [-1, 1].
    // (positive - negative), ignoring neutral.
    double net_score() const { return positive - negative; }
};

// ============================================================================
// ISentimentModel: abstract sentiment inference interface
// ============================================================================
// Both the rule-based dictionary engine and the ONNX neural model implement
// this interface, allowing the pipeline to swap implementations at runtime
// based on configuration or model availability.
//
class ISentimentModel {
public:
    virtual ~ISentimentModel() = default;

    // Model name, e.g. "rule_dict", "finbert_onnx"
    virtual std::string name() const = 0;

    // Predict sentiment for a single cleaned text.
    virtual SentimentResult predict(const std::string& text) = 0;

    // Batch prediction for efficiency (default: loops over predict()).
    virtual std::vector<SentimentResult> predict_batch(
        const std::vector<std::string>& texts) = 0;

    // Whether the model is loaded and ready for inference.
    virtual bool is_ready() const = 0;
};

} // namespace trade
