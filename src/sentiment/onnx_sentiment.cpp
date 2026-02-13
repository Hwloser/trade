#include "trade/sentiment/onnx_sentiment.h"

#ifdef HAVE_ONNXRUNTIME

#include <cmath>
#include <iostream>
#include <numeric>

namespace trade {

// PIMPL forward declarations
struct OnnxSentiment::OnnxSession {
    // Holds Ort::Session, Ort::Env, Ort::SessionOptions when ONNX Runtime
    // headers are available. For now, placeholder fields.
    bool loaded = false;
    std::string description;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

struct OnnxSentiment::Tokenizer {
    // Holds tokenizer vocabulary and encoding logic.
    // When a real tokenizer library is integrated, this will contain
    // the WordPiece / BPE vocabulary and encoding state.
    bool loaded = false;
    int max_length = 128;
};

OnnxSentiment::OnnxSentiment()
    : session_(std::make_unique<OnnxSession>())
    , tokenizer_(std::make_unique<Tokenizer>()) {}

OnnxSentiment::OnnxSentiment(const Config& cfg)
    : config_(cfg)
    , session_(std::make_unique<OnnxSession>())
    , tokenizer_(std::make_unique<Tokenizer>()) {
    if (!cfg.onnx_model_path.empty()) {
        load_model(cfg);
    }
}

OnnxSentiment::~OnnxSentiment() = default;

OnnxSentiment::OnnxSentiment(OnnxSentiment&&) noexcept = default;
OnnxSentiment& OnnxSentiment::operator=(OnnxSentiment&&) noexcept = default;

SentimentResult OnnxSentiment::predict(const std::string& text) {
    auto batch = predict_batch({text});
    if (batch.empty()) {
        // Return neutral default when model is not ready
        SentimentResult neutral;
        neutral.positive = 0.0;
        neutral.neutral = 1.0;
        neutral.negative = 0.0;
        neutral.confidence = 1.0;
        return neutral;
    }
    return batch[0];
}

std::vector<SentimentResult> OnnxSentiment::predict_batch(
    const std::vector<std::string>& texts) {
    if (!is_ready() || texts.empty()) {
        // Return neutral defaults for all inputs when model is unavailable
        std::vector<SentimentResult> defaults(texts.size());
        for (auto& r : defaults) {
            r.positive = 0.0;
            r.neutral = 1.0;
            r.negative = 0.0;
            r.confidence = 1.0;
        }
        return defaults;
    }

    // Tokenize -> run inference -> convert logits to results
    auto tokenized = tokenize_batch(texts);
    auto logits = run_inference(tokenized);

    std::vector<SentimentResult> results;
    results.reserve(logits.size());
    for (const auto& l : logits) {
        results.push_back(logits_to_result(l));
    }
    return results;
}

bool OnnxSentiment::is_ready() const {
    return session_ && session_->loaded &&
           tokenizer_ && tokenizer_->loaded;
}

bool OnnxSentiment::load_model(const std::string& onnx_path,
                                const std::string& tokenizer_path) {
    // Actual ONNX Runtime model loading:
    //   1. Create Ort::Env with ORT_LOGGING_LEVEL_WARNING
    //   2. Configure Ort::SessionOptions (num_threads, GPU provider if use_gpu)
    //   3. Create Ort::Session from onnx_path
    //   4. Extract input/output names from the session
    //   5. Load tokenizer vocabulary from tokenizer_path
    //
    // Placeholder: log a warning that ONNX Runtime integration is compiled
    // but model loading requires the actual ORT libraries to be linked.

    config_.onnx_model_path = onnx_path;
    config_.tokenizer_path = tokenizer_path;

    std::cerr << "[OnnxSentiment] load_model: ONNX Runtime compiled in but "
              << "model loading requires ORT libraries. Model path: "
              << onnx_path << "\n";

    // In a real implementation, set session_->loaded = true on success.
    // For now, remain unloaded so is_ready() returns false and predict
    // falls back to neutral defaults.
    session_->loaded = false;
    session_->description = "OnnxSentiment (not loaded)";
    session_->input_names = {"input_ids", "attention_mask"};
    session_->output_names = {"logits"};

    return false;
}

bool OnnxSentiment::load_model(const Config& cfg) {
    config_ = cfg;
    return load_model(cfg.onnx_model_path, cfg.tokenizer_path);
}

void OnnxSentiment::unload() {
    if (session_) session_->loaded = false;
    if (tokenizer_) tokenizer_->loaded = false;
}

void OnnxSentiment::set_max_sequence_length(int len) {
    config_.max_sequence_length = len;
    if (tokenizer_) tokenizer_->max_length = len;
}

void OnnxSentiment::set_batch_size(int bs) {
    config_.batch_size = bs;
}

void OnnxSentiment::set_num_threads(int threads) {
    config_.num_threads = threads;
}

std::string OnnxSentiment::model_description() const {
    if (session_) return session_->description;
    return "";
}

std::vector<std::string> OnnxSentiment::input_names() const {
    if (session_) return session_->input_names;
    return {};
}

std::vector<std::string> OnnxSentiment::output_names() const {
    if (session_) return session_->output_names;
    return {};
}

OnnxSentiment::TokenizedInput OnnxSentiment::tokenize(
    const std::string& text) const {
    // Placeholder tokenization: produces zero-padded sequences.
    // A real implementation would use WordPiece or BPE tokenization
    // to convert the text into token IDs from the vocabulary.
    TokenizedInput input;
    int max_len = config_.max_sequence_length;
    input.input_ids.resize(max_len, 0);
    input.attention_mask.resize(max_len, 0);

    // Simple placeholder: treat each byte as a token ID (capped at vocab size).
    // Set [CLS] = 101 at position 0, [SEP] = 102 at the end.
    input.input_ids[0] = 101; // [CLS]
    input.attention_mask[0] = 1;

    int token_pos = 1;
    for (size_t i = 0; i < text.size() && token_pos < max_len - 1; ++i) {
        input.input_ids[token_pos] = static_cast<int64_t>(
            static_cast<unsigned char>(text[i]));
        input.attention_mask[token_pos] = 1;
        ++token_pos;
    }

    // [SEP] token
    if (token_pos < max_len) {
        input.input_ids[token_pos] = 102; // [SEP]
        input.attention_mask[token_pos] = 1;
    }

    return input;
}

std::vector<OnnxSentiment::TokenizedInput> OnnxSentiment::tokenize_batch(
    const std::vector<std::string>& texts) const {
    std::vector<TokenizedInput> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        results.push_back(tokenize(text));
    }
    return results;
}

std::vector<std::array<float, 3>> OnnxSentiment::run_inference(
    const std::vector<TokenizedInput>& inputs) const {
    // Actual ONNX Runtime inference:
    //   1. Create Ort::MemoryInfo for CPU allocation
    //   2. Pack input_ids and attention_mask into Ort::Value tensors
    //   3. Call session_->Run() with input/output names
    //   4. Extract output logits from the result tensors
    //
    // Placeholder: return neutral logits (0, 1, 0) -> softmax gives ~neutral.
    std::vector<std::array<float, 3>> logits(inputs.size());
    for (auto& l : logits) {
        l = {0.0f, 1.0f, 0.0f}; // Neutral bias
    }
    return logits;
}

SentimentResult OnnxSentiment::logits_to_result(
    const std::array<float, 3>& logits) {
    // Softmax
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::array<float, 3> exp_vals;
    float sum = 0.0f;
    for (int i = 0; i < 3; ++i) {
        exp_vals[i] = std::exp(logits[i] - max_val);
        sum += exp_vals[i];
    }

    SentimentResult result;
    result.positive = static_cast<double>(exp_vals[0] / sum);
    result.neutral = static_cast<double>(exp_vals[1] / sum);
    result.negative = static_cast<double>(exp_vals[2] / sum);
    result.confidence = std::max({result.positive, result.neutral, result.negative});
    return result;
}

} // namespace trade

#endif // HAVE_ONNXRUNTIME
