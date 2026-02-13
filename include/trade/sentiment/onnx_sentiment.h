#pragma once

// ============================================================================
// OnnxSentiment: ONNX Runtime sentiment inference (FinBERT / RoBERTa)
// ============================================================================
// Wraps the ONNX Runtime C++ API to run an exported transformer model
// (e.g. FinBERT or Chinese-RoBERTa fine-tuned on financial sentiment)
// for three-class sentiment classification.
//
// Build guard:
//   This header and its implementation are only compiled when ONNX Runtime
//   is available (HAVE_ONNXRUNTIME is defined).  When absent, the pipeline
//   falls back to RuleSentiment.
//
// Hot-reload:
//   Call load_model() with a new path at any time to swap the underlying
//   ONNX model without restarting the process.
//

#ifdef HAVE_ONNXRUNTIME

#include "trade/sentiment/sentiment_model.h"

#include <memory>
#include <string>
#include <vector>

namespace trade {

class OnnxSentiment : public ISentimentModel {
public:
    struct Config {
        std::string onnx_model_path;          // path to .onnx model file
        std::string tokenizer_path;           // path to tokenizer.json / vocab.txt
        int max_sequence_length = 128;        // max tokens per input
        int batch_size = 32;                  // inference batch size
        int num_threads = 2;                  // ONNX Runtime intra-op threads
        bool use_gpu = false;                 // CUDA execution provider
    };

    OnnxSentiment();
    explicit OnnxSentiment(const Config& cfg);
    ~OnnxSentiment() override;

    // Non-copyable, movable
    OnnxSentiment(const OnnxSentiment&) = delete;
    OnnxSentiment& operator=(const OnnxSentiment&) = delete;
    OnnxSentiment(OnnxSentiment&&) noexcept;
    OnnxSentiment& operator=(OnnxSentiment&&) noexcept;

    // -- ISentimentModel -----------------------------------------------------
    std::string name() const override { return "onnx_sentiment"; }
    SentimentResult predict(const std::string& text) override;
    std::vector<SentimentResult> predict_batch(
        const std::vector<std::string>& texts) override;
    bool is_ready() const override;

    // -- Model lifecycle -----------------------------------------------------

    // Load (or hot-reload) an ONNX model + tokenizer.
    // Can be called multiple times to swap models without restart.
    // Returns true on success.
    bool load_model(const std::string& onnx_path,
                    const std::string& tokenizer_path);

    // Load using the Config struct.
    bool load_model(const Config& cfg);

    // Unload the current model and free resources.
    void unload();

    // -- Configuration -------------------------------------------------------
    const Config& config() const { return config_; }
    void set_max_sequence_length(int len);
    void set_batch_size(int bs);
    void set_num_threads(int threads);

    // -- Diagnostics ---------------------------------------------------------

    // Model metadata (populated after load_model).
    std::string model_description() const;
    std::vector<std::string> input_names() const;
    std::vector<std::string> output_names() const;

private:
    // Tokenize a single text into input tensors (input_ids, attention_mask).
    struct TokenizedInput {
        std::vector<int64_t> input_ids;
        std::vector<int64_t> attention_mask;
    };
    TokenizedInput tokenize(const std::string& text) const;

    // Tokenize a batch.
    std::vector<TokenizedInput> tokenize_batch(
        const std::vector<std::string>& texts) const;

    // Run inference on tokenized inputs and return raw logits.
    std::vector<std::array<float, 3>> run_inference(
        const std::vector<TokenizedInput>& inputs) const;

    // Softmax logits -> SentimentResult.
    static SentimentResult logits_to_result(const std::array<float, 3>& logits);

    Config config_;

    // PIMPL to hide ONNX Runtime headers from consumers.
    struct OnnxSession;
    std::unique_ptr<OnnxSession> session_;

    struct Tokenizer;
    std::unique_ptr<Tokenizer> tokenizer_;
};

} // namespace trade

#endif // HAVE_ONNXRUNTIME
