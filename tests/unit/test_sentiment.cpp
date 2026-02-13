#include <gtest/gtest.h>
#include <cmath>
#include <string>
#include <vector>

#include "trade/sentiment/sentiment_model.h"
#include "trade/sentiment/text_cleaner.h"
#include "trade/sentiment/rule_sentiment.h"
#include "trade/sentiment/sentiment_factor.h"

using namespace trade;

// =============================================================================
// SentimentResult tests
// =============================================================================

TEST(SentimentResultTest, DirectionPositive) {
    SentimentResult result;
    result.positive = 0.8;
    result.neutral = 0.1;
    result.negative = 0.1;
    EXPECT_EQ(result.direction(), SentimentDirection::kPositive);
}

TEST(SentimentResultTest, DirectionNegative) {
    SentimentResult result;
    result.positive = 0.1;
    result.neutral = 0.2;
    result.negative = 0.7;
    EXPECT_EQ(result.direction(), SentimentDirection::kNegative);
}

TEST(SentimentResultTest, DirectionNeutral) {
    SentimentResult result;
    result.positive = 0.2;
    result.neutral = 0.6;
    result.negative = 0.2;
    EXPECT_EQ(result.direction(), SentimentDirection::kNeutral);
}

TEST(SentimentResultTest, NetScore) {
    SentimentResult result;
    result.positive = 0.7;
    result.neutral = 0.1;
    result.negative = 0.2;
    EXPECT_DOUBLE_EQ(result.net_score(), 0.5);
}

TEST(SentimentResultTest, NetScoreZero) {
    SentimentResult result;
    result.positive = 0.3;
    result.neutral = 0.4;
    result.negative = 0.3;
    EXPECT_DOUBLE_EQ(result.net_score(), 0.0);
}

TEST(SentimentResultTest, Defaults) {
    SentimentResult result;
    EXPECT_DOUBLE_EQ(result.positive, 0.0);
    EXPECT_DOUBLE_EQ(result.neutral, 0.0);
    EXPECT_DOUBLE_EQ(result.negative, 0.0);
    EXPECT_DOUBLE_EQ(result.confidence, 0.0);
}

// =============================================================================
// TextCleaner tests
// =============================================================================

TEST(TextCleanerTest, RemoveHtmlTags) {
    std::string html = "<p>Hello <b>world</b></p>";
    std::string result = TextCleaner::remove_html_tags(html);
    EXPECT_EQ(result, "Hello world");
}

TEST(TextCleanerTest, RemoveHtmlTagsNested) {
    std::string html = "<div><p>Test <a href='url'>link</a></p></div>";
    std::string result = TextCleaner::remove_html_tags(html);
    EXPECT_EQ(result, "Test link");
}

TEST(TextCleanerTest, RemoveHtmlTagsNoTags) {
    std::string text = "Plain text";
    std::string result = TextCleaner::remove_html_tags(text);
    EXPECT_EQ(result, "Plain text");
}

TEST(TextCleanerTest, FullwidthToHalfwidth) {
    // Test fullwidth comma -> halfwidth comma
    std::string fullwidth = "\xEF\xBC\x8C";  // U+FF0C (fullwidth comma)
    std::string result = TextCleaner::fullwidth_to_halfwidth(fullwidth);
    EXPECT_EQ(result, ",");
}

TEST(TextCleanerTest, FullwidthDigits) {
    // U+FF10 = fullwidth '0'
    std::string fullwidth_zero = "\xEF\xBC\x90";
    std::string result = TextCleaner::fullwidth_to_halfwidth(fullwidth_zero);
    EXPECT_EQ(result, "0");
}

TEST(TextCleanerTest, ContentHashDeterministic) {
    std::string text = "test content";
    auto h1 = TextCleaner::content_hash(text);
    auto h2 = TextCleaner::content_hash(text);
    EXPECT_EQ(h1, h2);
    EXPECT_FALSE(h1.empty());
}

TEST(TextCleanerTest, ContentHashDifferent) {
    auto h1 = TextCleaner::content_hash("hello");
    auto h2 = TextCleaner::content_hash("world");
    EXPECT_NE(h1, h2);
}

TEST(TextCleanerTest, DefaultStopwords) {
    auto& sw = TextCleaner::default_stopwords();
    EXPECT_FALSE(sw.empty());
    // Should contain common Chinese stopwords
    EXPECT_TRUE(sw.count("\xe7\x9a\x84") > 0);  // 的
}

TEST(TextCleanerTest, CleanPipeline) {
    std::string input = "<p>Hello  World</p>";
    std::string cleaned = TextCleaner::clean(input);
    // Should remove HTML tags and normalize whitespace
    EXPECT_TRUE(cleaned.find('<') == std::string::npos);
    EXPECT_TRUE(cleaned.find('>') == std::string::npos);
}

// =============================================================================
// RuleSentiment tests
// =============================================================================

TEST(RuleSentimentTest, IsReadyDefault) {
    RuleSentiment model;
    // Without dictionaries, model may or may not be ready depending on impl
    // Just verify it doesn't crash
    [[maybe_unused]] bool ready = model.is_ready();
}

TEST(RuleSentimentTest, Name) {
    RuleSentiment model;
    EXPECT_EQ(model.name(), "rule_dict");
}

TEST(RuleSentimentTest, PredictEmpty) {
    RuleSentiment model;
    auto result = model.predict("");
    // Empty text should be neutral
    EXPECT_GE(result.neutral, result.positive);
    EXPECT_GE(result.neutral, result.negative);
}

TEST(RuleSentimentTest, AddPositiveWord) {
    RuleSentiment model;
    model.add_positive_word("good", 1.0);
    EXPECT_EQ(model.positive_dict_size(), 1u);
    auto result = model.predict("good");
    EXPECT_GT(result.positive, result.negative);
}

TEST(RuleSentimentTest, AddNegativeWord) {
    RuleSentiment model;
    model.add_negative_word("bad", -1.0);
    EXPECT_EQ(model.negative_dict_size(), 1u);
    auto result = model.predict("bad");
    EXPECT_GT(result.negative, result.positive);
}

TEST(RuleSentimentTest, PredictBatch) {
    RuleSentiment model;
    model.add_positive_word("up", 1.0);
    model.add_negative_word("down", -1.0);
    std::vector<std::string> texts = {"up", "down", ""};
    auto results = model.predict_batch(texts);
    EXPECT_EQ(results.size(), 3u);
    EXPECT_GT(results[0].positive, results[0].negative);
    EXPECT_GT(results[1].negative, results[1].positive);
}

TEST(RuleSentimentTest, NeutralThreshold) {
    RuleSentiment model;
    model.set_neutral_threshold(0.2);
    EXPECT_DOUBLE_EQ(model.neutral_threshold(), 0.2);
}

TEST(RuleSentimentTest, NegationWindow) {
    RuleSentiment model;
    model.set_negation_window(6);
    EXPECT_EQ(model.negation_window(), 6);
}

// =============================================================================
// SentimentFactor tests
// =============================================================================

TEST(SentimentFactorTest, ConfigDefaults) {
    SentimentFactorCalculator::Config cfg;
    EXPECT_EQ(cfg.ema_halflife, 5);
    EXPECT_GT(cfg.volatility_window, 0);
}

// =============================================================================
// NegShock detection simulation
// =============================================================================

TEST(SentimentTest, NegShockDetection) {
    std::vector<double> neg_series = {0.2, 0.3, 0.25, 0.3, 0.8};
    double alpha = 2.0 / (5 + 1);
    double ema = neg_series[0];
    for (size_t i = 1; i < neg_series.size() - 1; ++i) {
        ema = alpha * neg_series[i] + (1 - alpha) * ema;
    }
    double neg_today = neg_series.back();
    double neg_shock = neg_today - ema;
    EXPECT_GT(neg_shock, 0.0);
    EXPECT_GT(neg_shock, 0.4);
}

TEST(SentimentTest, NetSentimentCalc) {
    double pos = 10.0, neg = 3.0, neu = 7.0;
    double net = (pos - neg) / (pos + neg + neu);
    EXPECT_NEAR(net, 0.35, 0.01);
}
