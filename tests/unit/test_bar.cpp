#include <gtest/gtest.h>
#include "trade/model/bar.h"

using namespace trade;

// =============================================================================
// Helper: create a basic valid bar
// =============================================================================
static Bar make_bar(const std::string& symbol, double open, double high,
                    double low, double close, Volume volume,
                    double prev_close = 0.0) {
    Bar b;
    b.symbol = symbol;
    b.open = open;
    b.high = high;
    b.low = low;
    b.close = close;
    b.volume = volume;
    b.prev_close = prev_close;
    return b;
}

// =============================================================================
// change_pct tests
// =============================================================================

TEST(BarTest, ChangePctPositive) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.5, 11.0, 1000, 10.0);
    EXPECT_NEAR(b.change_pct(), 0.10, 1e-9);
}

TEST(BarTest, ChangePctNegative) {
    Bar b = make_bar("600000.SH", 10.0, 10.5, 9.0, 9.0, 1000, 10.0);
    EXPECT_NEAR(b.change_pct(), -0.10, 1e-9);
}

TEST(BarTest, ChangePctZero) {
    Bar b = make_bar("600000.SH", 10.0, 10.5, 9.5, 10.0, 1000, 10.0);
    EXPECT_DOUBLE_EQ(b.change_pct(), 0.0);
}

TEST(BarTest, ChangePctNoPrevClose) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.5, 11.0, 1000, 0.0);
    EXPECT_DOUBLE_EQ(b.change_pct(), 0.0);
}

// =============================================================================
// amplitude tests
// =============================================================================

TEST(BarTest, AmplitudeBasic) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.0, 10.5, 1000, 10.0);
    // amplitude = (11.0 - 9.0) / 10.0 = 0.20
    EXPECT_NEAR(b.amplitude(), 0.20, 1e-9);
}

TEST(BarTest, AmplitudeZeroPrevClose) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.0, 10.5, 1000, 0.0);
    EXPECT_DOUBLE_EQ(b.amplitude(), 0.0);
}

TEST(BarTest, AmplitudeFlat) {
    Bar b = make_bar("600000.SH", 10.0, 10.0, 10.0, 10.0, 1000, 10.0);
    EXPECT_DOUBLE_EQ(b.amplitude(), 0.0);
}

// =============================================================================
// open_gap tests
// =============================================================================

TEST(BarTest, OpenGapUp) {
    Bar b = make_bar("600000.SH", 10.5, 11.0, 10.0, 10.8, 1000, 10.0);
    // open_gap = (10.5 - 10.0) / 10.0 = 0.05
    EXPECT_NEAR(b.open_gap(), 0.05, 1e-9);
}

TEST(BarTest, OpenGapDown) {
    Bar b = make_bar("600000.SH", 9.5, 10.0, 9.0, 9.8, 1000, 10.0);
    // open_gap = (9.5 - 10.0) / 10.0 = -0.05
    EXPECT_NEAR(b.open_gap(), -0.05, 1e-9);
}

TEST(BarTest, OpenGapZeroPrevClose) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.0, 10.5, 1000, 0.0);
    EXPECT_DOUBLE_EQ(b.open_gap(), 0.0);
}

// =============================================================================
// is_valid tests
// =============================================================================

TEST(BarTest, IsValidGood) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.5, 10.5, 1000);
    EXPECT_TRUE(b.is_valid());
}

TEST(BarTest, IsValidEmptySymbol) {
    Bar b = make_bar("", 10.0, 11.0, 9.5, 10.5, 1000);
    EXPECT_FALSE(b.is_valid());
}

TEST(BarTest, IsValidZeroOpen) {
    Bar b = make_bar("600000.SH", 0.0, 11.0, 9.5, 10.5, 1000);
    EXPECT_FALSE(b.is_valid());
}

TEST(BarTest, IsValidHighBelowOpen) {
    // high < open should be invalid
    Bar b = make_bar("600000.SH", 11.0, 10.0, 9.5, 10.5, 1000);
    EXPECT_FALSE(b.is_valid());
}

TEST(BarTest, IsValidLowAboveOpen) {
    // low > open should be invalid
    Bar b = make_bar("600000.SH", 9.0, 11.0, 10.0, 10.5, 1000);
    EXPECT_FALSE(b.is_valid());
}

TEST(BarTest, IsValidZeroLow) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 0.0, 10.5, 1000);
    EXPECT_FALSE(b.is_valid());
}

TEST(BarTest, IsValidZeroClose) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.5, 0.0, 1000);
    EXPECT_FALSE(b.is_valid());
}

TEST(BarTest, IsValidZeroVolume) {
    // volume >= 0 is allowed (suspended stocks can have 0 volume)
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.5, 10.5, 0);
    EXPECT_TRUE(b.is_valid());
}

TEST(BarTest, IsValidNegativeVolume) {
    Bar b = make_bar("600000.SH", 10.0, 11.0, 9.5, 10.5, -1);
    EXPECT_FALSE(b.is_valid());
}

// =============================================================================
// ExtBar::compute_limits tests
// =============================================================================

TEST(ExtBarTest, ComputeLimitsMain) {
    ExtBar eb;
    eb.symbol = "600000.SH";
    eb.prev_close = 10.0;
    eb.close = 10.50;
    eb.board = Board::kMain;

    eb.compute_limits();

    // limit_up = 10.0 * 1.10 = 11.00
    // limit_down = 10.0 * 0.90 = 9.00
    EXPECT_NEAR(eb.limit_up, 11.00, 0.01);
    EXPECT_NEAR(eb.limit_down, 9.00, 0.01);
    EXPECT_FALSE(eb.hit_limit_up);
    EXPECT_FALSE(eb.hit_limit_down);
}

TEST(ExtBarTest, ComputeLimitsST) {
    ExtBar eb;
    eb.symbol = "000001.SZ";
    eb.prev_close = 5.0;
    eb.close = 5.25;
    eb.board = Board::kST;

    eb.compute_limits();

    // limit_up = 5.0 * 1.05 = 5.25
    // limit_down = 5.0 * 0.95 = 4.75
    EXPECT_NEAR(eb.limit_up, 5.25, 0.01);
    EXPECT_NEAR(eb.limit_down, 4.75, 0.01);
    // close >= limit_up - 0.005 => 5.25 >= 5.245 => true
    EXPECT_TRUE(eb.hit_limit_up);
    EXPECT_FALSE(eb.hit_limit_down);
}

TEST(ExtBarTest, ComputeLimitsChiNext) {
    ExtBar eb;
    eb.symbol = "300001.SZ";
    eb.prev_close = 20.0;
    eb.close = 16.01;
    eb.board = Board::kChiNext;

    eb.compute_limits();

    // limit_up = 20.0 * 1.20 = 24.00
    // limit_down = 20.0 * 0.80 = 16.00
    EXPECT_NEAR(eb.limit_up, 24.00, 0.01);
    EXPECT_NEAR(eb.limit_down, 16.00, 0.01);
    EXPECT_FALSE(eb.hit_limit_up);
    // close <= limit_down + 0.005 => 16.01 <= 16.005 => false
    EXPECT_FALSE(eb.hit_limit_down);
}

TEST(ExtBarTest, ComputeLimitsHitLimitDown) {
    ExtBar eb;
    eb.symbol = "600000.SH";
    eb.prev_close = 10.0;
    eb.close = 9.00;
    eb.board = Board::kMain;

    eb.compute_limits();

    // limit_down = 9.00
    // close <= limit_down + 0.005 => 9.00 <= 9.005 => true
    EXPECT_TRUE(eb.hit_limit_down);
    EXPECT_FALSE(eb.hit_limit_up);
}

TEST(ExtBarTest, ComputeLimitsBSE) {
    ExtBar eb;
    eb.symbol = "830001.BJ";
    eb.prev_close = 10.0;
    eb.close = 12.99;
    eb.board = Board::kBSE;

    eb.compute_limits();

    // limit_up = 10.0 * 1.30 = 13.00
    // limit_down = 10.0 * 0.70 = 7.00
    EXPECT_NEAR(eb.limit_up, 13.00, 0.01);
    EXPECT_NEAR(eb.limit_down, 7.00, 0.01);
    // close >= limit_up - 0.005 => 12.99 >= 12.995 => false
    EXPECT_FALSE(eb.hit_limit_up);
    EXPECT_FALSE(eb.hit_limit_down);
}

TEST(ExtBarTest, ComputeLimitsRounding) {
    // Test that limit prices are rounded to 2 decimal places
    ExtBar eb;
    eb.symbol = "600000.SH";
    eb.prev_close = 13.37;
    eb.close = 14.00;
    eb.board = Board::kMain;

    eb.compute_limits();

    // limit_up = 13.37 * 1.10 = 14.707 -> rounded to 14.71
    // limit_down = 13.37 * 0.90 = 12.033 -> rounded to 12.03
    EXPECT_NEAR(eb.limit_up, 14.71, 0.01);
    EXPECT_NEAR(eb.limit_down, 12.03, 0.01);
}

// =============================================================================
// BarSeries tests
// =============================================================================

TEST(BarSeriesTest, BasicOperations) {
    BarSeries bs;
    bs.symbol = "600000.SH";
    EXPECT_TRUE(bs.empty());
    EXPECT_EQ(bs.size(), 0u);

    bs.bars.push_back(make_bar("600000.SH", 10.0, 11.0, 9.5, 10.5, 1000));
    bs.bars.push_back(make_bar("600000.SH", 10.5, 12.0, 10.0, 11.0, 2000));

    EXPECT_FALSE(bs.empty());
    EXPECT_EQ(bs.size(), 2u);
    EXPECT_DOUBLE_EQ(bs[0].open, 10.0);
    EXPECT_DOUBLE_EQ(bs[1].close, 11.0);
}
