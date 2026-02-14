#pragma once

#include "trade/model/bar.h"
#include <vector>

namespace trade {

// Field mapping from provider-specific names to our canonical names
struct FieldMapping {
    std::string date_field = "日期";
    std::string open_field = "开盘";
    std::string high_field = "最高";
    std::string low_field = "最低";
    std::string close_field = "收盘";
    std::string volume_field = "成交量";
    std::string amount_field = "成交额";
    std::string turnover_field = "换手率";

    static FieldMapping akshare_daily();
};

class BarNormalizer {
public:
    // Normalize raw bars: sort by date, fill prev_close, compute VWAP
    static std::vector<Bar> normalize(std::vector<Bar> bars);

    // Fill missing prev_close from previous bar
    static void fill_prev_close(std::vector<Bar>& bars);

    // Compute VWAP = amount / volume
    static void compute_vwap(std::vector<Bar>& bars);

    // Sort by date ascending
    static void sort_by_date(std::vector<Bar>& bars);

    // Compute price limits from prev_close and board, detect limit hits
    static void compute_limits(std::vector<Bar>& bars, Board board);
};

} // namespace trade
