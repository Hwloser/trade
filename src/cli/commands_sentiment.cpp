#include "trade/cli/commands.h"

#include "trade/cli/shared.h"
#include "trade/common/time_utils.h"
#include "trade/sentiment/rss_source.h"
#include "trade/sentiment/rule_sentiment.h"
#include "trade/sentiment/sentiment_factor.h"
#include "trade/sentiment/symbol_linker.h"
#include "trade/sentiment/text_cleaner.h"
#include "trade/storage/metadata_store.h"
#include "trade/storage/parquet_writer.h"
#include "trade/storage/storage_path.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

namespace trade::cli {
int cmd_sentiment(const CliArgs& args, const trade::Config& config) {
    std::string src = args.source.empty() ? config.sentiment.default_source : args.source;
    spdlog::info("Sentiment (source: {})", src);

    if (src != "rss") {
        spdlog::error("CLI supports 'rss' source; '{}' requires API credentials", src);
        return 1;
    }

    trade::StoragePath paths(config.data.data_root);
    trade::MetadataStore metadata(paths.metadata_db());

    auto now = std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    auto sentiment_default_start =
        now - std::chrono::days{std::max(1, config.sentiment.default_history_days)};
    auto [start, end] = resolve_dates(args, trade::format_date(sentiment_default_start));
    if (args.start_date.empty()) {
        auto wm = metadata.last_watermark_date(src, "sentiment_text", src);
        if (wm) {
            auto next = *wm - std::chrono::days{std::max(0, config.sentiment.incremental_lookback_days)};
            if (next > start) start = next;
        }
    }

    auto run_id = src + "_sent_" +
        std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    metadata.begin_ingestion_run(run_id, src, "sentiment_pipeline", src, "incremental");

    try {
        trade::RssSource rss;
        if (config.sentiment.rss_feeds.empty()) {
            spdlog::error("sentiment.rss_feeds is empty in config");
            return 1;
        }
        for (const auto& feed : config.sentiment.rss_feeds) {
            rss.add_feed(feed.url, feed.name);
        }

        auto events = rss.fetch_range(start, end);
        if (events.empty()) {
            trade::MetadataStore::QualityCheckRecord qc;
            qc.run_id = run_id;
            qc.dataset_id = "raw.sentiment.bronze";
            qc.check_name = "event_count";
            qc.status = "warn";
            qc.severity = "warning";
            qc.metric_value = 0.0;
            qc.threshold_value = 1.0;
            qc.message = "No new sentiment events";
            qc.event_date = end;
            metadata.record_quality_check(qc);
            metadata.finish_ingestion_run(run_id, true, 0, 0);
            std::cout << "No new sentiment events in range "
                      << trade::format_date(start) << " to " << trade::format_date(end)
                      << std::endl;
            return 0;
        }

        trade::TextCleaner cleaner;
        for (auto& ev : events) {
            ev.clean_text = cleaner.clean(ev.raw_text);
        }

        // Bronze write by date partition
        std::map<trade::Date, std::vector<trade::TextEvent>> bronze_by_date;
        for (const auto& ev : events) {
            auto d = std::chrono::floor<std::chrono::days>(ev.timestamp);
            bronze_by_date[d].push_back(ev);
        }
        for (auto& [d, day_events] : bronze_by_date) {
            auto ymd = std::chrono::year_month_day{d};
            int y = static_cast<int>(ymd.year());
            int m = static_cast<unsigned>(ymd.month());
            auto path = paths.sentiment_bronze(y, m, src, d);
            trade::ParquetStore::write_text_events(path, day_events,
                trade::ParquetStore::MergeMode::kMergeByKey, d);
        }

        trade::RuleSentiment model;
        if (!config.sentiment.dict_path.empty()) {
            model.load_dict(config.sentiment.dict_path);
        }

        trade::SymbolLinker linker;
        auto instruments = metadata.get_all_instruments();
        linker.build_index(instruments);

        struct Agg {
            double pos = 0.0;
            double neu = 0.0;
            double neg = 0.0;
            int count = 0;
        };
        std::unordered_map<std::string, Agg> agg;

        int pos_cnt = 0, neg_cnt = 0, neu_cnt = 0;
        for (const auto& ev : events) {
            const auto txt = ev.clean_text.empty() ? ev.raw_text : ev.clean_text;
            auto score = model.predict(txt);

            if (score.positive > score.neutral && score.positive > score.negative) ++pos_cnt;
            else if (score.negative > score.neutral && score.negative > score.positive) ++neg_cnt;
            else ++neu_cnt;

            auto day = std::chrono::floor<std::chrono::days>(ev.timestamp);
            std::vector<trade::Symbol> syms;
            if (!args.symbol.empty()) {
                syms.push_back(args.symbol);
            } else {
                syms = linker.link_symbols(ev.title + " " + txt);
            }
            if (syms.empty()) continue;

            for (const auto& sym : syms) {
                auto key = sym + "|" + trade::format_date(day) + "|" + src;
                auto& a = agg[key];
                a.pos += score.positive;
                a.neu += score.neutral;
                a.neg += score.negative;
                a.count += 1;
            }
        }

        std::vector<trade::NlpResult> nlp_results;
        nlp_results.reserve(agg.size());
        for (const auto& [key, a] : agg) {
            auto p1 = key.find('|');
            auto p2 = key.find('|', p1 + 1);
            if (p1 == std::string::npos || p2 == std::string::npos || a.count <= 0) continue;

            trade::NlpResult r;
            r.symbol = key.substr(0, p1);
            r.date = trade::parse_date(key.substr(p1 + 1, p2 - p1 - 1));
            r.source = key.substr(p2 + 1);
            r.sentiment.positive = a.pos / a.count;
            r.sentiment.neutral = a.neu / a.count;
            r.sentiment.negative = a.neg / a.count;
            r.article_count = a.count;
            nlp_results.push_back(std::move(r));
        }

        // Silver write by date partition
        std::map<trade::Date, std::vector<trade::NlpResult>> silver_by_date;
        for (const auto& r : nlp_results) {
            silver_by_date[r.date].push_back(r);
        }
        for (auto& [d, day_results] : silver_by_date) {
            auto ymd = std::chrono::year_month_day{d};
            int y = static_cast<int>(ymd.year());
            int m = static_cast<unsigned>(ymd.month());
            auto path = paths.sentiment_silver(y, m, d);
            trade::ParquetStore::write_nlp_results(path, day_results,
                trade::ParquetStore::MergeMode::kMergeByKey, d);
        }

        // Gold factors
        std::unordered_map<trade::Symbol, trade::BarSeries> bar_map;
        for (const auto& r : nlp_results) {
            if (!bar_map.count(r.symbol)) {
                trade::BarSeries series;
                series.symbol = r.symbol;
                series.bars = load_bars(r.symbol, config);
                bar_map[r.symbol] = std::move(series);
            }
        }

        trade::SentimentFactorCalculator calc;
        auto factors = calc.compute(nlp_results, bar_map);

        std::map<trade::Date, std::vector<trade::SentimentFactors>> gold_by_date;
        for (const auto& f : factors) {
            gold_by_date[f.date].push_back(f);
        }
        for (auto& [d, day_factors] : gold_by_date) {
            auto ymd = std::chrono::year_month_day{d};
            int y = static_cast<int>(ymd.year());
            int m = static_cast<unsigned>(ymd.month());
            auto path = paths.sentiment_gold(y, m, d);
            trade::ParquetStore::write_sentiment_factors(path, day_factors,
                trade::ParquetStore::MergeMode::kMergeByKey, d);
        }

        trade::Date max_date = start;
        for (const auto& [d, _] : bronze_by_date) {
            if (d > max_date) max_date = d;
        }
        metadata.upsert_watermark(src, "sentiment_text", src, max_date);
        metadata.upsert_stream_checkpoint(src,
                                          "sentiment_text",
                                          src,
                                          R"({"mode":"incremental"})",
                                          max_date);

        trade::MetadataStore::QualityCheckRecord event_qc;
        event_qc.run_id = run_id;
        event_qc.dataset_id = "raw.sentiment.bronze";
        event_qc.check_name = "event_count";
        event_qc.status = events.empty() ? "warn" : "pass";
        event_qc.severity = events.empty() ? "warning" : "info";
        event_qc.metric_value = static_cast<double>(events.size());
        event_qc.threshold_value = 1.0;
        event_qc.message = "Fetched sentiment events";
        event_qc.event_date = max_date;
        metadata.record_quality_check(event_qc);

        trade::MetadataStore::QualityCheckRecord nlp_cov_qc;
        nlp_cov_qc.run_id = run_id;
        nlp_cov_qc.dataset_id = "curated.sentiment.silver";
        nlp_cov_qc.check_name = "nlp_symbol_coverage";
        nlp_cov_qc.metric_value = events.empty() ? 0.0
            : static_cast<double>(nlp_results.size()) / static_cast<double>(events.size());
        nlp_cov_qc.threshold_value = 0.05;
        nlp_cov_qc.status = nlp_cov_qc.metric_value >= nlp_cov_qc.threshold_value ? "pass" : "warn";
        nlp_cov_qc.severity = nlp_cov_qc.status == "pass" ? "info" : "warning";
        nlp_cov_qc.message = "NLP aggregated rows / events";
        nlp_cov_qc.event_date = max_date;
        metadata.record_quality_check(nlp_cov_qc);

        trade::MetadataStore::QualityCheckRecord factor_cov_qc;
        factor_cov_qc.run_id = run_id;
        factor_cov_qc.dataset_id = "curated.sentiment.gold";
        factor_cov_qc.check_name = "factor_row_coverage";
        factor_cov_qc.metric_value = nlp_results.empty() ? 0.0
            : static_cast<double>(factors.size()) / static_cast<double>(nlp_results.size());
        factor_cov_qc.threshold_value = 0.1;
        factor_cov_qc.status = factor_cov_qc.metric_value >= factor_cov_qc.threshold_value ? "pass" : "warn";
        factor_cov_qc.severity = factor_cov_qc.status == "pass" ? "info" : "warning";
        factor_cov_qc.message = "Factor rows / nlp rows";
        factor_cov_qc.event_date = max_date;
        metadata.record_quality_check(factor_cov_qc);

        int total = pos_cnt + neg_cnt + neu_cnt;
        std::cout << "=== Sentiment Incremental ===\n"
                  << "Range: " << trade::format_date(start) << " to " << trade::format_date(end) << "\n"
                  << "Events: " << events.size() << "\n"
                  << "NLP rows: " << nlp_results.size() << "\n"
                  << "Factor rows: " << factors.size() << "\n"
                  << "Summary: " << pos_cnt << " pos / " << neu_cnt << " neu / " << neg_cnt << " neg\n";
        if (total > 0) {
            std::cout << "Net sentiment: " << std::showpos << std::fixed << std::setprecision(3)
                      << (double(pos_cnt - neg_cnt) / total) << std::noshowpos << "\n";
        }

        metadata.finish_ingestion_run(run_id, true,
                                      static_cast<int64_t>(events.size()),
                                      static_cast<int64_t>(nlp_results.size() + factors.size()));
        return 0;
    } catch (const std::exception& e) {
        metadata.finish_ingestion_run(run_id, false, 0, 0, e.what());
        spdlog::error("Sentiment pipeline failed: {}", e.what());
        return 1;
    }
}


} // namespace trade::cli
