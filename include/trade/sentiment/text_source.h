#pragma once

#include "trade/common/types.h"

#include <string>
#include <vector>

namespace trade {

// ============================================================================
// TextEvent: a single piece of collected text from any source
// ============================================================================
struct TextEvent {
    std::string source;        // "rss", "xueqiu", "jin10"
    std::string url;
    Timestamp timestamp;
    std::string title;
    std::string raw_text;
    std::string clean_text;    // after cleaning
    std::string content_hash;  // dedup key (SHA-256 of raw_text)
};

// ============================================================================
// ITextSource: abstract data source interface for text collection
// ============================================================================
// Every text data source (RSS feeds, Xueqiu, Jin10, etc.) implements this
// interface.  The collector layer calls fetch() / fetch_range() and receives
// a vector of TextEvent objects ready for downstream NLP processing.
//
class ITextSource {
public:
    virtual ~ITextSource() = default;

    // Human-readable source name, e.g. "rss", "xueqiu", "jin10"
    virtual std::string name() const = 0;

    // Fetch all text events for a single date.
    virtual std::vector<TextEvent> fetch(Date date) = 0;

    // Fetch all text events in [start, end] inclusive.
    virtual std::vector<TextEvent> fetch_range(Date start, Date end) = 0;

    // Check whether the source is reachable / configured.
    virtual bool is_available() const = 0;
};

} // namespace trade
