#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unordered_map.h>
#include "trade/common/types.h"
#include "trade/model/bar.h"
#include "trade/model/instrument.h"
#include "trade/model/market.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_model(nb::module_& m) {

    // -----------------------------------------------------------------------
    // Enums
    // -----------------------------------------------------------------------

    nb::enum_<trade::Market>(m, "Market")
        .value("kSH", trade::Market::kSH)
        .value("kSZ", trade::Market::kSZ)
        .value("kBJ", trade::Market::kBJ)
        .value("kHK", trade::Market::kHK)
        .value("kUS", trade::Market::kUS)
        .value("kCrypto", trade::Market::kCrypto);

    nb::enum_<trade::Board>(m, "Board")
        .value("kMain", trade::Board::kMain)
        .value("kST", trade::Board::kST)
        .value("kSTAR", trade::Board::kSTAR)
        .value("kChiNext", trade::Board::kChiNext)
        .value("kBSE", trade::Board::kBSE)
        .value("kNewIPOMainDay1", trade::Board::kNewIPOMainDay1)
        .value("kNewIPOStarDay1", trade::Board::kNewIPOStarDay1);

    nb::enum_<trade::TradingStatus>(m, "TradingStatus")
        .value("kNormal", trade::TradingStatus::kNormal)
        .value("kSuspended", trade::TradingStatus::kSuspended)
        .value("kST", trade::TradingStatus::kST)
        .value("kStarST", trade::TradingStatus::kStarST)
        .value("kDelisting", trade::TradingStatus::kDelisting);

    nb::enum_<trade::Side>(m, "Side")
        .value("kBuy", trade::Side::kBuy)
        .value("kSell", trade::Side::kSell);

    nb::enum_<trade::OrderStatus>(m, "OrderStatus")
        .value("kPending", trade::OrderStatus::kPending)
        .value("kFilled", trade::OrderStatus::kFilled)
        .value("kPartialFill", trade::OrderStatus::kPartialFill)
        .value("kCancelled", trade::OrderStatus::kCancelled)
        .value("kRejected", trade::OrderStatus::kRejected);

    nb::enum_<trade::Regime>(m, "Regime")
        .value("kBull", trade::Regime::kBull)
        .value("kBear", trade::Regime::kBear)
        .value("kShock", trade::Regime::kShock);

    nb::enum_<trade::SentimentDirection>(m, "SentimentDirection")
        .value("kPositive", trade::SentimentDirection::kPositive)
        .value("kNeutral", trade::SentimentDirection::kNeutral)
        .value("kNegative", trade::SentimentDirection::kNegative);

    nb::enum_<trade::AlertLevel>(m, "AlertLevel")
        .value("kGreen", trade::AlertLevel::kGreen)
        .value("kYellow", trade::AlertLevel::kYellow)
        .value("kOrange", trade::AlertLevel::kOrange)
        .value("kRed", trade::AlertLevel::kRed);

    nb::enum_<trade::SWIndustry>(m, "SWIndustry")
        .value("kAgriculture", trade::SWIndustry::kAgriculture)
        .value("kMining", trade::SWIndustry::kMining)
        .value("kChemical", trade::SWIndustry::kChemical)
        .value("kSteel", trade::SWIndustry::kSteel)
        .value("kNonFerrousMetal", trade::SWIndustry::kNonFerrousMetal)
        .value("kElectronics", trade::SWIndustry::kElectronics)
        .value("kAuto", trade::SWIndustry::kAuto)
        .value("kHouseholdAppliance", trade::SWIndustry::kHouseholdAppliance)
        .value("kFoodBeverage", trade::SWIndustry::kFoodBeverage)
        .value("kTextile", trade::SWIndustry::kTextile)
        .value("kLightManufacturing", trade::SWIndustry::kLightManufacturing)
        .value("kMedicine", trade::SWIndustry::kMedicine)
        .value("kUtilities", trade::SWIndustry::kUtilities)
        .value("kTransportation", trade::SWIndustry::kTransportation)
        .value("kRealEstate", trade::SWIndustry::kRealEstate)
        .value("kCommerce", trade::SWIndustry::kCommerce)
        .value("kSocialService", trade::SWIndustry::kSocialService)
        .value("kBanking", trade::SWIndustry::kBanking)
        .value("kNonBankFinancial", trade::SWIndustry::kNonBankFinancial)
        .value("kConstruction", trade::SWIndustry::kConstruction)
        .value("kBuildingMaterial", trade::SWIndustry::kBuildingMaterial)
        .value("kMechanicalEquipment", trade::SWIndustry::kMechanicalEquipment)
        .value("kDefense", trade::SWIndustry::kDefense)
        .value("kComputer", trade::SWIndustry::kComputer)
        .value("kMedia", trade::SWIndustry::kMedia)
        .value("kTelecom", trade::SWIndustry::kTelecom)
        .value("kEnvironment", trade::SWIndustry::kEnvironment)
        .value("kElectricalEquipment", trade::SWIndustry::kElectricalEquipment)
        .value("kBeauty", trade::SWIndustry::kBeauty)
        .value("kCoal", trade::SWIndustry::kCoal)
        .value("kPetroleum", trade::SWIndustry::kPetroleum)
        .value("kUnknown", trade::SWIndustry::kUnknown);

    // -----------------------------------------------------------------------
    // Bar
    // -----------------------------------------------------------------------

    nb::class_<trade::Bar>(m, "Bar")
        .def(nb::init<>())
        .def_rw("symbol", &trade::Bar::symbol)
        .def_prop_rw("date",
            [](const trade::Bar& b) { return b.date.time_since_epoch().count(); },
            [](trade::Bar& b, int d) { b.date = trade::Date(std::chrono::days(d)); })
        .def_rw("open", &trade::Bar::open)
        .def_rw("high", &trade::Bar::high)
        .def_rw("low", &trade::Bar::low)
        .def_rw("close", &trade::Bar::close)
        .def_rw("volume", &trade::Bar::volume)
        .def_rw("amount", &trade::Bar::amount)
        .def_rw("turnover_rate", &trade::Bar::turnover_rate)
        .def_rw("prev_close", &trade::Bar::prev_close)
        .def_rw("vwap", &trade::Bar::vwap)
        .def_rw("limit_up", &trade::Bar::limit_up)
        .def_rw("limit_down", &trade::Bar::limit_down)
        .def_rw("hit_limit_up", &trade::Bar::hit_limit_up)
        .def_rw("hit_limit_down", &trade::Bar::hit_limit_down)
        .def_rw("bar_status", &trade::Bar::bar_status)
        .def_rw("board", &trade::Bar::board)
        .def_rw("north_net_buy", &trade::Bar::north_net_buy)
        .def_rw("margin_balance", &trade::Bar::margin_balance)
        .def_rw("short_sell_volume", &trade::Bar::short_sell_volume)
        .def("change_pct", &trade::Bar::change_pct)
        .def("amplitude", &trade::Bar::amplitude)
        .def("open_gap", &trade::Bar::open_gap)
        .def("is_valid", &trade::Bar::is_valid);

    // -----------------------------------------------------------------------
    // BarSeries
    // -----------------------------------------------------------------------

    nb::class_<trade::BarSeries>(m, "BarSeries")
        .def(nb::init<>())
        .def_rw("symbol", &trade::BarSeries::symbol)
        .def_rw("bars", &trade::BarSeries::bars)
        .def("size", &trade::BarSeries::size)
        .def("empty", &trade::BarSeries::empty);

    // -----------------------------------------------------------------------
    // Instrument
    // -----------------------------------------------------------------------

    nb::class_<trade::Instrument>(m, "Instrument")
        .def(nb::init<>())
        .def_rw("symbol", &trade::Instrument::symbol)
        .def_rw("name", &trade::Instrument::name)
        .def_rw("market", &trade::Instrument::market)
        .def_rw("board", &trade::Instrument::board)
        .def_rw("industry", &trade::Instrument::industry)
        .def_prop_rw("list_date",
            [](const trade::Instrument& inst) {
                return inst.list_date.time_since_epoch().count();
            },
            [](trade::Instrument& inst, int d) {
                inst.list_date = trade::Date(std::chrono::days(d));
            })
        .def_prop_rw("delist_date",
            [](const trade::Instrument& inst) -> std::optional<int> {
                if (inst.delist_date.has_value()) {
                    return inst.delist_date->time_since_epoch().count();
                }
                return std::nullopt;
            },
            [](trade::Instrument& inst, std::optional<int> d) {
                if (d.has_value()) {
                    inst.delist_date = trade::Date(std::chrono::days(*d));
                } else {
                    inst.delist_date = std::nullopt;
                }
            })
        .def_rw("status", &trade::Instrument::status)
        .def_rw("total_shares", &trade::Instrument::total_shares)
        .def_rw("float_shares", &trade::Instrument::float_shares)
        .def("is_tradable", &trade::Instrument::is_tradable)
        .def("is_st", &trade::Instrument::is_st)
        .def("days_listed", [](const trade::Instrument& inst, int d) {
            return inst.days_listed(trade::Date(std::chrono::days(d)));
        }, "date"_a)
        .def("is_new_stock", [](const trade::Instrument& inst, int d) {
            return inst.is_new_stock(trade::Date(std::chrono::days(d)));
        }, "date"_a);

    // -----------------------------------------------------------------------
    // MarketSnapshot
    // -----------------------------------------------------------------------

    nb::class_<trade::MarketSnapshot>(m, "MarketSnapshot")
        .def(nb::init<>())
        .def_prop_rw("date",
            [](const trade::MarketSnapshot& s) {
                return s.date.time_since_epoch().count();
            },
            [](trade::MarketSnapshot& s, int d) {
                s.date = trade::Date(std::chrono::days(d));
            })
        .def_rw("bars", &trade::MarketSnapshot::bars)
        .def_rw("instruments", &trade::MarketSnapshot::instruments)
        .def("stock_count", &trade::MarketSnapshot::stock_count)
        .def("has", &trade::MarketSnapshot::has, "symbol"_a)
        .def("bar", &trade::MarketSnapshot::bar, "symbol"_a,
             nb::rv_policy::reference_internal)
        .def("up_count", &trade::MarketSnapshot::up_count)
        .def("down_count", &trade::MarketSnapshot::down_count)
        .def("limit_up_count", &trade::MarketSnapshot::limit_up_count)
        .def("limit_down_count", &trade::MarketSnapshot::limit_down_count)
        .def("total_amount", &trade::MarketSnapshot::total_amount)
        .def("median_turnover", &trade::MarketSnapshot::median_turnover);

    // -----------------------------------------------------------------------
    // MarketPanel
    // -----------------------------------------------------------------------

    nb::class_<trade::MarketPanel>(m, "MarketPanel")
        .def(nb::init<>())
        .def_prop_rw("dates",
            [](const trade::MarketPanel& p) {
                std::vector<int> out;
                out.reserve(p.dates.size());
                for (const auto& d : p.dates) {
                    out.push_back(d.time_since_epoch().count());
                }
                return out;
            },
            [](trade::MarketPanel& p, const std::vector<int>& v) {
                p.dates.clear();
                p.dates.reserve(v.size());
                for (int d : v) {
                    p.dates.push_back(trade::Date(std::chrono::days(d)));
                }
            })
        .def_rw("symbols", &trade::MarketPanel::symbols)
        .def("add_series", &trade::MarketPanel::add_series,
             "symbol"_a, "bar_series"_a)
        .def("snapshot", [](const trade::MarketPanel& p, int d) {
            return p.snapshot(trade::Date(std::chrono::days(d)));
        }, "date"_a);
}
