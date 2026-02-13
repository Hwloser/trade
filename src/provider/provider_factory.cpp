#include "trade/provider/provider_factory.h"
#include <spdlog/spdlog.h>
#include <stdexcept>

// Forward declarations of provider classes
namespace trade {

// Defined in akshare_provider.cpp
class AkShareProvider : public IDataProvider {
public:
    explicit AkShareProvider(const AkShareConfig& config);
    std::string name() const override;
    std::vector<Bar> fetch_daily(const Symbol& symbol, Date start, Date end) override;
    std::vector<Instrument> fetch_instruments() override;
    bool ping() override;
private:
    AkShareConfig config_;
};

std::unique_ptr<IDataProvider> ProviderFactory::create(const std::string& name,
                                                        const Config& config) {
    if (name == "akshare") {
        return std::make_unique<AkShareProvider>(config.akshare);
    }
    throw std::invalid_argument("Unknown provider: " + name);
}

} // namespace trade
