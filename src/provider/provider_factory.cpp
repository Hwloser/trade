#include "trade/provider/provider_factory.h"
#include "trade/provider/eastmoney_provider.h"
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace trade {

std::unique_ptr<IDataProvider> ProviderFactory::create(const std::string& name,
                                                        const Config& config) {
    if (name == "eastmoney") {
        return std::make_unique<EastMoneyProvider>(config.eastmoney);
    }
    throw std::invalid_argument("Unknown provider: " + name);
}

} // namespace trade
