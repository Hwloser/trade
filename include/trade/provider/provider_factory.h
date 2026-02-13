#pragma once

#include "trade/provider/i_data_provider.h"
#include "trade/common/config.h"
#include <memory>
#include <string>

namespace trade {

class ProviderFactory {
public:
    static std::unique_ptr<IDataProvider> create(const std::string& name,
                                                  const Config& config);
};

} // namespace trade
