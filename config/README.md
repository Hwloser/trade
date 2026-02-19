# Config Layout

`config/config.yaml` is the entry file.
It supports `includes` and merges module files in order.

## Module files

- `config/modules/data_ingestion.yaml`: data root and ingestion windows
- `config/modules/provider_eastmoney.yaml`: provider connectivity
- `config/modules/storage_baidu.yaml`: local/cloud storage policy + Baidu credentials
- `config/modules/sentiment.yaml`: sentiment pipeline defaults + RSS feeds

## Usage

- Default: `trade_cli --config config ...`
- Single file also works: `trade_cli --config config/config.yaml ...`

## Override priority

1. Files listed earlier in `includes`
2. Files listed later in `includes` (override earlier)
3. Keys written directly in `config/config.yaml` (highest)
4. Environment fallback for secrets:
   - `BAIDU_ACCESS_TOKEN`
   - `BAIDU_REFRESH_TOKEN`
   - `BAIDU_APP_KEY`
   - `BAIDU_APP_SECRET`
