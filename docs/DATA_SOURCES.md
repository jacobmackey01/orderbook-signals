# Data sources: multi-session crypto replication

Acquired 2026-10-21 23:34–23:36 UTC (2026-10-22 local) with no account, API key or payment. Raw files are kept in `data/tardis/` (gitignored) and are **not** redistributed with this repository.

## What was checked

| source | outcome |
|---|---|
| LOBSTER public samples (https://php.lobsterdata.com/info/DataSamples.php) | Only 2012-06-21 files are offered (1–50 levels, AAPL/AMZN/GOOG/INTC/MSFT, SPY at 30/50 levels). No additional free equity sessions. Further dates need an academic subscription. |
| Tardis.dev downloadable CSVs (https://docs.tardis.dev/downloadable-csv-files/api) | Chosen. Files for the first day of each month are free for every exchange and data type, without an API key. `incremental_book_L2`, `book_snapshot_25/5`, `trades`, `quotes` are available for Binance spot. |
| Binance public archive (data.binance.vision, S3 listing checked 2026-10-22) | Not used. Spot daily data are only `trades`, `aggTrades` and `klines`. Depth exists only for USDⓈ-M futures, as `bookTicker` (best bid/ask) and `bookDepth` (aggregated depth, which as far as I know is percentage bands rather than price levels; not inspected further). No ten-level price-level book. |
| Coinbase via Tardis | Excluded: Tardis's terms add Coinbase-specific restrictions barring machine-learning use. |

## Chosen data

- Provider: Tardis.dev, `https://datasets.tardis.dev/v1/binance/{incremental_book_L2|trades}/YYYY/MM/01/{BTCUSDT|ETHUSDT}.csv.gz`.
- Venue: Binance spot. Recorded from Tardis's GCP Tokyo region; Binance's servers are in AWS Tokyo (Tardis Binance details page).
- `incremental_book_L2`: one row per price-level update; `is_snapshot` marks snapshot rows; `amount` is the absolute new level size (0 deletes). Binance's own WebSocket has no snapshots, so Tardis fetches a top-1000 REST snapshot at each (re)connection and validates the `U`/`u` sequence numbers during collection, restarting the connection on a missed message. The stream is `depth@100ms` (netted level changes every 100 ms). Sequence numbers are not included in the CSV.
- `trades`: one row per trade with the taker (aggressor) side; Tardis deduplicates CSV trades.
- Timestamps: microseconds since epoch, UTC. Files are split by `local_timestamp` (arrival time), which is monotonic.
- Tardis incident reports (`https://api.tardis.dev/v1/exchanges/binance`, `incidentReports`) list no incidents on the chosen dates (nearest: 2026-06-08 and 2026-08-13).
- Contract details (Binance `GET https://api.binance.com/api/v3/exchangeInfo`, public, retrieved 2026-10-22): BTCUSDT and ETHUSDT tick size 0.01 USDT; quantity step 0.00001 BTC and 0.0001 ETH.
- Fees: Binance spot fee schedule, https://www.binance.com/en/fee/schedule. The page's public data endpoint `https://www.binance.com/bapi/accounts/v1/public/account/trade-level/get` returned, for level 0 (Regular user), `makerCommission` 0.001 and `takerCommission` 0.001, i.e. 0.10% per side (retrieved 2026-10-22; response SHA-256 `2e9eda721fdf960a1ebee17bf7bd69ec993cb55f75aed7954a32b73822c24bdf`). The 25% BNB-payment discount is advertised by Binance but was not verified here.

## Terms (summary, with short quotes)

Source: Tardis.dev Terms of Service and Licence Agreement, https://docs.tardis.dev/legal/terms-of-service, and Billing FAQ, https://docs.tardis.dev/faq/billing-and-subscriptions (read 2026-10-22).

- Free access: the FAQ states first-day-of-month data is free "for every exchange and data type". The API docs say these files are available "without API key".
- Downloading or using any Data, including sample data, means accepting Tardis's Licence Agreement (browsewrap; no account was created).
- Permitted Use covers "internal business, research, educational or personal purposes".
- No redistribution of the Data or Manipulated Data (clause 9.2); raw files, processed caches, predictions and trade-level records therefore stay local and are gitignored.
- Derived Data (charts, statistics, research reports) that cannot substitute for or reconstruct the data may be shared (clause 9.3). The committed results are aggregate metrics and figures.
- Models trained on the data may not be published or made available to others (clause 9.5(b)). This project does not save or publish fitted models.
- Terms may change (clause 20); re-check before reuse.

## Reproduce

Run from the repository root of this worktree. The shared venv has an editable install pointing at another checkout, so `PYTHONPATH=src` is required:

```bash
PYTHONPATH=src ./scripts/python -m lob_research.cli crypto-download   # 48 files, ~4.2 GB, hashes checked against data/tardis-manifest.json
PYTHONPATH=src ./scripts/python -m pytest -q
PYTHONPATH=src ./scripts/python -m lob_research.cli crypto-run --phase development
PYTHONPATH=src ./scripts/python -m lob_research.cli crypto-run --phase heldout    # once, from a committed tree
```

The downloader refuses non-first-of-month dates, writes via a temporary file, checks gzip integrity (Tardis's recommended check) and fails on a hash mismatch with the committed manifest.

## File hashes (SHA-256 of the downloaded .csv.gz)

Total 4,218,579,210 bytes in 48 files; full records, including URLs and CSV line counts, are in `data/tardis-manifest.json`.

| symbol | date | data type | bytes | SHA-256 |
|---|---|---|---|---|
| BTCUSDT | 2025-11-01 | incremental_book_L2 | 84,853,359 | `f30adff81dfacb702793790fdcc6a3fb80dba61c22b0e616cf6f45dcb4aacb00` |
| BTCUSDT | 2025-11-01 | trades | 15,111,108 | `23d886d7862f912521aaee78198afeb2630d0bee1023a28d0946b4945b5a9650` |
| BTCUSDT | 2025-12-01 | incremental_book_L2 | 275,230,587 | `6e71f2b3cd4a45f12b5d3a753e2177323859837d3f6330c2b4bb303058eb4f78` |
| BTCUSDT | 2025-12-01 | trades | 55,797,317 | `3b4301c6b772c4ebc0d91999f157bc62038b6aaa33c57e2be25f6e19161b9bcc` |
| BTCUSDT | 2026-01-01 | incremental_book_L2 | 74,583,372 | `212a5ba898907974f99c804273c69a232dbb9e33e01534a829179033dd9af4a0` |
| BTCUSDT | 2026-01-01 | trades | 11,224,102 | `5963f8f3727153b27d982e6e2c8cd4f06384beb17757e9e8b89d4490f89f7e92` |
| BTCUSDT | 2026-02-01 | incremental_book_L2 | 258,298,508 | `14f847afb82051d3ab726589cc35300398d5d6cea181224237aea3f3b5726fd9` |
| BTCUSDT | 2026-02-01 | trades | 55,264,738 | `e97312e54ff58064dfa8fad5c2899808eedd98d4aea29d701308db2c06d6dcf0` |
| BTCUSDT | 2026-03-01 | incremental_book_L2 | 211,559,118 | `fbc855200e114869ac4912c46994633227675a5bafc79f1ef03c6fe00cfbc1ca` |
| BTCUSDT | 2026-03-01 | trades | 39,424,524 | `6c476f05ad9e4593bb586bb3bdf3f7ef17647791d10c8cc6f341c75ac06400e0` |
| BTCUSDT | 2026-04-01 | incremental_book_L2 | 191,040,125 | `c95b3f1f71d10218e850124e4d5a62642bc19cdddf2393a07a4e437b22d2d512` |
| BTCUSDT | 2026-04-01 | trades | 25,054,052 | `89c63587a140e815a5dd882e74fc61501e906a2a343a6b07ef8cb1da31f34eb1` |
| BTCUSDT | 2026-05-01 | incremental_book_L2 | 139,318,781 | `841b2dc5d0b065961c8d6e1f1a47b20c25e33d0b0ced1dec4de6803879920a39` |
| BTCUSDT | 2026-05-01 | trades | 17,002,571 | `b1de02eef9ad4a916ae15fb35812eff8208fd618dd9742aa7c9bcbd519333af0` |
| BTCUSDT | 2026-06-01 | incremental_book_L2 | 184,958,944 | `4dbd5de5bbfc64114d74c8709548ce26bc7ec39f3664948f092fa97da55243ee` |
| BTCUSDT | 2026-06-01 | trades | 29,496,391 | `72b6722704a695c6fd36663de458b2e3a24e29c9e922210e7d5ea708885e1fd9` |
| BTCUSDT | 2026-07-01 | incremental_book_L2 | 176,400,395 | `e15c61d568f6d58f7fce63ce4d9c1583acf9d4962baf56a0a4c4eea18dec8a52` |
| BTCUSDT | 2026-07-01 | trades | 27,972,950 | `f994c1852e8e5dac29bd2cd98c9022eb50a8dcb96fb44c80342e0b64279e586b` |
| BTCUSDT | 2026-08-01 | incremental_book_L2 | 57,384,961 | `b28c60ce9098d31d3aa84cb074bad24ca33aa5feb07f21a6f8cd2e716fb2db76` |
| BTCUSDT | 2026-08-01 | trades | 6,338,707 | `95e59ad8bccee8768fe0370bc0f3e18ab3f65262fbe0089dfc0575c439ec26a9` |
| BTCUSDT | 2026-09-01 | incremental_book_L2 | 146,571,680 | `cf97b98ca4d97f3cd80d7905c59d8894be9140475d4d39c19f6ece0b28dfb4d9` |
| BTCUSDT | 2026-09-01 | trades | 24,277,959 | `3fef1165d2683638a958d9b92745c442cdb2a8b0fdce77e2e22505da303eb8d3` |
| BTCUSDT | 2026-10-01 | incremental_book_L2 | 146,790,051 | `f6cfba668c552ccb49a51aa80a85982a313bf8464082cd0f35568884ad6ac8a7` |
| BTCUSDT | 2026-10-01 | trades | 18,560,179 | `7641440ba8e6aee25700acd60f76902400a508488d174bb85f2cb161a770ecc6` |
| ETHUSDT | 2025-11-01 | incremental_book_L2 | 85,733,154 | `28be9ec93057541d6cfb1ab9a3c42160f486cc83070e674f47291d4268fbeabd` |
| ETHUSDT | 2025-11-01 | trades | 14,269,061 | `f56c2cf6ded79ee5340ed03c5cbe6ee6f3ddb26b9c9baf77298d6957fefead02` |
| ETHUSDT | 2025-12-01 | incremental_book_L2 | 216,118,725 | `41bc2ed0a607fbc5a6ab33cff1e1042e49d6b44f5ce60cea84ff0627af37069e` |
| ETHUSDT | 2025-12-01 | trades | 54,611,963 | `5d566a00b7d4e3a92a1fb5010dd14b89cab409420ef625ed7484b5bcdc810d10` |
| ETHUSDT | 2026-01-01 | incremental_book_L2 | 59,225,930 | `5d6f16ba8fac40b304bba16cf1f13ff1ec0bb0624cf8c6b70b341544e2d58895` |
| ETHUSDT | 2026-01-01 | trades | 9,560,245 | `d151d49e1751840d77241f83e4845d43ebec08bedf7c989703a921c8d0065578` |
| ETHUSDT | 2026-02-01 | incremental_book_L2 | 272,558,186 | `6c1e87582c82dbbf1bd8b920b8b9fba4bf71da32869f68741cc0b56a68e2c027` |
| ETHUSDT | 2026-02-01 | trades | 79,225,704 | `395efcc692f9a3f3b71bc3c2973369197257eccd7d0c0486c528f575cdec59b0` |
| ETHUSDT | 2026-03-01 | incremental_book_L2 | 196,710,172 | `77a331e338d521a8ba1e47e97155ba1f82f2bb05b2f83bfd9f43014ace828e9b` |
| ETHUSDT | 2026-03-01 | trades | 49,022,412 | `e9df54cd315671c70c78577fedd7e5c71b46d6eec6e1ac7876aeb876f57cf3e0` |
| ETHUSDT | 2026-04-01 | incremental_book_L2 | 150,432,415 | `960cd03139149761dd50e9f98565404bf5c2eeb854c22bc08efb233ebe022464` |
| ETHUSDT | 2026-04-01 | trades | 29,056,400 | `7a027616279bb46c970c982b658b0743ad7367819f01fb9f9aeec379c2b904eb` |
| ETHUSDT | 2026-05-01 | incremental_book_L2 | 99,069,712 | `189c065b33313575eec52b438e7e488e8978853ae2b8e89ad29b470220366f8c` |
| ETHUSDT | 2026-05-01 | trades | 13,058,668 | `f3dd1395be65367cf9a0c9b722e29eea51dcd3d7b94bfe08b45eaafd288d59bf` |
| ETHUSDT | 2026-06-01 | incremental_book_L2 | 122,374,084 | `b389ab28b1435eb551c2b31cee5e0561afbb19d838acc108fcbeb117a532e462` |
| ETHUSDT | 2026-06-01 | trades | 26,900,339 | `311805d385fd300f98f9b1ef6e0ad7a2884f492af8d0782e0fbfe00552a9cf2d` |
| ETHUSDT | 2026-07-01 | incremental_book_L2 | 135,520,302 | `4d63e05bc410f5db9f9a4a4869680f13bf07d759d2508829d8fae7a97ca98a17` |
| ETHUSDT | 2026-07-01 | trades | 19,983,673 | `ba8246f268e0588f7720386a6a4cfccaf486214214f6e19a87a2c96dc521b5a5` |
| ETHUSDT | 2026-08-01 | incremental_book_L2 | 50,670,537 | `469c79f165eae6ffe47bd8cd37764ab378861002ecff44d2b847aa33c29d1921` |
| ETHUSDT | 2026-08-01 | trades | 6,342,660 | `0167c47b34e959fb066036fed483de8b8f691852ebfe11821faf922e8eeb93d1` |
| ETHUSDT | 2026-09-01 | incremental_book_L2 | 109,344,864 | `c2261135a093651edc7e9979fded8ab256180404fb0546cbe25b81edfa09368d` |
| ETHUSDT | 2026-09-01 | trades | 17,705,636 | `a79af6a22aa2de2314ba4193623fdb0bde836fa832370813514fe7b738e9a200` |
| ETHUSDT | 2026-10-01 | incremental_book_L2 | 112,112,980 | `01108cff2dba18182f233ed785be38ac61ba56ae90ebe00c3b60ba98d09596a9` |
| ETHUSDT | 2026-10-01 | trades | 16,456,909 | `745cb2b2146a7437dd1a03d6ebe6c4d624ba1dec77b4ea62720074e6d82e23ee` |
