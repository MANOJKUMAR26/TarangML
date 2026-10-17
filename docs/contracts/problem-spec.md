# ProblemSpec

The output of steps 1 and 2, read by every later step. Model: `ProblemSpec` in `src/tinyml/contracts.py`.

```json
{
  "recipe": "pump_health",
  "task": "anomaly",
  "vendor": "any",
  "connectivity": "wifi",
  "power": "battery",
  "budget_usd_per_unit": 5,
  "priority": "battery",
  "max_latency_ms": 1000,
  "description": "Alert me when my pump vibration looks abnormal…",
  "seed": 42
}
```

| field | values | notes |
|---|---|---|
| recipe | an id in `recipes/` | picks windows, features, targets and minimum data |
| task | `anomaly`, `classification` | anomaly trains on normal data only |
| vendor | `any`, `espressif`, `st`, `nordic`, `raspberrypi` | hard filter |
| connectivity | `wifi`, `ble`, `cellular`, `none`, `any` | `wifi`/`ble` are hard filters; cellular assumes an external modem |
| power | `battery`, `plugged` | sets the default priority |
| budget_usd_per_unit | number or null | hard filter on chip price |
| priority | `battery`, `cost`, `speed` | weights of the ranking score |
| max_latency_ms | number | hard filter, together with the hop time |
| seed | integer | fixed for every random step, recorded in the package |

# CSV summary

What the assistant sees instead of raw rows (`CsvSummary` / `ColumnSummary`). Per column: name, type (`numeric`, `text`, `datetime`), count, missing, min, max, mean, std, number of unique values, whether it always increases within each file, the median step, and a three-value sample; plus the use-case text. Raw sensor data is never sent to an assistant and never stored in the decision log.
