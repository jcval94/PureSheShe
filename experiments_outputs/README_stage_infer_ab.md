# Stage inference A/B results

This summary captures the latest regression results comparing the legacy
`_infer_stage` helper against the optimised implementation shipped with the
engine.

## Scenario compatibility

The comparison exercises four scenarios, including two score-adaptor runs that
force pandas-style DataFrame inputs and cache utilisation. Every scenario
produced matching stage names across both implementations.

| Scenario | Context notes | Legacy stage | Current stage |
| --- | --- | --- | --- |
| fake_deldel_stage | Simulated `DelDel.stage` call | `DelDel.stage` | `DelDel.stage` |
| plain_helper | Direct helper invocation | `deldel.stage_infer_ab.helper` | `deldel.stage_infer_ab.helper` |
| score_adaptor_dataframe_small | 3×4 float DataFrame batch | `DelDel.stage` | `DelDel.stage` |
| score_adaptor_dataframe_duplicate_rows | 4×5 DataFrame with duplicate rows | `DelDel.stage` | `DelDel.stage` |

See `infer_stage_ab_results.csv` for the authoritative record, including shape
metadata for the DataFrame scenarios.【F:experiments_outputs/infer_stage_ab_results.csv†L1-L6】

## Performance snapshot

Benchmarking the helpers across two iteration counts (each repeated four
times) shows the optimised helper consistently running over 100× faster than
the legacy variant.

| Iterations | Legacy avg runtime (s) | Current avg runtime (s) | Avg runtime ratio |
| --- | --- | --- | --- |
| 5,000 | 1.57 | 0.0147 | 0.00936 |
| 20,000 | 6.37 | 0.0597 | 0.00941 |

The detailed measurements, including every repeat, are stored in
`infer_stage_ab_timings.csv` for reference.【F:experiments_outputs/infer_stage_ab_timings.csv†L1-L25】
