# Benchmark CSV format

Place the selected paper benchmark here as `causal_pairs.csv` (do not commit the full dataset if its license or redistribution terms do not allow it).

Required columns:

```csv
entity_1,entity_2,context,label
FGF6,prostate cancer,"FGF6 contributes to the growth of prostate cancer.",causal
```

- `entity_1`, `entity_2`: English entity names that can be resolved to a unique exact Hetionet node name/ID. The evaluation/preparation resolver intentionally does not silently select the first fuzzy match. If the benchmark uses alternate names or gene symbols not matching the canonical node name, apply a curated alias-to-node-ID mapping before running the scripts.
- `context`: English metadata/text used for prediction.
- `label`: `causal` or `non-causal`; this field is for evaluation only and is never passed into the inference prompt.

The row above is a schema/example only, not a verified COMAGC example or a guarantee that the exact pair is connected in Hetionet. Prefer the released/preprocessed benchmark files from the paper repository and document any name mappings and excluded rows.
