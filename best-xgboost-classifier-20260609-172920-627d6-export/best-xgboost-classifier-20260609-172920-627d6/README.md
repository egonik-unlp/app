# Lensing model export — `best-xgboost-classifier-20260609-172920-627d6`

Portable, self-contained export of a trained `xgboost-classifier` model. Everything here is
framework-agnostic: an ONNX graph plus a declarative preprocessing spec. The only thing
you bring is an ONNX runtime (e.g. `onnxruntime-web`) and, optionally, the reference
featurizer `@lensing/inference`.

## Files

- `lensing-export.json` — manifest: schema version, predictor family, target spec, file index.
- `model.onnx` — the trained predictor as an ONNX graph.
  - input `input`: `float32[N, 115]` — the **assembled feature vector**.
  - output `output`: `float32[N, 1]` — the target in **transformed space**.
- `featurize.json` — how to build the feature vector from a raw item (see below).
- `pca_components.f32` — PCA basis, row-major `float32[1, 64]`.
- `input-schema.json` — the fields a caller must provide.

## Building the feature vector (`featurize.json`)

`featurize.json.columns` is an ordered list of 115 instructions, one per feature
column, each tagged by `op`:

- `pca` — component of the PCA projection of the embedding: subtract `pca.mean` from the
  `64`-dim embedding, multiply by `pca_components.f32` (`[dims, 64]`).
- `numeric_verbatim` / `numeric_log1p` / `numeric_present_raw` / `numeric_missing_flag` —
  read `field` from the item (0/absent treated as unspecified); see each op's rule.
- `coord_lat` / `coord_lon` / `coord_missing` — read `field` as `{lat, lon}`; in-bounds
  (`bounds = [[lat_min,lat_max],[lon_min,lon_max]]`) → value, else 0 / missing-flag 1.
- `onehot` — 1 when the item's `group` equals `value`; the trailing `value="__other__"`
  column is the catch-all for unseen values.

If `featurize.json.imputation` is present, fill missing numerics with the frozen medians
(keyed by the outlier-group field) **before** encoding.

## Target

`featurize.json.target.transform` = none — `out` is already in target space; clamp at 0.

## Inference recipe

1. Get the item's `64`-dim embedding (same embedding model the corpus used).
2. Build `features: float32[N, 115]` per `featurize.json.columns`.
3. Run `model.onnx` with input `input=features` → `output[N,1]` (transformed space).
4. Invert the target transform and clamp at 0.

Provenance: model `best-xgboost-classifier-20260609-172920-627d6` · run `run-20260609-172920-627d6-xgboost-classifier` · dataset `ds-20260609-141622-p1-s42` · created 2026-06-09T17:29:22.374555606+00:00.
