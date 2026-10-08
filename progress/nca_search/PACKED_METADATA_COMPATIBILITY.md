# Search packed-data metadata

The search samplers write new selected `.pt` datasets directly. They do not rewrite the raw NCA v1 trajectory files or the original packed reference. Each `.pt` still contains `sequences`, `targets`, and `metadata`; `sequences` and `targets` retain the original `int32[world, 1026]` layout and token/mask values. The model's training view uses `metadata["split_indices"]` and `metadata["grid_len"]`.

The **metadata keys are not identical** to the packed reference built by `examples/transfer_1m.py`:

| Reference packed metadata | Search packed metadata |
| --- | --- |
| `grid_len`, `num_colors`, `patch_size`, `packing_config`, `split_indices`, `usable_training_tokens`, `selection` | Same keys, with selected-corpus values |
| `packed_tokens` | Omitted; derive as `sequences.numel()` |
| `source_directory`, `source_manifest_sha256` | Omitted because these selected worlds are assembled from a new rule stream rather than one raw v1 manifest |
| — | `source_generator`, `source_reference_sha256`, `gzip_subbin_edges`, `gzip_subbin_quotas`, and selector-specific thresholds; the boosted sweep also records `source_anchor_packed_sha256` |

This is a provenance-metadata difference in a **new optional packed artifact**, not a change to the raw trajectory v1 schema, tensor format, tokenization, or model inputs. `PackedDataset.load` reads both dictionaries without schema validation, and the search training path consumes the shared split and grid keys. A consumer requiring reference-only fields can use the compatibility expressions below; no original file or active run needs mutation:

```python
packed_tokens = packed.metadata.get("packed_tokens", packed.sequences.numel())
source_directory = packed.metadata.get("source_directory")
source_manifest_sha256 = packed.metadata.get("source_manifest_sha256")
source_reference_sha256 = packed.metadata.get("source_reference_sha256")
```

For a selected search corpus, the two reference-only source values are genuinely unavailable; assigning them the reference values would misstate provenance. The search `sampling_config.json`, `rule_index.jsonl`, `done.json`, and audit receipts are its source records. Keep this distinction explicit in any downstream export.
