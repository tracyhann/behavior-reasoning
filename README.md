# Behavior Reasoning

This branch adds `meta_baselines/`, a reproducible package for CARE-style social-interaction baseline experiments.

Start here:

```text
meta_baselines/README.md
```

The package includes CE003, SIV-Bench, and Video-MME evaluation artifacts plus a pluggable backend runner. The backend runner can call Qwen2.5-VL, another local VLM, an API-backed model, or any subprocess that returns JSON. Raw media is intentionally excluded from git; manifests and expected paths are included.

