# Repository instructions

- All public times are floating-point seconds and half-open intervals: `0 <= start < end <= duration`.
- Use existing FFmpeg/PySceneDetect/BM25/OpenCLIP/LangGraph components; record third-party reuse in THIRD_PARTY.md.
- External commands use argument lists, `shell=False` and explicit timeouts. Validate outputs under `outputs/`.
- Read secrets through environment variables; never write keys or raw image data into traces.
- Preserve CPU/no-key Mock execution. Mock validates software, never model accuracy.
- Cache keys include video hash, pipeline version and step configuration. Preserve pipeline-v2 source timestamps.
- New behavior requires relevant pytest; synthetic videos must come from scripts/make_synthetic_fixture.py.
- Research scripts may require unpublished local data; report missing assets honestly. Never fabricate metrics.
- Read docs/RESEARCH_ROADMAP.md before planning new experiments. New research results remain TODO / unmeasured.
