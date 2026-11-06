"""Opt-in offline test against the exact downloaded tokenizer; no GPU/model calls."""
import json
import os

import pytest


def test_cached_tokenizer_table_preserves_all_legal_action_prefixes():
    model_path = os.environ.get("QUERYSEEK_CONSTRAINT_TEST_MODEL")
    if not model_path:
        pytest.skip("Set QUERYSEEK_CONSTRAINT_TEST_MODEL to the pinned local tokenizer")
    import torch
    from transformers import AutoTokenizer
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
    from scripts.gpu_agent_experiment import TransformersBackend

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    backend = object.__new__(TransformersBackend)
    backend.tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    # Return to the first schema after a different one to detect leaked parser state.
    for frame_count in [4, 16, 4]:
        choices = ["abstain", "observe", "zoom_in_left", "zoom_in_right", "zoom_out", "shift"]
        choices += [f"answer:{i}:{j}" for i in range(frame_count) for j in range(i, frame_count)]
        schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": choices}},
                  "required": ["choice"], "additionalProperties": False}
        reference = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(schema))
        cached = backend._prefix_constraint(schema)
        for choice in choices:
            prompt_ids = tokenizer.encode("A fresh search state:", add_special_tokens=False)
            target_ids = tokenizer.encode(json.dumps({"choice": choice}), add_special_tokens=False)
            for index in range(len(target_ids)+1):
                ids = torch.tensor(prompt_ids+target_ids[:index])
                assert set(reference(0, ids)) == set(cached(0, ids))
