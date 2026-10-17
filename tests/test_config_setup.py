"""`hermes memory setup` integration: schema choices and typed config saving."""

from __future__ import annotations

import yaml


def test_schema_choices_are_strings(memohood):
    # hermes' curses picker renders each choice as text; bool items crash it.
    schema = memohood.provider.MemoHoodMemoryProvider().get_config_schema()
    for field in schema:
        for choice in field.get("choices") or []:
            assert isinstance(choice, str), field["key"]
        if field.get("choices"):
            assert field["default"] in field["choices"], field["key"]


def test_save_coerces_wizard_strings_to_default_types(memohood, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    memohood.config.save_memohood_config_at(
        {
            "embedder.dims": "1024",
            "post_recall.mmr.lambda": "0.7",
            "post_recall.mmr.enabled": "false",
            "auto_capture": "true",
            "capture_threshold": "5",
            "gate.meaningful_terms_floor": "3",
            "gate.backend": "model2vec",
            "graph_rerank.max_neighbors": "not-a-number",
            "embedder": {"provider": "cloudflare"},
        },
        str(home),
    )
    saved = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))["memory"]["memohood"]
    assert saved["embedder"]["dims"] == 1024 and isinstance(saved["embedder"]["dims"], int)
    assert saved["embedder"]["provider"] == "cloudflare"
    assert saved["post_recall"]["mmr"]["lambda"] == 0.7
    assert saved["post_recall"]["mmr"]["enabled"] is False
    assert saved["auto_capture"] is True
    assert saved["capture_threshold"] == 5.0 and isinstance(saved["capture_threshold"], float)
    assert saved["gate"]["meaningful_terms_floor"] == 3
    assert saved["gate"]["backend"] == "model2vec"
    # Unparseable value is dropped rather than saved as a string.
    assert "max_neighbors" not in (saved.get("graph_rerank") or {})
