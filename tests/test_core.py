"""Tests for Student Feedback Summarizer."""
import pytest
import torch
from app.core import (
    load_feedback_data,
    build_model,
    summarize,
    evaluate_model,
    export_merged_model,
    compute_rouge1,
    baseline_first_sentence,
    LoRALinear,
)


def test_feedback_dataset():
    data = load_feedback_data()
    assert isinstance(data, list)
    assert len(data) >= 10
    assert all("feedback" in d and "summary" in d for d in data)
    assert all(len(d["summary"]) < len(d["feedback"]) for d in data)


def test_lora_training():
    res = build_model()
    assert "model" in res and "tokenizer" in res and "losses" in res
    assert res["losses"][-1] < res["losses"][0] + 0.5
    trainable = sum(
        p.numel() for p in res["model"].parameters() if p.requires_grad
    )
    assert trainable > 0


def test_summarize_function():
    res = build_model()
    summary = summarize(res["model"], res["tokenizer"], "The course was too fast.")
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_evaluation_with_baseline():
    res = evaluate_model()
    assert "avg_lora" in res and "avg_baseline" in res
    assert isinstance(res["avg_lora"], float)
    assert isinstance(res["avg_baseline"], float)
    assert 0.0 <= res["avg_lora"] <= 1.0
    assert 0.0 <= res["avg_baseline"] <= 1.0
    assert len(res["lora_scores"]) == 6
    assert len(res["baseline_scores"]) == 6


def test_adapter_merging():
    res = export_merged_model()
    assert isinstance(res["before_merge"], str)
    assert isinstance(res["after_merge"], str)
    for block in res["model"].transformer.h:
        assert not isinstance(block.attn.c_attn, LoRALinear)


def test_rouge1_math():
    perfect = compute_rouge1("fast paced lectures", "fast paced lectures")
    assert perfect == 1.0
    zero = compute_rouge1("good lectures", "bad assignments")
    assert zero == 0.0


def test_baseline_is_naive():
    out = baseline_first_sentence("The lectures were too fast and hard to follow.")
    assert isinstance(out, str)
    assert len(out) > 0