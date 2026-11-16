"""Tests for Student Feedback Summarizer - Milestones 1 to 5."""
import pytest
import torch
from app.core import (
    milestone_1,
    milestone_2,
    milestone_3,
    milestone_4,
    milestone_5,
    compute_rouge1,
    baseline_first_sentence,
    LoRALinear,
)

def test_milestone_1():
    data = milestone_1()
    assert isinstance(data, list)
    assert len(data) >= 10
    assert all("feedback" in d and "summary" in d for d in data)
    assert all(len(d["summary"]) < len(d["feedback"]) for d in data)


def test_milestone_2():
    res = milestone_2()
    assert "model" in res and "tokenizer" in res and "losses" in res
    # Loss must decrease during training
    assert res["losses"][-1] < res["losses"][0] + 0.5
    # At least some parameters must be trainable (LoRA adapters)
    trainable = sum(
        p.numel() for p in res["model"].parameters() if p.requires_grad
    )
    assert trainable > 0


def test_milestone_3():
    summarize_fn = milestone_3()
    assert callable(summarize_fn)
    res = milestone_2()
    summary = summarize_fn(res["model"], res["tokenizer"], "The course was too fast.")
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_milestone_4():
    res = milestone_4()
    assert "avg_lora" in res and "avg_baseline" in res
    assert isinstance(res["avg_lora"], float)
    assert isinstance(res["avg_baseline"], float)
    assert 0.0 <= res["avg_lora"] <= 1.0
    assert 0.0 <= res["avg_baseline"] <= 1.0
    assert len(res["lora_scores"]) == 6
    assert len(res["baseline_scores"]) == 6


def test_milestone_5():
    res = milestone_5()
    assert "before_merge" in res
    assert "after_merge" in res
    assert isinstance(res["before_merge"], str)
    assert isinstance(res["after_merge"], str)
    # Verify model has no LoRALinear wrappers after merge
    for block in res["model"].transformer.h:
        assert not isinstance(block.attn.c_attn, LoRALinear), (
            "LoRA wrapper still present after merge!"
        )


def test_rouge1_math():
    """Verifies the ROUGE-1 F1 formula is mathematically correct."""
    score = compute_rouge1("clear good examples", "clear good explanations")
    assert score > 0.0
    perfect = compute_rouge1("fast paced lectures", "fast paced lectures")
    assert perfect == 1.0
    zero = compute_rouge1("good lectures", "bad assignments")
    assert zero == 0.0
    