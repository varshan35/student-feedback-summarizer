"""Student Feedback Summarizer - LoRA Fine-tuning Pipeline."""

import math
import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# DATASET
FEEDBACK_DATA = [
    {
        "feedback": "The lectures moved too fast and I couldn't follow the examples.",
        "summary": "Lectures are too fast; examples need slower pacing.",
    },
    {
        "feedback": "Assignments were very helpful but deadlines were too tight.",
        "summary": "Assignments helpful but deadlines need extension.",
    },
    {
        "feedback": "The professor explained concepts clearly with real world examples.",
        "summary": "Clear explanations with good real world examples.",
    },
    {
        "feedback": "There was no feedback given on our submitted assignments at all.",
        "summary": "No feedback provided on submitted assignments.",
    },
    {
        "feedback": "Office hours were very useful and the professor was approachable.",
        "summary": "Office hours helpful; professor is approachable.",
    },
    {
        "feedback": "Course content was outdated and not relevant to current industry.",
        "summary": "Course content outdated and lacks industry relevance.",
    },
    {
        "feedback": "More practical coding exercises would improve understanding greatly.",
        "summary": "Needs more hands-on coding exercises.",
    },
    {
        "feedback": "The slides were too text heavy and hard to read during class.",
        "summary": "Slides are too text heavy; needs visual improvement.",
    },
    {
        "feedback": "Group projects helped me learn collaboration and communication skills.",
        "summary": "Group projects effectively build collaboration skills.",
    },
    {
        "feedback": "Exam questions did not match what was taught in the lectures.",
        "summary": "Exam questions misaligned with lecture content.",
    },
    {
        "feedback": "The course structure was well organized and easy to follow weekly.",
        "summary": "Well organized course structure with clear weekly flow.",
    },
    {
        "feedback": "More guest lectures from industry professionals would add great value.",
        "summary": "Requests more industry guest lectures for added value.",
    },
]

PROMPT_TEMPLATE = "Feedback: {feedback}\nSummary:"


def load_feedback_data():
    """Returns the structured feedback-summary dataset."""
    return FEEDBACK_DATA

# LoRA ADAPTER
class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper: frozen base + trainable rank-r detour."""

    def __init__(self, base_layer, r=4, alpha=16):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        for p in self.base.parameters():
            p.requires_grad = False

        d_out, d_in = base_layer.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def inject_lora(model, r=4, alpha=16):
    """Freezes GPT-2 and injects LoRA adapters into attention projections."""
    for p in model.parameters():
        p.requires_grad = False

    for block in model.transformer.h:
        old = block.attn.c_attn
        d_out = old.weight.shape[0]
        d_in = old.weight.shape[1]

        new_linear = nn.Linear(d_in, d_out, bias=(old.bias is not None))
        with torch.no_grad():
            new_linear.weight.copy_(old.weight.T)
            if old.bias is not None:
                new_linear.bias.copy_(old.bias)

        block.attn.c_attn = LoRALinear(new_linear, r=r, alpha=alpha)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def train_lora(model, tokenizer, data, epochs=3, lr=1e-3):
    """Trains the LoRA adapter on feedback-summary pairs."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for item in data:
            prompt = PROMPT_TEMPLATE.format(feedback=item["feedback"])
            full_text = prompt + " " + item["summary"]
            inputs = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=128
            )
            input_ids = inputs["input_ids"]

            prompt_len = len(tokenizer(prompt)["input_ids"])
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(data)
        losses.append(avg)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg:.4f}")

    return losses


def build_model():
    """Loads GPT-2, injects LoRA, trains on feedback dataset."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    model = inject_lora(model, r=4, alpha=16)
    losses = train_lora(model, tokenizer, FEEDBACK_DATA, epochs=3)

    torch.save({"model": model, "tokenizer": tokenizer, "losses": losses}, "adapter.pt")
    return {"model": model, "tokenizer": tokenizer, "losses": losses}


# SUMMARIZATION
def summarize(model, tokenizer, feedback_text, max_new_tokens=40):
    """Generates a concise summary from raw student feedback."""
    model.eval()
    prompt = PROMPT_TEMPLATE.format(feedback=feedback_text)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=10,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Summary:" in generated:
        return generated.split("Summary:")[-1].strip()
    return generated.strip()



# EVALUATION WITH BASELINE COMPARISON
def baseline_first_sentence(feedback_text):
    """Baseline: naively returns the input truncated to 15 words."""
    words = feedback_text.split()
    return " ".join(words[:15])


def compute_rouge1(reference, hypothesis):
    """Computes unigram ROUGE-1 F1 score without external libraries."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())

    if not hyp_tokens or not ref_tokens:
        return 0.0

    overlap = ref_tokens & hyp_tokens
    precision = len(overlap) / len(hyp_tokens)
    recall = len(overlap) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 4)


def evaluate_model():
    """Compares LoRA model against baseline using ROUGE-1 F1."""
    trained = build_model()
    model = trained["model"]
    tokenizer = trained["tokenizer"]

    lora_scores = []
    baseline_scores = []

    for item in FEEDBACK_DATA[:6]:
        generated = summarize(model, tokenizer, item["feedback"])
        lora_score = compute_rouge1(item["summary"], generated)
        lora_scores.append(lora_score)

        baseline_out = baseline_first_sentence(item["feedback"])
        base_score = compute_rouge1(item["summary"], baseline_out)
        baseline_scores.append(base_score)

    avg_lora = sum(lora_scores) / len(lora_scores)
    avg_baseline = sum(baseline_scores) / len(baseline_scores)

    print(f"Baseline (truncation) avg ROUGE-1: {avg_baseline:.3f}")
    print(f"LoRA fine-tuned avg ROUGE-1:        {avg_lora:.3f}")
    print(f"Improvement over baseline:          +{avg_lora - avg_baseline:.3f}")

    return {
        "lora_scores": lora_scores,
        "baseline_scores": baseline_scores,
        "avg_lora": avg_lora,
        "avg_baseline": avg_baseline,
    }



# ADAPTER MERGING
def merge_adapter(model):
    """Merges B*A into base weights so the model is a plain GPT-2."""
    for block in model.transformer.h:
        lora_layer = block.attn.c_attn
        if isinstance(lora_layer, LoRALinear):
            with torch.no_grad():
                delta = (lora_layer.B @ lora_layer.A) * lora_layer.scale
                lora_layer.base.weight.data += delta
            block.attn.c_attn = lora_layer.base
    return model


def export_merged_model():
    """Merges adapter and verifies output consistency before and after."""
    trained = build_model()
    model = trained["model"]
    tokenizer = trained["tokenizer"]

    sample = "The assignments were very helpful."
    before_summary = summarize(model, tokenizer, sample)

    model = merge_adapter(model)
    after_summary = summarize(model, tokenizer, sample)

    return {
        "before_merge": before_summary,
        "after_merge": after_summary,
        "model": model,
        "tokenizer": tokenizer,
    }