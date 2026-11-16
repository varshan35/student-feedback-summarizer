"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

"""Student Feedback Summarizer - LoRA Fine-tuning Pipeline."""
import math
import os
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# MILESTONE 1: SYNTHETIC FEEDBACK DATASET
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
def milestone_1():
    """Returns the structured feedback-summary dataset."""
    return FEEDBACK_DATA


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).



# Milestone 3: Develop a Streamlit UI to upload feedback and trigger summarization.


# Milestone 4: Evaluate the quality of generated summaries against original feedback.


# Milestone 5: Implement adapter merging for inference.
class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper: frozen base + trainable rank-r detour (Drill 3)."""
    def __init__(self, base_layer, r=4, alpha=16):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        # Freeze the original pretrained weights
        for p in self.base.parameters():
            p.requires_grad = False
        d_out, d_in = base_layer.weight.shape
        # A: random init (compressed), B: zero init (safe no-op at step 0)
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
def add_lora(model, r=4, alpha=16):
    """Injects LoRA adapters into GPT-2 attention projections."""
    # Step 1: Freeze every weight in the base model
    for p in model.parameters():
        p.requires_grad = False
    # Step 2: Wrap only the attention c_attn projections
    for block in model.transformer.h:
        old = block.attn.c_attn
        d_out = old.weight.shape[0]
        d_in = old.weight.shape[1]
        # Create a standard nn.Linear matching c_attn dimensions
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
            # Only compute loss on the summary tokens
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
def milestone_2():
    """Loads GPT-2, injects LoRA, trains on feedback dataset."""
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model = add_lora(model, r=4, alpha=16)
    losses = train_lora(model, tokenizer, FEEDBACK_DATA, epochs=3)
    # Save adapter weights only
    adapter_path = "adapter.pt"
    adapter_state = {
        k: v
        for k, v in model.state_dict().items()
        if "lora" in k.lower() or ".A" in k or ".B" in k
    }
    torch.save({"model": model, "tokenizer": tokenizer, "losses": losses}, adapter_path)
    return {"model": model, "tokenizer": tokenizer, "losses": losses}