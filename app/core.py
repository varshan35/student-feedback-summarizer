"""Student Feedback Summarizer - LoRA Fine-tuning Pipeline."""

import copy
from collections import Counter
from pathlib import Path
import re

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_NAME = "distilgpt2"
LORA_R = 8
LORA_ALPHA = 16
EPOCHS = 8
LEARNING_RATE = 2e-3
SEED = 1337
ADAPTER_PATH = Path("adapter.pt")


# SYNTHETIC FEEDBACK DATASET
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
    {
        "feedback": "The online quizzes every week kept me on track with the material.",
        "summary": "Weekly quizzes help students stay on track.",
    },
    {
        "feedback": "The teaching assistants were slow to answer questions on the forum.",
        "summary": "TAs respond too slowly on the forum.",
    },
    {
        "feedback": "I loved the final project because it let me build something real.",
        "summary": "Final project was engaging and practical.",
    },
    {
        "feedback": "The classroom was always too crowded and the projector barely worked.",
        "summary": "Classroom overcrowded; projector unreliable.",
    },
    {
        "feedback": "Grading criteria were unclear so I never knew how to improve.",
        "summary": "Grading criteria unclear; improvement guidance missing.",
    },
    {
        "feedback": "Recorded lectures made it easy to review difficult topics later.",
        "summary": "Recorded lectures help review difficult topics.",
    },
    {
        "feedback": "The reading list was far too long for a single semester.",
        "summary": "Reading list too long for one semester.",
    },
    {
        "feedback": "Peer review sessions gave me useful ideas to improve my work.",
        "summary": "Peer review sessions provide useful improvement ideas.",
    },
]
# The last six examples are never trained on, so ROUGE measures generalization.
TRAIN_DATA = FEEDBACK_DATA[:-6]
EVAL_DATA = FEEDBACK_DATA[-6:]

PROMPT_TEMPLATE = "Feedback: {feedback}\nSummary:"


def milestone_1():
    """Returns the structured feedback-summary dataset."""
    return FEEDBACK_DATA


#  LORA ADAPTER INJECTION AND TRAINING
class LoRALinear(nn.Module):
    """Drop-in LoRA wrapper: frozen base + trainable rank-r detour."""

    def __init__(self, base_layer, r=LORA_R, alpha=LORA_ALPHA):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        for p in self.base.parameters():
            p.requires_grad = False
        d_out, d_in = base_layer.weight.shape  # nn.Linear stores (out, in)
        # A: random init (compress), B: zero init (a no-op at step 0)
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model, r=LORA_R, alpha=LORA_ALPHA, verbose=False):
    """Injects LoRA adapters into GPT-2 attention projections."""
    for p in model.parameters():
        p.requires_grad = False

    for block in model.transformer.h:
        old = block.attn.c_attn
        # Hugging Face GPT-2 uses Conv1D, which stores its weight as
        # (in_features, out_features): the transpose of nn.Linear.
        weight = old.weight if isinstance(old, nn.Linear) else old.weight.T
        d_out, d_in = weight.shape

        new_linear = nn.Linear(d_in, d_out, bias=old.bias is not None)
        with torch.no_grad():
            new_linear.weight.copy_(weight)
            if old.bias is not None:
                new_linear.bias.copy_(old.bias)
        block.attn.c_attn = LoRALinear(new_linear, r=r, alpha=alpha)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def load_base():
    """Loads the pretrained tokenizer and model."""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    return model, tokenizer


def train_lora(model, tokenizer, data, epochs=EPOCHS, lr=LEARNING_RATE, verbose=False):
    """Trains the LoRA adapter on feedback-summary pairs."""
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for item in data:
            prompt = PROMPT_TEMPLATE.format(feedback=item["feedback"])
            # The end-of-text token teaches the model when to stop.
            full_text = prompt + " " + item["summary"] + tokenizer.eos_token
            input_ids = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=128
            )["input_ids"]

            # Only compute loss on the summary tokens
            prompt_len = len(tokenizer(prompt)["input_ids"])
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100

            loss = model(input_ids=input_ids, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(data)
        losses.append(avg)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg:.4f}")
    return losses


def adapter_state_dict(model):
    """Returns only the LoRA matrices (the A and B parameters)."""
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k.endswith((".A", ".B"))
    }


def save_adapter(model, path):
    """Saves just the adapter weights, a tiny file compared with the model."""
    first = model.transformer.h[0].attn.c_attn
    torch.save(
        {"adapter": adapter_state_dict(model), "r": first.r, "alpha": first.alpha},
        path,
    )


def load_adapter(path):
    """Rebuilds a LoRA model from the base weights plus a saved adapter."""
    ckpt = torch.load(path, map_location="cpu")
    model, tokenizer = load_base()
    add_lora(model, r=ckpt["r"], alpha=ckpt["alpha"])
    model.load_state_dict(ckpt["adapter"], strict=False)
    model.eval()
    return model, tokenizer


def milestone_2(epochs=EPOCHS, lr=LEARNING_RATE, seed=SEED, save_path=ADAPTER_PATH):
    """Loads distilgpt2, injects LoRA, and trains on the feedback dataset."""
    torch.manual_seed(seed)
    model, tokenizer = load_base()
    model = add_lora(model)
    losses = train_lora(model, tokenizer, TRAIN_DATA, epochs=epochs, lr=lr)
    if save_path is not None:
        save_adapter(model, save_path)
    return {"model": model, "tokenizer": tokenizer, "losses": losses}

#  SUMMARIZATION FUNCTION FOR THE UI
def summarize(
    model,
    tokenizer,
    feedback_text,
    max_new_tokens=40,
    do_sample=True,
    temperature=0.7,
    top_k=10,
    seed=None,
):
    """Generates a one-line summary from raw feedback text."""
    if not feedback_text or not feedback_text.strip():
        raise ValueError("feedback_text is empty.")
    if do_sample and temperature <= 0:
        raise ValueError("temperature must be positive when sampling.")

    model.eval()
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(PROMPT_TEMPLATE.format(feedback=feedback_text.strip()),
                       return_tensors="pt")
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs.update(temperature=temperature, top_k=top_k)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **kwargs,
        )

    # Decode only the newly generated tokens, and keep the first line.
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip().split("\n")[0].strip()


def milestone_3():
    """Returns the summarize function reference for the Streamlit UI."""
    return summarize

# EVALUATION
def _words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def compute_rouge1(reference, hypothesis):
    """ROUGE-1 F1: unigram overlap with counts, ignoring case and punctuation."""
    ref, hyp = Counter(_words(reference)), Counter(_words(hypothesis))
    overlap = sum((ref & hyp).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(hyp.values())
    recall = overlap / sum(ref.values())
    return round(2 * precision * recall / (precision + recall), 4)


def milestone_4(trained=None):
    """Scores greedy summaries of held-out feedback with ROUGE-1 F1."""
    trained = trained if trained is not None else milestone_2()
    generated, scores = [], []
    for item in EVAL_DATA:
        summary = summarize(
            trained["model"], trained["tokenizer"], item["feedback"], do_sample=False
        )
        generated.append(summary)
        scores.append(compute_rouge1(item["summary"], summary))
    return {
        "rouge1_scores": scores,
        "avg_rouge1": sum(scores) / len(scores),
        "generated": generated,
    }

# ADAPTER MERGING FOR INFERENCE
def merge_lora(model):
    """Merges the scaled B @ A update into the base weights, in place."""
    for block in model.transformer.h:
        layer = block.attn.c_attn
        if isinstance(layer, LoRALinear):
            with torch.no_grad():
                layer.base.weight += (layer.B @ layer.A) * layer.scale
            block.attn.c_attn = layer.base
    return model


def _logits(model, tokenizer, text):
    model.eval()
    ids = tokenizer(PROMPT_TEMPLATE.format(feedback=text), return_tensors="pt")["input_ids"]
    with torch.no_grad():
        return model(input_ids=ids).logits


def milestone_5(trained=None):
    """Merges the adapter into a copy of the model and checks nothing changed."""
    trained = trained if trained is not None else milestone_2()
    model = copy.deepcopy(trained["model"])
    tokenizer = trained["tokenizer"]

    sample = "The assignments were very helpful."
    before_logits = _logits(model, tokenizer, sample)
    before_summary = summarize(model, tokenizer, sample, do_sample=False)

    model = merge_lora(model)

    after_logits = _logits(model, tokenizer, sample)
    after_summary = summarize(model, tokenizer, sample, do_sample=False)
    return {
        "before_merge": before_summary,
        "after_merge": after_summary,
        "max_logit_diff": (before_logits - after_logits).abs().max().item(),
        "model": model,
        "tokenizer": tokenizer,
    }
def baseline_first_sentence(feedback_text):
    """Baseline: just returns the first sentence as the 'summary'."""
    sentences = feedback_text.replace(".", ".\n").split("\n")
    return sentences[0].strip() if sentences else feedback_text