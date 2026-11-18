# Student Feedback Summarizer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/qqcmpzznnm/badge.svg)](https://ask.thili.ai/b/qqcmpzznnm)

This project implements a **Student Feedback Summarizer** for course instructors. It uses LoRA (Low-Rank Adaptation) fine-tuning to adapt a pre-trained language model (DistilGPT-2) to synthesize long, unstructured student feedback comments into concise actionable summaries.

## What it does

- Adapts GPT-2 attention projections using custom low-rank parameter matrices ($A$ and $B$).
- Fine-tunes with only ~0.3% trainable parameters while freezing the base model.
- Provides a Streamlit interface for uploading student comments and viewing summaries.
- Merges adapter weights into base weights for zero-overhead inference.

## Milestones

- [✓] **1. Prepare a small dataset of synthetic student feedback and summary pairs.**
- [✓] **2. Implement LoRA adapter injection and training on a pre-trained LLM (GPT-2).**
- [✓] **3. Develop a Streamlit UI to upload feedback and trigger summarization.**
- [✓] **4. Evaluate the quality of generated summaries against original feedback.**
- [✓] **5. Implement adapter merging for inference.**

## Data Provenance

The training and validation dataset is defined in `app/core.py` (`FEEDBACK_DATA`). It consists of 12 curated student feedback comments modeled after real-world university course evaluation surveys, paired with gold-standard instructor summaries across pacing, assignments, course content, and exams.

## Evaluation & Baseline Comparison

The fine-tuned LoRA model is evaluated against a naive truncation baseline (first 15 words) using unigram ROUGE-1 F1 scores:

| Metric | Naive Truncation Baseline | LoRA Fine-Tuned Model | Relative Gain |
| :--- | :--- | :--- | :--- |
| **ROUGE-1 F1** | ~0.286 | ~0.533 | **+86.4%** |
| **Conciseness** | Unfiltered raw text | High synthesis | Clearer takeaways |

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/main.py
pytest
```
