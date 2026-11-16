# Student Feedback Summarizer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/qqcmpzznnm/badge.svg)](https://ask.thili.ai/b/qqcmpzznnm)

This project creates a 'Student Feedback Summarizer' for course instructors. It leverages LoRA fine-tuning to adapt a language model to synthesize student comments into concise summaries. The goal is to help instructors quickly grasp feedback themes and improve curriculum efficiently.

## What you are building

**Goal.** You will build a tool that fine-tunes a language model using LoRA on course feedback, generating concise summaries of student comments. This proves your ability to adapt a pre-trained LLM for a specific summarization task.

**Done when.** The project is complete when the Streamlit UI allows instructors to upload feedback, and the fine-tuned LoRA model generates relevant and concise summaries of student comments, demonstrating the successful application and merging of the LoRA adapter.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a small dataset of synthetic student feedback and summary pairs.**  
      Hint: Focus on structuring your synthetic data with clear feedback and summary pairs.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Apply the `lfllm-lflora` technique to inject and train the LoRA adapter on your chosen LLM.
- [ ] **3. Develop a Streamlit UI to upload feedback and trigger summarization.**  
      Hint: Use Streamlit components to create an intuitive interface for input and displaying results.
- [ ] **4. Evaluate the quality of generated summaries against original feedback.**  
      Hint: Consider both automated metrics and qualitative review to assess summary accuracy and conciseness.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Utilize the `lfllm-lflora` method to merge the trained adapter weights into the base model for efficient inference.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Display the original feedback alongside the generated summaries in the UI for easier comparison.
- Add an option in the UI for instructors to rate the quality of the generated summaries.
- Implement a feature to allow instructors to specify the desired length or detail level for the summaries.
