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
