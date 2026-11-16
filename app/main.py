"""Streamlit entry point: `streamlit run app/main.py`"""
import streamlit as st
from app.core import milestone_2, summarize, compute_rouge1, FEEDBACK_DATA

st.set_page_config(page_title="Student Feedback Summarizer")
st.title("Student Feedback Summarizer")
st.write(
    "Course instructors use this tool to fine-tune a language model with LoRA"
    " on course feedback, generating concise summaries of student comments"
    " for rapid curriculum improvement."
)

# Load model once and cache it
@st.cache_resource
def load_model():
    res = milestone_2()
    return res["model"], res["tokenizer"]


tab1, tab2 = st.tabs(["Summarize Feedback", "Evaluation Results"])

with tab1:
    st.subheader("Enter Student Feedback")
    user_input = st.text_area(
        "Paste raw student feedback below:",
        value="The lectures moved too fast and I couldn't follow the examples.",
        height=120,
    )

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Loading model and generating summary..."):
            model, tokenizer = load_model()
            result = summarize(model, tokenizer, user_input)
        st.success("Summary generated!")
        st.write(f"**Summary:** {result}")

with tab2:
    st.subheader("ROUGE-1 Evaluation on Training Data")
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating summaries..."):
            model, tokenizer = load_model()
            scores = []
            for item in FEEDBACK_DATA[:6]:
                generated = summarize(model, tokenizer, item["feedback"])
                score = compute_rouge1(item["summary"], generated)
                scores.append(
                    {
                        "Feedback": item["feedback"][:60] + "...",
                        "Reference": item["summary"],
                        "Generated": generated[:80],
                        "ROUGE-1 F1": score,
                    }
                )
            st.table(scores)
            avg = sum(s["ROUGE-1 F1"] for s in scores) / len(scores)
            st.metric("Average ROUGE-1 F1", f"{avg:.3f}")