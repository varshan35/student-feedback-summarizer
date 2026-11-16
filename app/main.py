"""Streamlit entry point: `streamlit run app/main.py`"""
import streamlit as st
from app.core import build_model, summarize, compute_rouge1, baseline_first_sentence, FEEDBACK_DATA

st.set_page_config(page_title="Student Feedback Summarizer")
st.title("Student Feedback Summarizer")
st.write(
    "Course instructors use this tool to fine-tune a language model with LoRA"
    " on course feedback, generating concise summaries of student comments"
    " for rapid curriculum improvement."
)

@st.cache_resource
def load_model():
    res = build_model()
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
    st.subheader("LoRA vs Baseline Comparison")
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            model, tokenizer = load_model()
            rows = []
            for item in FEEDBACK_DATA[:6]:
                lora_out = summarize(model, tokenizer, item["feedback"])
                base_out = baseline_first_sentence(item["feedback"])
                rows.append({
                    "Feedback": item["feedback"][:50] + "...",
                    "Baseline": base_out[:50],
                    "LoRA": lora_out[:50],
                    "Baseline ROUGE": compute_rouge1(item["summary"], base_out),
                    "LoRA ROUGE": compute_rouge1(item["summary"], lora_out),
                })
            st.table(rows)