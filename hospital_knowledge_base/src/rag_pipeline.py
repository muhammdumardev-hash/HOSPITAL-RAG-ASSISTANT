"""
rag_pipeline.py
Connects retrieval with the Groq LLM to generate grounded answers
from the hospital knowledge base.
"""

from groq import Groq
import streamlit as st

from src.retriever import retrieve
from src.prompts import SYSTEM_PROMPT, build_user_prompt


def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def generate_answer(query, top_k=5):
    """Retrieve relevant chunks and generate a grounded answer."""

    results = retrieve(query, top_k=top_k)

    if not results:
        return {
            "answer": "I could not find this information in the hospital knowledge base.",
            "sources": [],
        }

    user_prompt = build_user_prompt(query, results)

    client = get_groq_client()

    response = client.chat.completions.create(
        model=st.secrets["GROQ_MODEL"],
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": results,
    }