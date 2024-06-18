# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import textwrap
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai

# Configure API key and model
genai.configure(api_key='AIzaSyCUvxW0CQ7auPgWb9hwiZonYNy2kuiF63A')  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the sentence transformer model
embedding_model = SentenceTransformer('msmarco-distilbert-base-tas-b')

# Function to load chat history for a given case ID
def load_case_env(case_id):
    try:
        with open(f"{case_id}_history.pkl", "rb") as f:
            history = pickle.load(f)
        chat = model.start_chat(history=history)
    except FileNotFoundError:
        st.error(f"No history found for case {case_id}.")
        return None
    return chat

# Function to close case environment
def close_case_env(case_id):
    if f"chat_{case_id}" in st.session_state:
        del st.session_state[f"chat_{case_id}"]

# Function to send a query to the chat model
def send_case_query(chat, query):
    response = chat.send_message(query)
    return response.text

# Function to query documents
def query_documents(query, documents, model, top_k=3):
    # Encode the query
    query_embedding = model.encode(query)

    # Convert embeddings to tensor
    embeddings_list = [doc['embedding'] for doc in documents]
    document_embeddings = torch.tensor(np.array(embeddings_list))

    # Calculate cosine similarities
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)

    # Get top_k most similar documents
    top_results = cos_scores.topk(k=top_k, largest=True)

    return [(documents[idx.item()]['id'], documents[idx.item()]['content'], cos_scores[0][idx].item()) for idx in top_results[1][0]]

# Load documents from pickle file
pickle_file = 'documents.pkl'  # Ensure this file is in the same directory or adjust the path
with open(pickle_file, 'rb') as f:
    documents = pickle.load(f)

# Streamlit app layout
st.set_page_config(page_title="Legal Case Query System", page_icon="⚖️")
st.title("⚖️ Legal Case Query System")

# Initialize session state variables if not already defined
if 'chat' not in st.session_state:
    st.session_state.chat = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'case_id' not in st.session_state:
    st.session_state.case_id = None
if 'documents' not in st.session_state:
    st.session_state.documents = documents
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# Case selection section
st.markdown("### 🔍 Search Cases")

# Form to submit search query
with st.form(key='search_form', clear_on_submit=True):
    search_query = st.text_input("Enter Your Query", value=st.session_state.search_query)
    search_submit_button = st.form_submit_button(label="Search 🔍")

if search_submit_button and search_query:
    with st.spinner("Searching Cases..."):
        st.session_state.search_results = query_documents(search_query, st.session_state.documents, embedding_model)
        st.session_state.search_query = search_query
        st.session_state.case_id = None
        st.session_state.chat = None
        st.session_state.history = []

# Display search query
if st.session_state.search_query:
    st.markdown(f"### 📄 Search Results for: *{st.session_state.search_query}*")

# Display search results
if st.session_state.search_results:
    for i, (doc_id, content, score) in enumerate(st.session_state.search_results):
        if st.button(f"Select Case {doc_id} 📄", key=doc_id):
            if st.session_state.case_id:
                close_case_env(st.session_state.case_id)
            st.session_state.case_id = doc_id
            st.session_state.selected_content = content
            st.session_state.chat = load_case_env(doc_id)
            st.session_state.history = []
            if st.session_state.chat:
                st.success(f"Case {doc_id} loaded successfully and Ready to Chat! 💬")
            else:
                st.error(f"Failed to load Case {doc_id} ❌")

# Display selected case summary and chat option
if st.session_state.case_id:
    st.markdown("### 📜 Selected Case Summary:")
    st.write(st.session_state.selected_content)

    # Chat history and query section
    if st.session_state.chat:
        st.markdown("---")
        for q, r in st.session_state.history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Response:** {r}")
            st.markdown("---")

        # Form to submit queries
        with st.form(key='query_form', clear_on_submit=True):
            query = st.text_input("Enter Your Query")
            submit_button = st.form_submit_button(label="Send Query 🚀")

        # Handle query submission
        if submit_button and query:
            with st.spinner("Looking Through..."):
                response = send_case_query(st.session_state.chat, query)
                st.session_state.history.append((query, response))
                st.rerun()
else:
    st.warning("Please enter a query to search for cases.")
