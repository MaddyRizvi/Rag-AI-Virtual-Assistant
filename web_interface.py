#!/usr/bin/env python3
"""
Simple web interface for document upload and Q&A
"""

import streamlit as st
import requests
import json
import tempfile
import os
from typing import Optional

# API Configuration
API_BASE_URL = "http://127.0.0.1:8001"

def main():
    st.set_page_config(
        page_title="Document Q&A System",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Q&A System")
    st.markdown("Upload documents and ask questions about them!")
    
    # Initialize session state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # Text upload option
        st.subheader("Upload Text")
        text_content = st.text_area(
            "Or paste text content here:",
            height=150,
            placeholder="Paste your document text here..."
        )
        
        if st.button("Upload Text", key="upload_text_btn"):
            if text_content.strip():
                upload_text(text_content)
            else:
                st.warning("Please enter some text content.")
        
        st.markdown("---")
        
        # File upload option
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose PDF or TXT files:",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if st.button("Upload Files", key="upload_files_btn"):
            if uploaded_files:
                upload_files(uploaded_files)
            else:
                st.warning("Please select files to upload.")
        
        st.markdown("---")
        
        # Document management
        st.subheader("📊 Document Stats")
        if st.button("View Statistics", key="stats_btn"):
            show_statistics()
        
        st.markdown("---")
        
        # Clear chat
        if st.button("Clear Chat", key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
    
    # Main content area for Q&A
    st.header("💬 Ask Questions")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
    
    # Input for new question
    question = st.chat_input("Ask a question about your uploaded documents...")
    
    if question:
        ask_question(question)

def upload_text(text_content: str):
    """Upload text content to the API"""
    try:
        with st.spinner("Processing text..."):
            response = requests.post(
                f"{API_BASE_URL}/upload/text",
                json={
                    "text": text_content,
                    "metadata": {"source": "web_interface", "type": "text"}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Text uploaded successfully!")
                st.info(f"Document ID: {result.get('document_id')}")
                st.info(f"Chunks processed: {result.get('chunks_processed')}")
            else:
                st.error(f"❌ Failed to upload text: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error uploading text: {str(e)}")

def upload_files(uploaded_files):
    """Upload files to the API"""
    for uploaded_file in uploaded_files:
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name
                
                # Upload file via API
                with open(tmp_file_path, 'rb') as f:
                    files = {"file": (uploaded_file.name, f, uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/upload/file", files=files)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                    st.info(f"Document ID: {result.get('document_id')}")
                    st.info(f"Chunks processed: {result.get('chunks_processed')}")
                else:
                    st.error(f"❌ Failed to upload '{uploaded_file.name}': {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error uploading '{uploaded_file.name}': {str(e)}")

def ask_question(question: str):
    """Ask a question using the RAG system"""
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_BASE_URL}/rag/invoke",
                json={"input": question}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('output', 'No answer received.')
                
                # Add to chat history
                st.session_state.chat_history.append((question, answer))
                
                # Rerun to display the new message
                st.rerun()
            else:
                st.error(f"❌ Failed to get answer: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error asking question: {str(e)}")

def show_statistics():
    """Display document statistics"""
    try:
        with st.spinner("Fetching statistics..."):
            response = requests.get(f"{API_BASE_URL}/documents/stats")
            
            if response.status_code == 200:
                stats = response.json()
                st.json(stats)
            else:
                st.error(f"❌ Failed to fetch statistics: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error fetching statistics: {str(e)}")

if __name__ == "__main__":
    # Check if API server is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            main()
        else:
            st.error("❌ API server is not responding properly. Please make sure the server is running.")
    except requests.exceptions.RequestException:
        st.error(f"""
        ❌ Cannot connect to API server at {API_BASE_URL}
        
        Please start the API server first by running:
        ```bash
        python -m uvicorn app.server:app --host 127.0.0.1 --port 8001
        ```
        
        Then refresh this page.
        """)
