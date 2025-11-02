#!/usr/bin/env python3
"""
Self-contained web interface for Azure deployment
Includes both Streamlit UI and optional API functionality in one file
"""

import streamlit as st
import requests
import json
import tempfile
import os
from typing import Optional, List
import threading
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the necessary components from the app
from app.document_processor import DocumentProcessor
from app.chain import create_rag_chain
from langchain_core.documents import Document
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio

# Initialize the document processor and chain (lazy loading for better startup)
doc_processor = None
rag_chain = None

def get_doc_processor():
    """Lazy load document processor"""
    global doc_processor
    if doc_processor is None:
        print("Initializing document processor...")
        doc_processor = DocumentProcessor()
    return doc_processor

def get_rag_chain():
    """Lazy load RAG chain"""
    global rag_chain
    if rag_chain is None:
        print("Initializing RAG chain...")
        rag_chain = create_rag_chain()
    return rag_chain

# Create FastAPI app for the backend (optional)
api_app = FastAPI(title="Document RAG API", description="Upload documents and ask questions about them")

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class TextUploadRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None

class DocumentResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    chunks_processed: Optional[int] = None

class SearchResponse(BaseModel):
    success: bool
    results: List[dict]
    count: int

# API Endpoints
@api_app.get("/")
async def redirect_root_to_docs():
    """Redirect root to API documentation"""
    return RedirectResponse("/docs")

@api_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Document RAG API"}

@api_app.post("/upload/text", response_model=DocumentResponse)
async def upload_text(request: TextUploadRequest):
    """Upload raw text content"""
    try:
        # Get document processor (lazy loaded)
        processor = get_doc_processor()
        
        # Process the text
        documents = processor.process_text(request.text, request.metadata or {})
        
        # Add to vector store
        success = processor.add_documents_to_vectorstore(documents)
        
        if success:
            doc_id = documents[0].metadata.get("doc_id") if documents else None
            return DocumentResponse(
                success=True,
                message="Text uploaded and processed successfully",
                document_id=doc_id,
                chunks_processed=len(documents)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to add documents to vector store")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@api_app.post("/upload/file", response_model=DocumentResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file (PDF, TXT, JSON, XLSX, or CSV)"""
    try:
        # Check file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.txt', '.pdf', '.json', '.xlsx', '.csv']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only .txt, .pdf, .json, .xlsx, and .csv files are supported")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name
        
        try:
            # Get document processor (lazy loaded)
            processor = get_doc_processor()
            
            # Process the file
            documents = processor.process_file(temp_file_path, {"original_filename": file.filename})
            
            # Add to vector store
            success = processor.add_documents_to_vectorstore(documents)
            
            if success:
                doc_id = documents[0].metadata.get("doc_id") if documents else None
                return DocumentResponse(
                    success=True,
                    message=f"File '{file.filename}' uploaded and processed successfully",
                    document_id=doc_id,
                    chunks_processed=len(documents)
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to add documents to vector store")
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@api_app.post("/rag/invoke")
async def rag_invoke(request: dict):
    """Invoke the RAG chain"""
    try:
        input_text = request.get("input", "")
        if not input_text:
            raise HTTPException(status_code=400, detail="Input is required")
        
        # Get RAG chain (lazy loaded)
        chain = get_rag_chain()
        
        # Invoke the chain
        result = chain.invoke(input_text)
        
        return {"output": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invoking RAG chain: {str(e)}")

@api_app.get("/documents/stats")
async def get_document_stats():
    """Get statistics about stored documents"""
    try:
        return {
            "total_documents": "Unknown (implement Pinecone stats retrieval)",
            "vector_store_status": "active",
            "index_name": os.environ.get("PINECONE_INDEX_NAME", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Function to run the API server in a separate thread
def run_api_server():
    """Run the API server in a separate thread"""
    try:
        uvicorn.run(api_app, host="127.0.0.1", port=8001, log_level="error")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port 8001 is already in use. Trying port 8002...")
            try:
                uvicorn.run(api_app, host="127.0.0.1", port=8002, log_level="error")
            except OSError as e2:
                print(f"Port 8002 is also in use. API server will not be available.")
                print("Streamlit interface will continue to work normally.")
        else:
            raise e

# Streamlit Interface
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
                upload_text_direct(text_content)
            else:
                st.warning("Please enter some text content.")
        
        st.markdown("---")
        
        # File upload option
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose PDF, TXT, JSON, XLSX, or CSV files:",
            type=['pdf', 'txt', 'json', 'xlsx', 'csv'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if st.button("Upload Files", key="upload_files_btn"):
            if uploaded_files:
                upload_files_direct(uploaded_files)
            else:
                st.warning("Please select files to upload.")
        
        st.markdown("---")
        
        # Document management
        st.subheader("📊 Document Stats")
        if st.button("View Statistics", key="stats_btn"):
            show_statistics_direct()
        
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
        ask_question_direct(question)

def upload_text_direct(text_content: str):
    """Upload text content directly"""
    try:
        with st.spinner("Processing text..."):
            # Get document processor (lazy loaded)
            processor = get_doc_processor()
            
            # Process the text directly
            documents = processor.process_text(text_content, {"source": "web_interface", "type": "text"})
            
            # Add to vector store
            success = processor.add_documents_to_vectorstore(documents)
            
            if success:
                doc_id = documents[0].metadata.get("doc_id") if documents else None
                st.success(f"✅ Text uploaded successfully!")
                st.info(f"Document ID: {doc_id}")
                st.info(f"Chunks processed: {len(documents)}")
            else:
                st.error("❌ Failed to upload text")
    
    except Exception as e:
        st.error(f"❌ Error uploading text: {str(e)}")

def upload_files_direct(uploaded_files):
    """Upload files directly"""
    for uploaded_file in uploaded_files:
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name
                
                # Get document processor (lazy loaded)
                processor = get_doc_processor()
                
                # Process the file directly
                documents = processor.process_file(tmp_file_path, {"original_filename": uploaded_file.name})
                
                # Add to vector store
                success = processor.add_documents_to_vectorstore(documents)
                
                if success:
                    doc_id = documents[0].metadata.get("doc_id") if documents else None
                    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
                    st.info(f"Document ID: {doc_id}")
                    st.info(f"Chunks processed: {len(documents)}")
                else:
                    st.error(f"❌ Failed to upload '{uploaded_file.name}'")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        except Exception as e:
            st.error(f"❌ Error uploading '{uploaded_file.name}': {str(e)}")

def ask_question_direct(question: str):
    """Ask a question using the RAG system directly"""
    try:
        with st.spinner("Thinking..."):
            # Get RAG chain (lazy loaded)
            chain = get_rag_chain()
            
            # Invoke the chain directly
            result = chain.invoke(question)
            
            # Add to chat history
            st.session_state.chat_history.append((question, result))
            
            # Rerun to display the new message
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error asking question: {str(e)}")

def show_statistics_direct():
    """Display document statistics directly"""
    try:
        with st.spinner("Fetching statistics..."):
            stats = {
                "total_documents": "Unknown (implement Pinecone stats retrieval)",
                "vector_store_status": "active",
                "index_name": os.environ.get("PINECONE_INDEX_NAME", "unknown")
            }
            st.json(stats)
    
    except Exception as e:
        st.error(f"❌ Error fetching statistics: {str(e)}")

if __name__ == "__main__":
    # Check if we should start the API server
    # For local testing, we can skip the API server since Streamlit works directly
    start_api = os.environ.get("START_API_SERVER", "false").lower() == "true"
    
    if start_api:
        print("Starting API server in background...")
        # Start the API server in a background thread
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()
        # Give the API server a moment to start
        time.sleep(2)
    else:
        print("API server disabled. Streamlit interface will work directly.")
    
    # Run the Streamlit interface
    main()
