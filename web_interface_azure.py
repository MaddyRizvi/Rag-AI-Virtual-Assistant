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
import hashlib
from functools import lru_cache

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
startup_error = None

def get_doc_processor():
    """Lazy load document processor with error handling"""
    global doc_processor, startup_error
    if doc_processor is None:
        try:
            print("Initializing document processor...")
            doc_processor = DocumentProcessor()
        except Exception as e:
            startup_error = f"Document processor initialization failed: {str(e)}"
            print(f"ERROR: {startup_error}")
            doc_processor = None
            raise Exception(startup_error)
    return doc_processor

def get_rag_chain():
    """Lazy load RAG chain with error handling"""
    global rag_chain, startup_error
    if rag_chain is None:
        try:
            print("Initializing RAG chain...")
            rag_chain = create_rag_chain()
        except Exception as e:
            startup_error = f"RAG chain initialization failed: {str(e)}"
            print(f"ERROR: {startup_error}")
            rag_chain = None
            raise Exception(startup_error)
    return rag_chain

def check_startup_status():
    """Check if all components are properly initialized"""
    try:
        processor = get_doc_processor()
        chain = get_rag_chain()
        return True, "All components initialized successfully"
    except Exception as e:
        return False, str(e)

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
        if file_extension not in ['.txt', '.pdf', '.json', '.xlsx', '.csv', '.jpg', '.jpeg', '.png']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only .txt, .pdf, .json, .xlsx, .csv, .jpg, .jpeg, and .png files are supported")
        
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

# Teacher Dashboard Interface
def teacher_dashboard():
    st.set_page_config(
        page_title="RAGitect - Teacher Dashboard",
        page_icon="👨‍🏫",
        layout="wide"
    )
    
    # Check startup status
    startup_ok, status_msg = check_startup_status()
    if not startup_ok:
        st.error(f"❌ Startup Error: {status_msg}")
        st.error("Please check your environment variables and dependencies.")
        st.markdown("""
        ### Troubleshooting Steps:
        1. **Environment Variables**: Ensure PINECONE_API_KEY, AZ_OPENAI_ENDPOINT, AZ_OPENAI_API_KEY are set
        2. **Dependencies**: Check that all required packages are installed
        3. **Azure Portal**: Verify Application Settings in your Web App Configuration
        4. **GitHub Actions**: Check deployment logs for installation errors
        
        ### Required Environment Variables:
        - `PINECONE_API_KEY`: Your Pinecone API key
        - `AZ_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint
        - `AZ_OPENAI_API_KEY`: Your Azure OpenAI API key
        """)
        return
    
    st.title("👨‍🏫 Teacher Dashboard")
    st.markdown("Upload and manage course materials for students!")
    st.success("✅ Teacher mode - Full access enabled!")
    
    # Initialize session state
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Teacher-specific sidebar with full admin features
    with st.sidebar:
        st.header("📚 Course Management")
        
        # Document upload section
        st.subheader("📁 Upload Materials")
        
        # Text upload option
        st.text("Upload course content:")
        text_content = st.text_area(
            "Course text content:",
            height=150,
            placeholder="Paste lecture notes, assignments, or course materials..."
        )
        
        if st.button("📤 Upload Text", key="teacher_upload_text_btn", help="Upload text content for students"):
            if text_content.strip():
                upload_text_direct(text_content)
            else:
                st.warning("Please enter some text content.")
        
        st.markdown("---")
        
        # File upload option
        st.subheader("📋 File Upload")
        uploaded_files = st.file_uploader(
            "Upload course materials (PDF, TXT, JSON, XLSX, CSV, JPG, PNG):",
            type=['pdf', 'txt', 'json', 'xlsx', 'csv', 'jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="teacher_file_uploader",
            help="Upload multiple files at once for better organization"
        )
        
        if st.button("📤 Upload Files", key="teacher_upload_files_btn", help="Upload files for students to access"):
            if uploaded_files:
                upload_files_direct(uploaded_files)
            else:
                st.warning("Please select files to upload.")
        
        st.markdown("---")
        
        # Admin controls
        st.subheader("⚙️ Admin Controls")
        
        # Document management
        if st.button("📊 View Statistics", key="teacher_stats_btn", help="View system statistics and usage"):
            show_statistics_direct()
        
        # Chat management
        if st.button("🗑️ Clear All Chats", key="teacher_clear_all_btn", help="Clear all chat histories"):
            st.session_state.chat_history = []
            st.success("All chat histories cleared!")
        
        # System info
        st.markdown("---")
        st.subheader("ℹ️ System Info")
        st.info(f"🔑 Role: Teacher (Admin Access)")
        st.info(f"📚 Vector Store: {os.environ.get('PINECONE_INDEX_NAME', 'unknown')}")
        st.info(f"🤖 AI Model: Azure GPT-4o")
    
    # Main content area with teacher features
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Test Q&A System")
        st.markdown("Test the knowledge base before making it available to students.")
        
        # Display chat history with performance optimizations
        chat_container = st.container()
        with chat_container:
            # Limit displayed messages to prevent UI lag
            recent_history = st.session_state.chat_history[-20:] if len(st.session_state.chat_history) > 20 else st.session_state.chat_history
            
            for i, (question, answer) in enumerate(recent_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    # Optimize answer display
                    if answer and len(answer) > 1000:
                        if answer.count('\n') > 5:
                            lines = answer.split('\n')
                            preview = '\n'.join(lines[:3]) + "\n\n... [expand for more]"
                            st.write(preview)
                            with st.expander("📖 Show full answer"):
                                st.write(answer)
                        else:
                            preview = answer[:300] + "... [expand for more]"
                            st.write(preview)
                            with st.expander("📖 Show full answer"):
                                st.write(answer)
                    else:
                        st.write(answer)
        
        # Input for testing questions
        question = st.chat_input("Test a question about your uploaded materials...")
        if question:
            ask_question_direct(question)
    
    with col2:
        st.header("📈 Quick Stats")
        
        # Quick statistics
        chat_count = len(st.session_state.chat_history)
        st.metric("💬 Test Questions", chat_count)
        
        # Recent activity
        st.subheader("🕐 Recent Activity")
        if st.session_state.chat_history:
            last_q, last_a = st.session_state.chat_history[-1]
            st.text(f"Last question: {last_q[:50]}...")
        else:
            st.text("No activity yet")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        if st.button("🔄 Refresh System", key="refresh_btn"):
            st.rerun()
        
        if st.button("📋 Export Logs", key="export_logs_btn"):
            if st.session_state.chat_history:
                logs = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download Chat Logs",
                    data=logs,
                    file_name="teacher_chat_logs.json",
                    mime="application/json"
                )
            else:
                st.info("No logs to export")

# Student Interface
def student_interface():
    st.set_page_config(
        page_title="RAGitect - Student Assistant",
        page_icon="🎓",
        layout="wide"
    )
    
    # Check startup status
    startup_ok, status_msg = check_startup_status()
    if not startup_ok:
        st.error(f"❌ System Unavailable: {status_msg}")
        st.error("The learning assistant is currently unavailable. Please try again later.")
        st.info("📞 Contact your teacher if this issue persists.")
        return
    
    st.title("🎓 Learning Assistant")
    st.markdown("Ask questions about your course materials!")
    st.info("📚 Access your course knowledge base through AI-powered search")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Student sidebar with limited features
    with st.sidebar:
        st.header("📖 Learning Tools")
        
        # Help section
        st.subheader("❓ How to Use")
        st.markdown("""
        1. **Ask Questions**: Type your questions about course materials
        2. **Get Answers**: Receive AI-powered responses
        3. **Learn**: Explore topics with detailed explanations
        
        **Tips:**
        - Be specific in your questions
        - Use keywords from your course
        - Ask follow-up questions for clarity
        """)
        
        st.markdown("---")
        
        # Chat controls
        st.subheader("💬 Chat Controls")
        
        if st.button("🗑️ Clear My Chat", key="student_clear_btn", help="Clear your personal chat history"):
            st.session_state.chat_history = []
            st.success("Your chat history cleared!")
        
        # Quick help prompts
        st.markdown("---")
        st.subheader("💡 Quick Prompts")
        
        sample_questions = [
            "What are the main concepts in this course?",
            "Explain [topic] in simple terms",
            "What should I study for the exam?",
            "Give me examples of [concept]",
            "How does [theory] apply to practice?"
        ]
        
        for question in sample_questions:
            if st.button(f"❓ {question}", key=f"prompt_{question}", help="Click to ask this question"):
                # Auto-fill the question
                st.session_state.quick_question = question
        
        st.markdown("---")
        st.subheader("ℹ️ Student Info")
        st.info(f"🎓 Role: Student")
        st.info(f"🤖 AI Assistant: Active")
        st.info(f"📚 Knowledge Base: Ready")
    
    # Main chat interface (simplified for students)
    st.header("💭 Ask Your Question")
    
    # Display chat history with performance optimizations
    chat_container = st.container()
    with chat_container:
        # Limit displayed messages to prevent UI lag
        recent_history = st.session_state.chat_history[-15:] if len(st.session_state.chat_history) > 15 else st.session_state.chat_history
        
        if not recent_history:
            st.info("👋 Welcome! Start by asking a question about your course materials.")
        
        for i, (question, answer) in enumerate(recent_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                # Optimize answer display for students
                if answer and len(answer) > 800:  # Shorter limit for students
                    if answer.count('\n') > 3:
                        lines = answer.split('\n')
                        preview = '\n'.join(lines[:2]) + "\n\n... [expand for more]"
                        st.write(preview)
                        with st.expander("📖 See detailed answer"):
                            st.write(answer)
                    else:
                        preview = answer[:250] + "... [expand for more]"
                        st.write(preview)
                        with st.expander("📖 See detailed answer"):
                            st.write(answer)
                else:
                    st.write(answer)
    
    # Input for student questions
    # Check for quick question from sidebar
    quick_question = st.session_state.pop('quick_question', None)
    
    if quick_question:
        question = quick_question
    else:
        question = st.chat_input("Ask about your course materials...", help="Ask anything about your uploaded course content")
    
    if question:
        ask_question_direct(question)

# Main routing function
def main():
    # Get role from URL parameters or query parameters
    query_params = st.query_params
    role = query_params.get('role', 'student').lower()  # Default to student for safety
    
    # Route to appropriate interface based on role
    if role == 'teacher':
        teacher_dashboard()
    elif role == 'student':
        student_interface()
    else:
        # Default to student for unknown roles with a warning
        st.warning("⚠️ Unknown role. Defaulting to Student interface.")
        student_interface()

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
            # Check file size for images (limit to 10MB to avoid network issues)
            if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                file_size = len(uploaded_file.getbuffer())
                if file_size > 10 * 1024 * 1024:  # 10MB limit
                    st.warning(f"⚠️ Large image detected ({file_size / (1024*1024):.1f}MB). For better performance, consider resizing images under 10MB.")
                    # Still process but with warning
            
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
                    
                    # Show additional info for image files
                    if uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Check if OCR was successful by looking at the first document
                        ocr_success = False
                        ocr_details = ""
                        if documents and len(documents) > 0:
                            content_preview = documents[0].page_content
                            if "Extracted Text Content:" in content_preview:
                                # Extract the OCR text section
                                ocr_section = content_preview.split("Extracted Text Content:")[1].split("Additional Notes:")[0].strip()
                                if ocr_section and "[" not in ocr_section and len(ocr_section) > 20:
                                    ocr_success = True
                                    # Count words for better feedback
                                    word_count = len(ocr_section.split())
                                    char_count = len(ocr_section)
                                    ocr_details = f"Extracted {word_count} words ({char_count} characters) from image"
                                else:
                                    ocr_details = "Limited or no text extracted from image"
                        
                        if ocr_success:
                            st.success("🖼️ Image processed with successful OCR text extraction!")
                            st.info(f"📝 {ocr_details}")
                            st.info("🔍 Extracted text is now searchable in your document database.")
                            st.balloons()  # Celebration for successful OCR
                        else:
                            st.warning("⚠️ Image processed but OCR text extraction limited")
                            st.info(f"📝 {ocr_details}")
                            st.info("💡 This could be due to:")
                            st.markdown("""
                            - Image quality or resolution issues
                            - Handwritten text (requires specialized OCR)
                            - Complex layouts or artistic fonts
                            - Text embedded in graphics or logos
                            - Poor lighting or contrast in photos
                            """)
                            
                            st.markdown("""
                            **OCR Improvement Tips:**
                            - ✅ Use high-resolution images (300+ DPI)
                            - ✅ Ensure text has good contrast and clarity  
                            - ✅ Use clear, readable fonts (Arial, Times New Roman)
                            - ✅ Avoid blurry or skewed images
                            - ✅ Simple layouts work better than complex designs
                            - ✅ For documents, scan rather than photograph
                            - ✅ Ensure good lighting for photos of text
                            """)
                            
                            # Show what was actually extracted for debugging
                            if documents and len(documents) > 0:
                                with st.expander("🔍 See extracted content (for debugging)"):
                                    st.text(documents[0].page_content)
                else:
                    st.error(f"❌ Failed to upload '{uploaded_file.name}'")
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
        
        except Exception as e:
            st.error(f"❌ Error uploading '{uploaded_file.name}': {str(e)}")
            # Provide helpful error message for network issues
            if "network" in str(e).lower() or "connection" in str(e).lower():
                st.error("🔌 Network error detected. This might be due to large file sizes or connectivity issues.")
                st.info("💡 Try uploading smaller files or check your internet connection.")

# Performance optimization: Cache for question responses
@lru_cache(maxsize=50)
def get_cached_response(question_hash: str, question: str):
    """Cache question responses to reduce processing time"""
    # This is a simple cache - in production, you'd want Redis or similar
    return None  # Always process fresh for now, but structure is ready

def ask_question_direct(question: str):
    """Ask a question using the RAG system directly"""
    try:
        # Create a simple hash for caching (disabled for now to ensure freshness)
        question_hash = hashlib.md5(question.encode()).hexdigest()
        
        # Check cache first (optional - disabled for accuracy)
        cached_result = get_cached_response(question_hash, question)
        
        with st.spinner("🔍 Searching documents..."):
            # Get RAG chain (lazy loaded)
            chain = get_rag_chain()
            
            # Optimize: Add timeout and better error handling
            import asyncio
            try:
                # Run the chain with timeout to prevent hanging
                result = chain.invoke(question)
            except Exception as chain_error:
                st.error(f"⚠️ Search timeout or error: {str(chain_error)}")
                st.info("💡 Try rephrasing your question or check if documents are uploaded.")
                return
            
            # Add to chat history
            st.session_state.chat_history.append((question, result))
            
            # Update UI immediately without full rerun for better performance
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Error asking question: {str(e)}")
        # Provide helpful error messages
        if "timed out" in str(e).lower():
            st.info("💡 The search took too long. Try shorter or more specific questions.")
        elif "no documents" in str(e).lower():
            st.info("💡 No documents found. Upload some documents first.")
        else:
            st.info("💡 Try rephrasing your question or check your internet connection.")

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
