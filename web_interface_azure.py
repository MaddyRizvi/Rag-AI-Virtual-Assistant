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
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import csv

# Initialize the document processor and chain (lazy loading for better startup)
doc_processor = None
rag_chains = {}  # Cache chains per course
startup_error = None

# --- Course persistence helpers ---
def _courses_file_path() -> str:
    """Return path to courses.json (configurable via COURSES_FILE)."""
    path = os.environ.get("COURSES_FILE")
    if not path:
        data_dir = os.path.join(os.getcwd(), "data")
        try:
            os.makedirs(data_dir, exist_ok=True)
        except Exception:
            # Fallback to CWD if data dir cannot be created
            data_dir = os.getcwd()
        path = os.path.join(data_dir, "courses.json")
    return path

def load_courses_from_disk(default=None) -> list:
    """Load courses from disk, return default if missing or invalid."""
    if default is None:
        default = [
            "general", "math101", "physics101", "chemistry101", "biology101", "history101"
        ]
    path = _courses_file_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return data
        return default
    except Exception:
        return default

def save_courses_to_disk(courses: list) -> bool:
    """Persist courses to disk; return True on success."""
    try:
        path = _courses_file_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(courses, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

# --- Logging helper for analytics ---
def _logs_file_path() -> str:
    return os.path.join(os.getcwd(), "logs.csv")

def log_student_query(role: str, course_id: str, question: str, answer_length: int) -> None:
    """Append a row to logs.csv with timestamp, role, course_id, question, response_length."""
    try:
        path = _logs_file_path()
        file_exists = os.path.exists(path)
        with open(path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "role", "course_id", "question", "response_length"])
            from datetime import datetime
            writer.writerow([
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                role,
                course_id,
                question.replace("\n", " ").strip(),
                int(answer_length),
            ])
    except Exception:
        # Non-fatal: logging should not break the app
        pass

# --- Azure OpenAI helper ---
def get_azure_chat_model() -> AzureChatOpenAI:
    deployment = os.environ.get("AZ_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
    endpoint = os.environ.get("AZ_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZ_OPENAI_API_KEY")
    api_version = os.environ.get("AZ_OPENAI_API_VERSION", "2024-12-01-preview")
    if not endpoint or not api_key:
        raise ValueError("Azure OpenAI credentials missing: set AZ_OPENAI_ENDPOINT and AZ_OPENAI_API_KEY")
    return AzureChatOpenAI(
        temperature=0,
        azure_deployment=deployment,
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

# --- Quiz display helper for Student UI ---
def display_quiz(course_id: str):
    """Render quiz form for the given course_id based on generated CSV."""
    quiz_path = os.path.join(os.getcwd(), "generated_quizzes", f"{course_id}_quiz.csv")
    if not os.path.exists(quiz_path):
        st.info("No quiz found for this course yet.")
        return
    import csv as _csv
    rows = []
    try:
        with open(quiz_path, "r", encoding="utf-8") as f:
            reader = _csv.reader(f)
            data = list(reader)
            if data and len(data[0]) >= 2 and "question" in data[0][0].lower():
                data = data[1:]
            for r in data:
                if len(r) >= 2:
                    rows.append((r[0], r[1]))
    except Exception:
        rows = []

    if not rows:
        st.info("Quiz not available for this course yet.")
        return

    with st.form("take_quiz_form", clear_on_submit=False):
        user_answers = []
        for idx, (q, a) in enumerate(rows[:5]):
            st.write(f"Q{idx+1}: {q}")
            ans = st.text_input(f"Your answer {idx+1}", key=f"quiz_ans_{idx}")
            user_answers.append((ans, a))
        submitted = st.form_submit_button("Submit Quiz")
    if submitted and st.session_state.get('role', 'student') == 'student':
        # Simple marking with exact and partial keyword matches
        total = min(5, len(user_answers))
        total_points = 0.0
        import re
        for ans, actual in user_answers[:total]:
            ans_norm = (ans or "").strip().lower()
            act_norm = (actual or "").strip().lower()
            if not ans_norm:
                continue
            if ans_norm == act_norm or act_norm in ans_norm or ans_norm in act_norm:
                total_points += 1.0
                continue
            tokens_ans = set([t for t in re.split(r"[^a-z0-9]+", ans_norm) if len(t) > 2])
            tokens_act = set([t for t in re.split(r"[^a-z0-9]+", act_norm) if len(t) > 2])
            if tokens_ans.intersection(tokens_act):
                total_points += 0.5
        keyword_percent = (total_points / max(1, total)) * 100.0

        # Optional semantic grading with Azure OpenAI (fallback to keyword)
        final_percent = keyword_percent
        try:
            model = get_azure_chat_model()
            from langchain_core.messages import SystemMessage, HumanMessage
            import json as _json
            items = []
            for idx, (ans, actual) in enumerate(user_answers[:total], start=1):
                items.append({"q": idx, "student_answer": ans or "", "correct_answer": actual or ""})
            msgs = [
                SystemMessage(content=(
                    "Grade each answer from 0-100 based on correctness and completeness. "
                    "Return ONLY a JSON array of numbers (no text)."
                )),
                HumanMessage(content=_json.dumps(items)),
            ]
            resp = model.invoke(msgs)
            content = getattr(resp, "content", str(resp))
            import json as __json
            scores = __json.loads(content)
            if isinstance(scores, list) and scores:
                nums = [float(x) for x in scores[:total]]
                final_percent = sum(nums) / len(nums)
        except Exception:
            pass

        st.success(f"Your score: {final_percent:.0f}%")

        # Append to student_progress.csv (timestamp, student_id, course_id, score)
        try:
            from datetime import datetime
            path = _progress_file_path()
            file_exists = os.path.exists(path)
            import csv as __csv
            with open(path, mode="a", encoding="utf-8", newline="") as f:
                writer = __csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "student_id", "course_id", "score"])  # percent 0-100
                writer.writerow([datetime.utcnow().isoformat(timespec="seconds") + "Z", "student001", course_id, round(final_percent, 2)])
            st.info("Progress saved successfully.")
        except Exception:
            st.warning("Could not save progress.")

        # Recompute and show quick summary metrics
        try:
            dfp = pd.read_csv(_progress_file_path())
            my_df = dfp[(dfp.get('student_id') == 'student001') & (dfp.get('course_id') == course_id)]
            if not my_df.empty:
                avg_percent = float(my_df['score'].astype(float).mean()) if 'score' in my_df.columns else None
                if avg_percent is not None:
                    st.metric("Last Quiz Score", f"{int(round(final_percent))}%")
                    st.metric("Average Score", f"{avg_percent:.1f}")
        except Exception:
            pass

        # Refresh to update analytics elsewhere
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

# --- Student progress helpers ---
def _progress_file_path() -> str:
    """Return path to student_progress.csv in ./data or CWD fallback."""
    data_dir = os.path.join(os.getcwd(), "data")
    try:
        os.makedirs(data_dir, exist_ok=True)
    except Exception:
        data_dir = os.getcwd()
    return os.path.join(data_dir, "student_progress.csv")

def log_student_progress(student_id: str, course_id: str, score: int, total_questions: int) -> None:
    """Append a quiz attempt to student_progress.csv with required schema."""
    try:
        path = _progress_file_path()
        file_exists = os.path.exists(path)
        with open(path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["student_id", "course_id", "score", "total_questions", "timestamp"])
            from datetime import datetime
            writer.writerow([
                (student_id or "anonymous"),
                course_id,
                int(score),
                int(total_questions),
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
            ])
    except Exception:
        pass

def get_student_id() -> str:
    """Resolve student_id from query params or environment; fallback to 'anonymous'."""
    try:
        qp = getattr(st, "query_params", {})
        if qp and qp.get("student_id"):
            return str(qp.get("student_id")).strip()
    except Exception:
        pass
    env_id = os.environ.get("STUDENT_ID")
    if env_id:
        return env_id.strip()
    return "anonymous"

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

def get_rag_chain(course_id: str = "general"):
    """Lazy load RAG chain with course support"""
    global rag_chains, startup_error
    if course_id not in rag_chains:
        try:
            print(f"Initializing RAG chain for course: {course_id}")
            rag_chains[course_id] = create_rag_chain(course_id)
        except Exception as e:
            startup_error = f"RAG chain initialization failed: {str(e)}"
            print(f"ERROR: {startup_error}")
            rag_chains[course_id] = None
            raise Exception(startup_error)
    return rag_chains[course_id]

def check_startup_status():
    """Check if all components are properly initialized"""
    try:
        processor = get_doc_processor()
        chain = get_rag_chain()
        return True, "All components initialized successfully"
    except Exception as e:
        return False, str(e)

# --- Student course assignment helpers ---
def get_assigned_course_id() -> Optional[str]:
    """Determine assigned course_id for the current student session.
    Priority: URL query params (course or course_id) -> env STUDENT_COURSE_ID.
    """
    try:
        qp = getattr(st, "query_params", {})
        # st.query_params behaves like a dict of strings
        if qp:
            if "course_id" in qp and qp.get("course_id"):
                return str(qp.get("course_id")).strip()
            if "course" in qp and qp.get("course"):
                return str(qp.get("course")).strip()
    except Exception:
        pass
    env_course = os.environ.get("STUDENT_COURSE_ID")
    if env_course:
        return env_course.strip()
    return None

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
    if 'current_course' not in st.session_state:
        st.session_state.current_course = "general"
    # Ensure courses list exists before sidebar uses it
    if 'courses' not in st.session_state:
        try:
            st.session_state.courses = load_courses_from_disk()
        except Exception:
            st.session_state.courses = [
                "general", "math101", "physics101", "chemistry101", "biology101", "history101"
            ]
    if 'courses' not in st.session_state:
        st.session_state.courses = load_courses_from_disk()
    if 'courses' not in st.session_state:
        st.session_state.courses = load_courses_from_disk()
    
    # Teacher-specific sidebar with full admin features
    with st.sidebar:
        st.header("📚 Course Management")
        
        # Course selection
        st.subheader("🎯 Course Selection")
        available_courses = st.session_state.courses
        selected_course = st.selectbox(
            "Select Course:",
            options=available_courses,
            index=available_courses.index(st.session_state.current_course) if st.session_state.current_course in available_courses else 0,
            key="course_selector"
        )
        st.session_state.current_course = selected_course
        # Persist courses list on each render (idempotent)
        save_courses_to_disk(st.session_state.courses)
        st.info(f"📍 Current course: `{selected_course}`")
        
        # Create a new course
        st.markdown("---")
        st.text("Create a new course:")
        new_course = st.text_input(
            "New course ID",
            placeholder="e.g., comp_sci_2025",
            key="new_course_id_input"
        )
        if st.button("➕ Add Course", key="add_course_btn"):
            sanitized = (new_course or "").strip()
            import re
            if not sanitized:
                st.warning("Please enter a course ID.")
            elif not re.match(r"^[A-Za-z0-9_-]{2,64}$", sanitized):
                st.error("Course ID must be 2-64 chars: letters, numbers, '-' or '_'.")
            else:
                if sanitized not in st.session_state.courses:
                    st.session_state.courses.append(sanitized)
                    st.success(f"Added course '{sanitized}'.")
                else:
                    st.info("Course already exists. Selected it.")
                st.session_state.current_course = sanitized

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

        # Quiz generation section
        st.markdown("---")
        st.subheader("🧠 Generate Quiz")
        st.caption("Generate 5 short Q&A as CSV from current course materials")
        if st.button("📝 Generate Quiz", key="generate_quiz_btn"):
            course_id = st.session_state.get("current_course", "general")
            try:
                with st.spinner("Generating quiz..."):
                    processor = get_doc_processor()
                    results = processor.search_documents(query=course_id, course_id=course_id, k=5) or []
                    if not results:
                        st.warning("No documents found for this course.")
                    else:
                        context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in results])
                        prompt_msgs = [
                            SystemMessage(content=(
                                "You create concise quizzes in CSV format. "
                                "Output exactly 5 rows with header 'Question,Answer'. "
                                "Each row contains a short question and its correct answer. "
                                "Do not include explanations, code fences, or extra commentary."
                            )),
                            HumanMessage(content=(
                                f"Course ID: {course_id}\n\n"
                                "Course material excerpts:\n" + context_text + "\n\n"
                                "Generate 5 short quiz questions (with correct answers) as CSV with header 'Question,Answer'."
                            )),
                        ]
                        model = get_azure_chat_model()
                        resp = model.invoke(prompt_msgs)
                        quiz_text = getattr(resp, "content", str(resp))

                        out_dir = os.path.join(os.getcwd(), "generated_quizzes")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"{course_id}_quiz.csv")
                        with open(out_path, "w", encoding="utf-8") as f:
                            f.write(quiz_text)

                        st.success("Quiz generated successfully.")
                        with st.expander("View Generated Quiz (CSV)", expanded=True):
                            st.text(quiz_text)
                        st.download_button(
                            label="Download Quiz (.csv)",
                            data=quiz_text,
                            file_name=f"{course_id}_quiz.csv",
                            mime="text/csv",
                            key="download_quiz_btn",
                        )
            except Exception as e:
                st.error(f"❌ Failed to generate quiz: {str(e)}")

        st.markdown("---")

        # Learning Analytics
        st.subheader("📊 Learning Analytics")
        logs_path = _logs_file_path()
        if not os.path.exists(logs_path):
            st.info("No analytics data available")
        else:
            try:
                df = pd.read_csv(logs_path)
                if df.empty or 'role' not in df.columns:
                    st.info("No analytics data available")
                else:
                    sdf = df[df['role'] == 'student'].copy()
                    total_q = int(len(sdf))
                    st.metric(label="Total Student Queries", value=total_q)
                    if total_q > 0:
                        per_course = sdf.groupby('course_id').size().rename('queries')
                        st.caption("Queries per course")
                        st.bar_chart(per_course)
                        # Average response length per course (supports legacy column name)
                        resp_col = 'response_length' if 'response_length' in sdf.columns else ('answer_length' if 'answer_length' in sdf.columns else None)
                        if resp_col:
                            avg_len = sdf.groupby('course_id')[resp_col].mean()
                            st.caption("Average response length per course")
                            st.bar_chart(avg_len)
                    else:
                        st.info("No student queries logged yet")
            except Exception as e:
                st.warning(f"Analytics not available: {str(e)}")

        # Student progress analytics (quiz marks)
        try:
            sp_path = _progress_file_path()
            if os.path.exists(sp_path):
                dfp = pd.read_csv(sp_path)
                if not dfp.empty:
                    # Overall average quiz score
                    if 'score' in dfp.columns and 'total_questions' in dfp.columns:
                        overall_avg = float((dfp['score'] / dfp['total_questions']).mean() * 100.0)
                        st.metric("Avg Quiz Score", f"{overall_avg:.2f}")
                        st.caption("Average quiz score per course")
                        st.bar_chart((dfp.assign(percent=(dfp['score']/dfp['total_questions']*100.0))).groupby('course_id')['percent'].mean())
                    elif 'quiz_score' in dfp.columns:
                        st.metric("Avg Quiz Score", f"{float(dfp['quiz_score'].mean()):.2f}")
                        st.caption("Average quiz score per course")
                        st.bar_chart(dfp.groupby('course_id')['quiz_score'].mean())

                    # Selected course table and metrics
                    selected = selected_course if 'selected_course' in locals() else st.session_state.get('current_course','general')
                    course_df = dfp[dfp.get('course_id') == selected]
                    if not course_df.empty:
                        if 'score' in course_df.columns and 'total_questions' in course_df.columns:
                            perc = (course_df['score']/course_df['total_questions'])*100.0
                            avg_score = float(perc.mean())
                            hi = float(perc.max())
                            lo = float(perc.min())
                        elif 'quiz_score' in course_df.columns:
                            avg_score = float(course_df['quiz_score'].mean())
                            hi = float(course_df['quiz_score'].max())
                            lo = float(course_df['quiz_score'].min())
                        else:
                            avg_score = 0.0; hi=0.0; lo=0.0
                        students_attempted = course_df['student_id'].nunique() if 'student_id' in course_df.columns else len(course_df)
                        last_ts = str(course_df['timestamp'].max()) if 'timestamp' in course_df.columns else "N/A"
                        st.metric("Avg Score (selected)", f"{avg_score:.1f}")
                        st.metric("Students Attempted", students_attempted)
                        st.metric("Last Submission", last_ts)
                        st.caption(f"Highest: {hi:.1f} • Lowest: {lo:.1f}")
                        st.dataframe(course_df, use_container_width=True)
        except Exception:
            pass

        # Manual refresh option
        if st.button("🔄 Refresh Analytics", key="refresh_analytics_btn"):
            try:
                st.rerun()
            except Exception:
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

        st.markdown("---")

        # Admin controls
        st.subheader("⚙️ Admin Controls")
        
        # Document management
        if st.button("📊 View Statistics", key="teacher_stats_btn", help="View system statistics and usage"):
            show_statistics_direct()
        
        # Chat management
        if st.button(" Clear All Chats", key="teacher_clear_all_btn", help="Clear all chat histories"):
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
        st.header("📈 Learning Analytics")
        
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
    if 'current_course' not in st.session_state:
        st.session_state.current_course = "general"
    # Ensure courses list exists before sidebar uses it
    if 'courses' not in st.session_state:
        try:
            st.session_state.courses = load_courses_from_disk()
        except Exception:
            st.session_state.courses = [
                "general", "math101", "physics101", "chemistry101", "biology101", "history101"
            ]
    
    # Student sidebar with limited features
    with st.sidebar:
        st.header("📖 Learning Tools")
        
        # Course selection
        st.subheader("🎯 Course Selection")
        available_courses = st.session_state.courses
        selected_course = st.selectbox(
            "Select Course:",
            options=available_courses,
            index=available_courses.index(st.session_state.current_course) if st.session_state.current_course in available_courses else 0,
            key="student_course_selector"
        )
        st.session_state.current_course = selected_course
        st.info(f"📍 Current course: `{selected_course}`")
        
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
        # Older Streamlit versions don't support `help` on chat_input
        # Keep compatibility by removing the help kwarg
        question = st.chat_input("Ask about your course materials...")
    
    if question:
        ask_question_direct(question)

    # Take Quiz section (uses generated CSV quiz for current course)
    st.markdown("---")
    st.subheader("📝 Take Quiz")
    st.caption("Answer 5 questions from your course quiz (if available)")
    course_id = st.session_state.get('assigned_course', st.session_state.current_course)
    if st.button("🎯 Take Quiz", key="student_take_quiz"):
        display_quiz(course_id)
    quiz_path = os.path.join(os.getcwd(), "generated_quizzes", f"{course_id}_quiz.csv")
    if False and os.path.exists(quiz_path):
        import csv as _csv
        rows = []
        try:
            with open(quiz_path, "r", encoding="utf-8") as f:
                reader = _csv.reader(f)
                data = list(reader)
                if data and len(data[0]) >= 2 and "question" in data[0][0].lower():
                    data = data[1:]
                for r in data:
                    if len(r) >= 2:
                        rows.append((r[0], r[1]))
        except Exception:
            rows = []

        if not rows:
            st.info("Quiz not available for this course yet.")
        else:
            with st.form("take_quiz_form", clear_on_submit=False):
                user_answers = []
                for idx, (q, a) in enumerate(rows[:5]):
                    st.write(f"Q{idx+1}: {q}")
                    ans = st.text_input(f"Your answer {idx+1}", key=f"quiz_ans_{idx}")
                    user_answers.append((ans, a))
                submitted = st.form_submit_button("Submit Quiz")
            if submitted:
                correct = 0
                total = min(5, len(user_answers))
                for ans, actual in user_answers[:total]:
                    ans_norm = (ans or "").strip().lower()
                    actual_norm = (actual or "").strip().lower()
                    if ans_norm and (actual_norm in ans_norm or ans_norm in actual_norm):
                        correct += 1
                score = int(round((correct / max(1, total)) * 100))
                st.success(f"Your score: {score}% ({correct}/{total})")
                if score > 80:
                    st.info("Excellent!")
                elif score >= 60:
                    st.info("Good job, keep improving!")
                else:
                    st.info("Needs review — revisit your notes!")

                try:
                    sid = get_student_id()
                    log_student_progress(sid, course_id, score)
                except Exception:
                    pass
    else:
        pass
    # My Progress section (consolidated)
    st.markdown("---")
    st.subheader("My Progress")
    progress_path = _progress_file_path()
    if not os.path.exists(progress_path):
        st.info("No progress data available yet.")
    else:
        try:
            dfp = pd.read_csv(progress_path)
            sid = get_student_id()
            cur = dfp[(dfp.get('student_id') == sid) & (dfp.get('course_id') == course_id)] if not dfp.empty else pd.DataFrame()
            if cur.empty:
                st.info("No progress data available yet.")
            else:
                if 'score' in cur.columns:
                    avg_percent = float(pd.to_numeric(cur['score'], errors='coerce').dropna().mean())
                elif 'quiz_score' in cur.columns:
                    avg_percent = float(pd.to_numeric(cur['quiz_score'], errors='coerce').dropna().mean())
                else:
                    avg_percent = 0.0
                latest_ts = str(cur['timestamp'].max()) if 'timestamp' in cur.columns else "N/A"
                st.metric("Average Score", f"{avg_percent:.1f}%")
                st.caption(f"Latest quiz date: {latest_ts}")
                st.progress(min(max(int(round(avg_percent)), 0), 100) / 100)
                if avg_percent >= 80:
                    st.success("Excellent progress! Keep it up.")
                elif avg_percent >= 60:
                    st.info("Good job! Review topics for higher scores.")
                else:
                    st.warning("Needs review. Try another quiz.")
                if 'timestamp' in cur.columns:
                    try:
                        hist = cur.copy()
                        hist['timestamp'] = pd.to_datetime(hist['timestamp'], errors='coerce')
                        hist = hist.dropna(subset=['timestamp']).sort_values('timestamp')
                        series = pd.to_numeric(hist['score'] if 'score' in hist.columns else hist['quiz_score'], errors='coerce')
                        st.caption("Quiz score history")
                        st.line_chart(series.set_axis(hist['timestamp']))
                    except Exception:
                        pass
        except Exception as e:
            st.warning(f"Unable to load progress: {str(e)}")
                        pass
        except Exception as e:
            st.warning(f"Unable to load progress: {str(e)}")

# Main routing function
def main():
    # Get role from URL parameters or query parameters
    query_params = st.query_params
    role = query_params.get('role', 'student').lower()  # Default to student for safety
    # Persist role in session for downstream logic
    st.session_state.role = role
    
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
            
            # Process the text directly with course_id
            documents = processor.process_text(text_content, {"source": "web_interface", "type": "text", "course_id": st.session_state.current_course})
            
            # Add to vector store with course namespace
            success = processor.add_documents_to_vectorstore(documents, st.session_state.current_course)
            
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
                
                # Process the file directly with course_id
                documents = processor.process_file(tmp_file_path, {"original_filename": uploaded_file.name}, st.session_state.current_course)
                
                # Add to vector store with course namespace
                success = processor.add_documents_to_vectorstore(documents, st.session_state.current_course)
                
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
        
        # Enforce assigned course from URL/env if present
        try:
            assigned = get_assigned_course_id()
            if assigned:
                st.session_state.assigned_course = assigned
        except Exception:
            pass
        
        with st.spinner("🔍 Searching documents..."):
            # Get RAG chain (lazy loaded) with enforced course_id
            course = st.session_state.get('assigned_course', st.session_state.current_course)
            st.session_state.current_course = course
            chain = get_rag_chain(course)
            
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

            # Learning analytics logging (only for student role)
            try:
                role = st.session_state.get('role', 'student')
                if role == 'student':
                    ans_text = result if isinstance(result, str) else str(result)
                    log_student_query(role, course, question, len(ans_text))
            except Exception:
                pass
            
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



