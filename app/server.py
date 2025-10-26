from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from app.chain import chain as document_rag_chain, doc_processor
from typing import List, Optional
import tempfile
import os
import shutil
from pydantic import BaseModel

app = FastAPI(title="Document RAG API", description="Upload documents and ask questions about them")

# Add CORS middleware
app.add_middleware(
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

@app.get("/")
async def redirect_root_to_docs():
    """Redirect root to API documentation"""
    return RedirectResponse("/docs")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Document RAG API"}

@app.post("/upload/text", response_model=DocumentResponse)
async def upload_text(request: TextUploadRequest):
    """Upload raw text content"""
    try:
        # Process the text
        documents = doc_processor.process_text(request.text, request.metadata or {})
        
        # Add to vector store
        success = doc_processor.add_documents_to_vectorstore(documents)
        
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

@app.post("/upload/file", response_model=DocumentResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a file (PDF or TXT)"""
    try:
        # Check file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.txt', '.pdf']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only .txt and .pdf files are supported")
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Process the file
            documents = doc_processor.process_file(temp_file_path, {"original_filename": file.filename})
            
            # Add to vector store
            success = doc_processor.add_documents_to_vectorstore(documents)
            
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

@app.post("/upload/files", response_model=List[DocumentResponse])
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    results = []
    
    for file in files:
        try:
            result = await upload_file(file)
            results.append(result)
        except Exception as e:
            results.append(DocumentResponse(
                success=False,
                message=f"Error processing '{file.filename}': {str(e)}"
            ))
    
    return results

@app.get("/search", response_model=SearchResponse)
async def search_documents(query: str, k: int = 5):
    """Search for relevant documents"""
    try:
        if not query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        results = doc_processor.search_documents(query, k)
        
        # Convert results to dict format
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return SearchResponse(
            success=True,
            results=formatted_results,
            count=len(formatted_results)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the vector store"""
    try:
        success = doc_processor.clear_documents(doc_id)
        
        if success:
            return {"success": True, "message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/documents/stats")
async def get_document_stats():
    """Get statistics about stored documents"""
    try:
        # This is a simplified implementation
        # In a production environment, you would want to get actual stats from Pinecone
        return {
            "total_documents": "Unknown (implement Pinecone stats retrieval)",
            "vector_store_status": "active",
            "index_name": os.environ.get("PINECONE_INDEX_NAME", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# Add the RAG chain endpoint
add_routes(app, document_rag_chain, path="/rag", enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
