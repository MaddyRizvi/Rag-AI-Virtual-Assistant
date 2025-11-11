import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from app.document_processor import DocumentProcessor

load_dotenv()

# Load the .env file explicitly
load_dotenv(dotenv_path=".env")

# LangSmith configuration
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "default"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize document processor (lazy loading)
doc_processor = None

def get_doc_processor():
    """Lazy load document processor"""
    global doc_processor
    if doc_processor is None:
        doc_processor = DocumentProcessor()
    return doc_processor

# Create a custom synchronous retriever to avoid async session issues
class SyncRetriever(BaseRetriever):
    doc_processor: DocumentProcessor
    course_id: str
    
    def __init__(self, doc_processor: DocumentProcessor, course_id: str = "general"):
        super().__init__(doc_processor=doc_processor)
        self.course_id = course_id
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous document retrieval to avoid async session issues"""
        try:
            # Use synchronous search method with course namespace
            return self.doc_processor.search_documents(query, course_id=self.course_id, k=5)
        except Exception as e:
            print(f"Error in document retrieval for course '{self.course_id}': {e}")
            return []

# Get custom retriever (lazy loading with course support)
retrievers = {}  # Cache retrievers per course

def get_retriever(course_id: str = "general"):
    """Lazy load retriever for specific course"""
    global retrievers
    if course_id not in retrievers:
        retrievers[course_id] = SyncRetriever(get_doc_processor(), course_id)
    return retrievers[course_id]

# RAG prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}

Provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided documents."
"""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the model using Azure OpenAI
# Get Azure OpenAI configuration with defaults
azure_deployment = os.environ.get("AZ_OPENAI_CHAT_DEPLOYMENT", "gpt-4o")
azure_endpoint = os.environ.get("AZ_OPENAI_ENDPOINT")
azure_api_key = os.environ.get("AZ_OPENAI_API_KEY")
azure_api_version = os.environ.get("AZ_OPENAI_API_VERSION", "2024-12-01-preview")

# Validate required Azure OpenAI variables
if not azure_endpoint or not azure_api_key:
    error_msg = """
    ❌ Azure OpenAI credentials are missing!
    
    To fix this issue:
    
    1. In Azure Portal:
       - Go to your Web App
       - Click on "Configuration"
       - Under "Application settings", add these settings:
       - Name: AZ_OPENAI_ENDPOINT
         Value: your_azure_openai_endpoint_here
       - Name: AZ_OPENAI_API_KEY
         Value: your_azure_openai_api_key_here
       - Name: AZ_OPENAI_CHAT_DEPLOYMENT
         Value: gpt-4o (or your deployment name)
       - Click OK, then Save
    
    2. Or locally, add to your .env file:
       AZ_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
       AZ_OPENAI_API_KEY=your_azure_openai_api_key_here
       AZ_OPENAI_CHAT_DEPLOYMENT=gpt-4o
    
    3. Get your Azure OpenAI credentials from:
       - Azure Portal -> Azure OpenAI -> Your resource -> Keys and Endpoint
    """
    raise ValueError(error_msg)

model = AzureChatOpenAI(
    temperature=0,
    azure_deployment=azure_deployment,
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=azure_api_version
)

# Create the RAG chain (lazy loading)
chain = None

def get_rag_chain(course_id: str = "general"):
    """Lazy load RAG chain with course support"""
    global chain
    if chain is None:
        chain = (
            RunnableParallel({"context": get_retriever(course_id), "question": RunnablePassthrough()})
            | prompt
            | model
            | StrOutputParser()
        )
    return chain

# Expose the document processor for use in server.py
__all__ = ["chain", "doc_processor"]

def create_rag_chain(course_id: str = "general"):
    """Create and return the RAG chain for use in other modules with course support"""
    # Create chain for specific course
    chain = (
        RunnableParallel({"context": get_retriever(course_id), "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )
    return chain
