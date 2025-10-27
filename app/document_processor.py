import os
import tempfile
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from dotenv import load_dotenv
import hashlib

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Initialize Pinecone client
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME", "index-azure-openai-3072")
        
        # Validate required environment variables
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pinecone_client = PineconeClient(
            api_key=self.pinecone_api_key,
            environment=self.pinecone_environment
        )
        
        # Initialize embeddings using Azure OpenAI
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            
            # Get Azure OpenAI configuration with defaults
            azure_deployment = os.environ.get("AZ_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
            azure_endpoint = os.environ.get("AZ_OPENAI_ENDPOINT")
            azure_api_key = os.environ.get("AZ_OPENAI_API_KEY")
            azure_api_version = os.environ.get("AZ_OPENAI_API_VERSION", "2024-12-01-preview")
            
            # Validate required Azure OpenAI variables
            if not azure_endpoint or not azure_api_key:
                print("Azure OpenAI credentials not found, falling back to Cohere embeddings")
                raise ValueError("Azure OpenAI credentials missing")
            
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=azure_deployment,
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=azure_api_version
            )
            print(f"Using Azure OpenAI embeddings: {azure_deployment}")
        except Exception as e:
            print(f"Error initializing Azure OpenAI embeddings: {e}")
            # Fallback to Cohere as backup
            try:
                self.embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")
                print("Fallback: Using Cohere embed-multilingual-v3.0 model (1024 dimensions)")
            except Exception as e2:
                print(f"Error initializing fallback embeddings: {e2}")
                self.embeddings = CohereEmbeddings(model="multilingual-22-12")
                print("WARNING: Using Cohere multilingual-22-12 model (768 dimensions) - this may not match index dimensions")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize vector store
        self.vectorstore = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the Pinecone vector store"""
        try:
            # Check if index exists
            if self.pinecone_index_name not in self.pinecone_client.list_indexes().names():
                print(f"Index '{self.pinecone_index_name}' not found. Creating new index...")
                # Create index with correct dimensions for Azure OpenAI text-embedding-3-large (3072 dimensions)
                self.pinecone_client.create_index(
                    name=self.pinecone_index_name,
                    dimension=3072,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Index '{self.pinecone_index_name}' created successfully")
            
            # Initialize vector store with existing index using the new API
            self.vectorstore = Pinecone.from_existing_index(
                index_name=self.pinecone_index_name,
                embedding=self.embeddings
            )
            print(f"Vector store initialized successfully with index '{self.pinecone_index_name}'")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            # Try alternative initialization for newer Pinecone API
            try:
                # Try using the newer Pinecone API structure
                from pinecone import Pinecone as NewPineconeClient
                new_client = NewPineconeClient(api_key=self.pinecone_api_key)
                
                # Get the index
                index = new_client.Index(self.pinecone_index_name)
                
                # Initialize vector store with the new client
                self.vectorstore = Pinecone(
                    index=index,
                    embedding_function=self.embeddings,
                    text_key="text"
                )
                print(f"Vector store initialized successfully with new API for index '{self.pinecone_index_name}'")
                
            except Exception as e2:
                print(f"Failed to initialize with new API as well: {e2}")
                raise Exception(f"Failed to initialize Pinecone vector store: {str(e)}")
    
    def _generate_document_id(self, content: str) -> str:
        """Generate a unique ID for the document based on its content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def process_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Process raw text into chunks"""
        if metadata is None:
            metadata = {}
        
        # Generate document ID
        doc_id = self._generate_document_id(text)
        metadata["doc_id"] = doc_id
        metadata["source_type"] = "text"
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = f"{doc_id}_{i}"
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        return documents
    
    def process_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Process a file based on its extension"""
        if metadata is None:
            metadata = {}
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            metadata["source_type"] = "text_file"
            metadata["file_name"] = os.path.basename(file_path)
            return self.process_text(content, metadata)
        
        elif file_extension == '.pdf':
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"
                metadata["source_type"] = "pdf_file"
                metadata["file_name"] = os.path.basename(file_path)
                metadata["num_pages"] = len(reader.pages)
                return self.process_text(content, metadata)
            except Exception as e:
                raise Exception(f"Error processing PDF file: {e}")
        
        else:
            raise Exception(f"Unsupported file type: {file_extension}")
    
    def add_documents_to_vectorstore(self, documents: List[Document]):
        """Add processed documents to the vector store"""
        try:
            # Add documents to Pinecone
            self.vectorstore.add_documents(documents)
            return True
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            return False
    
    def search_documents(self, query: str, k: int = 5):
        """Search for relevant documents"""
        if self.vectorstore is None:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def get_retriever(self):
        """Get the retriever for the RAG chain"""
        if self.vectorstore is None:
            raise Exception("Vector store not initialized")
        
        # Use synchronous retriever to avoid session closure issues
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def clear_documents(self, doc_id: str = None):
        """Clear documents from the vector store"""
        # This is a simplified implementation
        # In a production environment, you would want to implement proper document deletion
        print(f"Clear documents functionality not fully implemented. Doc ID: {doc_id}")
        return True
