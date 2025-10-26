# RAG AI Virtual Assistant

A sophisticated Retrieval-Augmented Generation (RAG) AI Virtual Assistant built with LangChain, Azure OpenAI, and Pinecone. This application allows users to upload documents (PDF, TXT) and ask questions based on the content of those documents.

## Features

- **Document Processing**: Upload and process PDF and TXT files
- **Intelligent Q&A**: Ask questions about your uploaded documents
- **Multiple LLM Support**: Azure OpenAI with fallback options
- **Vector Database**: Pinecone for efficient document retrieval
- **Web Interface**: Clean and intuitive Streamlit interface
- **API Support**: FastAPI backend with LangServe integration
- **Scalable**: Ready for Azure Web App deployment

## Technology Stack

- **Backend**: Python, FastAPI, LangChain, LangServe
- **Frontend**: Streamlit
- **Vector Database**: Pinecone
- **LLM**: Azure OpenAI (GPT-4o, text-embedding-3-large)
- **Fallback**: OpenAI, Cohere
- **Deployment**: Azure Web App, GitHub Actions

## Prerequisites

- Python 3.8+
- Azure OpenAI Resource
- Pinecone Account
- (Optional) LangSmith Account for tracing

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MaddyRizvi/Rag-AI-Virtual-Assistant.git
cd Rag-AI-Virtual-Assistant
```

### 2. Set Up Environment Variables

Copy the example environment file and fill in your actual values:

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys and endpoints:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=index-azure-openai-3072

# Azure OpenAI Configuration
AZ_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com/
AZ_OPENAI_API_KEY=your_azure_openai_api_key_here
AZ_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZ_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZ_OPENAI_API_VERSION=2024-12-01-preview

# Optional: LangSmith for tracing
LANGCHAIN_API_KEY=your_langsmith_api_key_here
```

### 3. Install Dependencies

Using conda (recommended):

```bash
conda create -n rag-assistant python=3.10
conda activate rag-assistant
conda install -c conda-forge poetry
poetry install
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the API server:

```bash
conda activate rag-assistant
python -m uvicorn app.server:app --host 127.0.0.1 --port 8001
```

In another terminal, start the web interface:

```bash
conda activate rag-assistant
streamlit run web_interface.py --server.port 8501
```

### 5. Access the Application

- Web Interface: http://localhost:8501
- API Documentation: http://localhost:8001/docs
- LangServe Playground: http://localhost:8001/rag/playground/

## Azure Deployment

### 1. Azure Web App Setup

The application is configured for automatic deployment to Azure Web App using GitHub Actions.

#### Required Azure Resources:
- Azure Web App (Linux)
- Azure OpenAI Service
- Pinecone Database
- Application Settings (Environment Variables)

#### Environment Variables for Azure:

Set these in your Azure Web App Configuration:

```bash
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=index-azure-openai-3072
AZ_OPENAI_ENDPOINT=https://your-azure-openai-resource.openai.azure.com/
AZ_OPENAI_API_KEY=your_azure_openai_api_key
AZ_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZ_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZ_OPENAI_API_VERSION=2024-12-01-preview
LANGCHAIN_API_KEY=your_langsmith_api_key
```

### 2. GitHub Actions Workflow

The repository includes a GitHub Actions workflow for automatic deployment:

1. **Push to main branch**: Automatically deploys to Azure Web App
2. **Pull requests**: Runs tests and validation
3. **Manual triggers**: Can be triggered manually from GitHub Actions tab

### 3. Deployment Steps

1. Fork or clone this repository
2. Set up Azure Web App with the required configuration
3. Configure GitHub Secrets in your repository:
   - `AZURE_WEBAPP_PUBLISH_PROFILE`: Download from Azure portal
   - `AZURE_WEBAPP_NAME`: Your web app name
   - `AZURE_WEBAPP_SLOT_NAME`: Production (or your slot name)
4. Push your changes to trigger automatic deployment

## API Usage

### Chat Endpoint

```bash
curl -X POST "http://localhost:8001/rag/invoke" \
     -H "Content-Type: application/json" \
     -d '{"input": "What is the main topic of the document?"}'
```

### Upload Document

```bash
curl -X POST "http://localhost:8001/upload/text" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your document content here..."}'
```

## Development

### Project Structure

```
Rag-AI-Virtual-Assistant/
├── app/
│   ├── __init__.py
│   ├── chain.py              # LangChain RAG chain
│   ├── document_processor.py  # Document processing logic
│   └── server.py             # FastAPI server
├── web_interface.py           # Streamlit frontend
├── start_interface.py        # Helper script
├── .env.example             # Environment variables template
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Poetry configuration
├── Dockerfile             # Docker configuration
└── README.md              # This file
```

### Adding New Features

1. **New Document Types**: Extend `document_processor.py`
2. **New LLM Providers**: Update `chain.py`
3. **UI Enhancements**: Modify `web_interface.py`
4. **API Endpoints**: Add to `server.py`

## Troubleshooting

### Common Issues

1. **Pinecone Index Issues**: Make sure your Pinecone index has the correct dimensions (3072 for Azure OpenAI text-embedding-3-large)
2. **Azure OpenAI Errors**: Verify your endpoint URL and API key
3. **Environment Variables**: Ensure all required variables are set
4. **Dependency Issues**: Use conda environment for better compatibility

### Logs and Monitoring

- **Application Logs**: Available in Azure Portal
- **LangSmith Tracing**: Configure with LANGCHAIN_API_KEY
- **Streamlit Logs**: Console output when running locally

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

## Acknowledgments

- LangChain for the RAG framework
- Azure OpenAI for powerful LLM capabilities
- Pinecone for vector database
- Streamlit for the web interface
- FastAPI for the backend API
