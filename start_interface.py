#!/usr/bin/env python3
"""
Startup script for the Document Q&A System
This script will help you start both the API server and web interface
"""

import subprocess
import time
import sys
import os
from threading import Thread
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_api_server(port=8001):
    """Check if the API server is running"""
    try:
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_web_interface(port=8501):
    """Check if the web interface is running"""
    try:
        response = requests.get(f"http://127.0.0.1:{port}", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the API server in a separate process"""
    print("Starting API server...")
    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "app.server:app", 
            "--host", "127.0.0.1", 
            "--port", "8001",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("Waiting for API server to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_api_server(8001):
                print("SUCCESS: API server is running on http://127.0.0.1:8001")
                return process
            time.sleep(1)
            print(f"   Waiting... ({i+1}/30)")
        
        print("ERROR: API server failed to start within 30 seconds")
        return None
        
    except Exception as e:
        print(f"ERROR: Failed to start API server: {e}")
        return None

def start_web_interface():
    """Start the Streamlit web interface"""
    print("Starting web interface...")
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "web_interface.py",
            "--server.port", "8502",
            "--server.address", "127.0.0.1"
        ])
    except Exception as e:
        print(f"ERROR: Failed to start web interface: {e}")

def main():
    print("=" * 60)
    print("Document Q&A System - Startup")
    print("=" * 60)
    
    # Check if environment variables are set
    required_env_vars = [
        'PINECONE_API_KEY',
        'PINECONE_ENVIRONMENT', 
        'PINECONE_INDEX_NAME',
        'OPENAI_API_KEY',
        'COHERE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("ERROR: Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file and try again.")
        return
    
    print("SUCCESS: All required environment variables are set")
    
    # Check if services are already running
    api_running = check_api_server(8001)
    web_running = check_web_interface(8501)
    
    if api_running:
        print("SUCCESS: API server is already running on http://127.0.0.1:8001")
    
    if web_running:
        print("SUCCESS: Web interface is already running on http://127.0.0.1:8501")
        print("\nYou can access the system at: http://127.0.0.1:8501")
        print("Both services are running and ready to use!")
        return
    
    # Start API server if not running
    api_process = None
    if not api_running:
        api_process = start_api_server()
        if not api_process:
            print("ERROR: Cannot proceed without API server")
            return
    else:
        api_process = None  # Already running, don't manage it
    
    # Start web interface if not running
    if not web_running:
        print("\n" + "=" * 60)
        print("Starting web interface at http://127.0.0.1:8501")
        print("=" * 60)
        print("Instructions:")
        print("1. The web interface will open in your browser")
        print("2. Upload documents using the sidebar (PDF or TXT files)")
        print("3. Ask questions about your uploaded documents")
        print("4. Close this terminal to stop both services")
        print("=" * 60)
        
        try:
            start_web_interface()
        except KeyboardInterrupt:
            print("\nShutting down...")
            if api_process:
                api_process.terminate()
            print("Services stopped")

if __name__ == "__main__":
    main()
