"""
GraphRAG Application Launcher.
Starts the FastAPI server and automatically opens http://localhost:8000 in your browser.
"""

import os
import sys
import time
import webbrowser
import uvicorn

def main():
    # Ensure current directory is in sys.path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    host = "127.0.0.1"
    port = 8000
    url = f"http://localhost:{port}"

    print("=" * 60)
    print(" 🚀 GraphRAG - Knowledge Graph Enhanced Q&A System")
    print(f" 🌐 Dashboard URL: {url}")
    print(f" 📖 API Docs:      {url}/docs")
    print("=" * 60)

    # Open browser after short delay
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start Uvicorn
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    main()
