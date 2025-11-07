"""
Quick Start Script for Flask App
Checks dependencies and starts the server
"""

import subprocess
import sys
from pathlib import Path

def check_flask_installed():
    """Check if Flask is installed"""
    try:
        import flask
        print(f"✓ Flask {flask.__version__} is installed")
        return True
    except ImportError:
        print("✗ Flask is not installed")
        return False

def install_flask():
    """Install Flask dependencies"""
    print("\nInstalling Flask dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_flask.txt"])

def main():
    print("="*60)
    print("DNS SPOOFING DETECTION - FLASK WEB APP")
    print("="*60)
    
    # Check Flask
    if not check_flask_installed():
        response = input("\nInstall Flask now? (y/n): ")
        if response.lower() == 'y':
            install_flask()
        else:
            print("Please install Flask first: pip install -r requirements_flask.txt")
            return
    
    # Check directories
    templates_dir = Path(__file__).parent / 'templates'
    uploads_dir = Path(__file__).parent / 'uploads'
    
    if not templates_dir.exists():
        print(f"\n✗ Templates directory not found: {templates_dir}")
        return
    
    uploads_dir.mkdir(exist_ok=True)
    print(f"✓ Uploads directory ready: {uploads_dir}")
    
    # Start server
    print("\n" + "="*60)
    print("Starting Flask server...")
    print("Access the web interface at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\nServer stopped.")

if __name__ == "__main__":
    main()
