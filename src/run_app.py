#!/usr/bin/env python3
"""
VerifAI-ML Application Launcher
Simple script to launch the Streamlit app with proper configuration
"""

import subprocess
import sys
import os

def main():
    """Launch the VerifAI-ML Streamlit application"""
    
    print("🚀 Starting VerifAI-ML Application...")
    print("=" * 50)
    
    # Check if requirements are installed
    try:
        import streamlit
        import torch
        import ultralytics
        print("✅ All required packages are installed")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install requirements with: pip install -r requirements_phase4.txt")
        return
    
    # Launch Streamlit app
    try:
        print("🌐 Launching web interface...")
        print("📱 The app will open in your default browser")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run streamlit with optimized settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 VerifAI-ML application stopped")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

if __name__ == "__main__":
    main()
