"""
Vercel Serverless Function Entry Point
This file adapts the Flask app for Vercel's serverless environment
"""
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# Export the Flask app for Vercel
# Vercel will automatically detect and use the 'app' variable

