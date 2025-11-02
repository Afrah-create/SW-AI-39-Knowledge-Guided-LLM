"""
Vercel Serverless Function Entry Point
This file adapts the Flask app for Vercel's serverless environment
"""
from app.main import app

# Vercel serverless function handler
def handler(request):
    """Handle Vercel serverless function requests"""
    # Return the Flask WSGI application
    return app

