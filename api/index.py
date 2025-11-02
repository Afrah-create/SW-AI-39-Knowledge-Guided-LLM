"""
Vercel Serverless Function Entry Point
This file adapts the Flask app for Vercel's serverless environment
"""
from app.main import app

# Vercel automatically detects and uses the 'app' variable
# The Flask app will be served as a serverless function

