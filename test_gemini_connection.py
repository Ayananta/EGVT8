#!/usr/bin/env python3
"""
Test Gemini Connection
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is loaded
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key loaded: {'Yes' if api_key else 'No'}")
if api_key:
    print(f"API Key (first 20 chars): {api_key[:20]}...")

# Test Gemini connection
if api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, this is a test message. Please respond with 'Connection successful!'")
        
        print(f"✅ Gemini connection successful!")
        print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
else:
    print("❌ No API key found")
