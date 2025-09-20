#!/usr/bin/env python3
"""
Test script for the web application.
"""

import requests
import time
import json

def test_web_app():
    base_url = "http://localhost:5000"
    
    print("Testing web application...")
    
    # Wait a moment for server to start
    time.sleep(2)
    
    try:
        # Test status endpoint
        print("1. Testing status endpoint...")
        response = requests.get(f"{base_url}/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status', 'unknown')}")
            if data.get('success'):
                print(f"   ✅ Documents: {data.get('total_chunks', 0)} chunks")
            else:
                print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Status code: {response.status_code}")
        
        # Test sample questions endpoint
        print("\n2. Testing sample questions endpoint...")
        response = requests.get(f"{base_url}/api/sample_questions", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Sample questions loaded: {len(data.get('questions', []))} questions")
        else:
            print(f"   ❌ Status code: {response.status_code}")
        
        # Test a simple query
        print("\n3. Testing query endpoint...")
        query_data = {
            "question": "What are the payment terms?",
            "top_k": 3,
            "max_context": 2000
        }
        response = requests.post(f"{base_url}/api/query", 
                               json=query_data, 
                               timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   ✅ Query successful")
                print(f"   ✅ Answer length: {len(data.get('answer', ''))} characters")
                print(f"   ✅ Sources found: {data.get('chunks_count', 0)} chunks")
            else:
                print(f"   ❌ Query failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"   ❌ Status code: {response.status_code}")
            print(f"   Response: {response.text}")
        
        print("\n🎉 Web application is working!")
        print(f"🌐 Open your browser and go to: {base_url}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the web server.")
        print("   Make sure the server is running on http://localhost:5000")
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The server might be slow to respond.")
    except Exception as e:
        print(f"❌ Error testing web app: {e}")

if __name__ == "__main__":
    test_web_app()
