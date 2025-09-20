#!/usr/bin/env python3
"""
Gemini API Setup Script

This script helps you set up your Gemini API key for the RAG system.
"""

import os
from pathlib import Path
import google.generativeai as genai


def setup_gemini_api():
    """Setup Gemini API key."""
    print("🤖 Gemini API Setup")
    print("=" * 40)
    
    # Check if API key already exists
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print("✅ GEMINI_API_KEY found in environment variables")
        try:
            genai.configure(api_key=api_key)
            print("✅ API key is valid and working")
            return True
        except Exception as e:
            print(f"❌ API key validation failed: {e}")
            api_key = None
    
    if not api_key:
        print("\n📝 To get your Gemini API key:")
        print("1. Go to: https://makersuite.google.com/app/apikey")
        print("2. Sign in with your Google account")
        print("3. Click 'Create API Key'")
        print("4. Copy the generated API key")
        
        print("\n🔧 Setup options:")
        print("Option 1: Create .env file (recommended)")
        print("Option 2: Set environment variable")
        print("Option 3: Enter API key now (temporary)")
        
        choice = input("\nChoose option (1/2/3): ").strip()
        
        if choice == '1':
            # Create .env file
            api_key = input("Enter your Gemini API key: ").strip()
            if api_key:
                with open('.env', 'w') as f:
                    f.write(f'GEMINI_API_KEY={api_key}\n')
                print("✅ API key saved to .env file")
                return True
        
        elif choice == '2':
            # Set environment variable
            print("\nRun this command in your terminal:")
            print(f'set GEMINI_API_KEY=your_api_key_here')
            print("\nOr for PowerShell:")
            print(f'$env:GEMINI_API_KEY="your_api_key_here"')
            return False
        
        elif choice == '3':
            # Temporary setup
            api_key = input("Enter your Gemini API key: ").strip()
            if api_key:
                os.environ['GEMINI_API_KEY'] = api_key
                print("✅ API key set for current session")
                return True
        
        else:
            print("❌ Invalid choice")
            return False
    
    return False


def test_gemini_connection():
    """Test Gemini API connection."""
    print("\n🧪 Testing Gemini connection...")
    
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            print("❌ No API key found")
            return False
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content("Hello, this is a test message. Please respond with 'Connection successful!'")
        
        print("✅ Gemini connection successful!")
        print(f"Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def create_example_questions():
    """Create example questions file."""
    questions = [
        "What is this legal case about?",
        "What are the main arguments of the appellants?",
        "What does the West Bengal Apartment Ownership Act of 1972 say?",
        "What are the duties and liabilities of a promoter?",
        "What was the court's final decision?",
        "What is the New Town Kolkata Development Authority's role?",
        "What are the requirements for building construction permits?",
        "What are the rights of apartment owners?",
        "What is the relationship between NKDA Act and WB Apartment Ownership Act?",
        "What are the legal precedents cited in this case?"
    ]
    
    with open('example_questions.txt', 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(question + '\n')
    
    print("✅ Created example_questions.txt with sample questions")


def main():
    """Main setup function."""
    print("🚀 Setting up Gemini RAG System")
    print("=" * 50)
    
    # Setup API key
    if setup_gemini_api():
        # Test connection
        if test_gemini_connection():
            print("\n🎉 Setup complete! You can now use the Gemini RAG system.")
            
            # Create example questions
            create_example_questions()
            
            print("\n📋 Next steps:")
            print("1. Try a single question:")
            print("   python gemini_rag.py --question 'What is this legal case about?'")
            print("\n2. Start interactive chat:")
            print("   python gemini_rag.py --chat")
            print("\n3. Process example questions:")
            print("   python gemini_rag.py --batch example_questions.txt")
        else:
            print("\n❌ Setup incomplete. Please check your API key.")
    else:
        print("\n⚠️  Please complete the API key setup and run this script again.")


if __name__ == "__main__":
    main()
