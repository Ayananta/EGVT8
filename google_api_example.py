"""
Example Python script using Google API Key with environment variables
"""
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_google_api_key():
    """Get Google API Key from environment variables"""
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    return api_key

def main():
    """Main function demonstrating API key usage"""
    try:
        # Get the API key
        api_key = get_google_api_key()
        print(f"Google API Key loaded successfully: {api_key[:10]}...")
        
        # Example: Using the API key for Google services
        # You can use this with various Google APIs like:
        # - Google Maps API
        # - Google Translate API
        # - Google Cloud APIs
        # - YouTube Data API
        # - etc.
        
        # Example URL construction for Google Maps API
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        address = "New York, NY"
        url = f"{base_url}?address={address}&key={api_key}"
        
        print(f"Example API URL: {url}")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure you have a .env file with GOOGLE_API_KEY defined")

if __name__ == "__main__":
    main()




