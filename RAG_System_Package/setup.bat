@echo off
echo Installing RAG System Dependencies...
pip install -r requirements.txt

echo.
echo Setting up Gemini API...
echo Please get your API key from: https://makersuite.google.com/app/apikey
echo.
python setup_gemini.py

echo.
echo Testing system...
python test_gemini_integration.py

echo.
echo Setup complete! Try asking a question:
echo python gemini_rag.py --question "What is this legal case about?"
pause
