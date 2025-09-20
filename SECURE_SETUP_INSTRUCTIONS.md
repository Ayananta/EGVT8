# 🔐 SECURE SETUP INSTRUCTIONS

## ⚠️ **CRITICAL SECURITY WARNING**

**Your Gemini API key has been exposed in your GitHub repository!**

## 🚨 **IMMEDIATE ACTIONS REQUIRED:**

### **1. Regenerate Your API Key (URGENT)**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **DELETE** the exposed API key: `AIzaSyDk0JhMnS5S41RAsV6f8v3qCuS3PFGnn-Y`
3. **CREATE** a new API key
4. **KEEP** the new key secure

### **2. Secure Your Repository**
```bash
# Remove .env from Git tracking (run these commands)
git rm --cached .env
git commit -m "Remove exposed API key from repository"
git push origin main

# The .gitignore file will prevent future commits of .env
```

### **3. Set Up Environment Variables Securely**

#### **Option A: Local Development**
1. Copy `env_template.txt` to `.env`
2. Replace `YOUR_API_KEY_HERE` with your new API key
3. The `.env` file is now ignored by Git

#### **Option B: Production/Sharing**
**NEVER** share your `.env` file. Instead:

1. **For others to use your system:**
   - Share `env_template.txt`
   - Tell them to get their own API key
   - They create their own `.env` file

2. **For cloud deployment:**
   - Set environment variables in your hosting platform
   - Never put API keys in code

## 🔒 **Security Best Practices:**

### **✅ DO:**
- Keep API keys in `.env` files
- Add `.env` to `.gitignore`
- Use environment variables in production
- Share template files, not actual keys
- Rotate API keys regularly

### **❌ NEVER:**
- Commit API keys to Git
- Share `.env` files
- Put API keys in code
- Use the same key across projects

## 📋 **For Users Setting Up the System:**

1. **Get your own API key:**
   - Visit: https://makersuite.google.com/app/apikey
   - Create a new API key
   - Copy the key

2. **Create your .env file:**
   ```bash
   # Copy the template
   copy env_template.txt .env
   
   # Edit .env and replace YOUR_API_KEY_HERE with your actual key
   ```

3. **Verify it works:**
   ```bash
   python test_gemini_connection.py
   ```

## 🆘 **If Your Key Was Misused:**

1. **Immediately** delete the exposed key
2. **Monitor** your Google Cloud Console for unusual usage
3. **Create** a new key
4. **Update** all your applications with the new key
5. **Review** your Google Cloud billing

## 📞 **Support:**

If you need help securing your setup:
- Check Google Cloud Console for API usage
- Review the `.gitignore` file to ensure `.env` is ignored
- Test your setup with the new API key

---

**Remember: API keys are like passwords - keep them secret!**
