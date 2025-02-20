# 🔒 SECURITY POLICY  

## 📌 Supported Versions  

| Version | Status                 | Notes                  |  
|---------|------------------------|------------------------|  
| v1.x    | ❌ No Longer Maintained | Python-only version   |  

## 🛠️ Reporting a Vulnerability  
If you find a security issue, please **do not** disclose it publicly. Instead, follow these steps:  

1. **Email the Issue:** Send details to **[omgohel760l@gmail.com]**.  
2. **Include:**  
   - Steps to reproduce the issue  
   - Affected components  
   - Suggested fixes (if known)  
3. **Response Time:** We will acknowledge within **48 hours** and work on a fix.  

## 🔐 Security Best Practices  
- **API Keys:** Do **not** expose your **Spotify API keys** in public repositories. Store them in an environment file:  
  ```sh
  export SPOTIFY_CLIENT_ID="your_client_id"
  export SPOTIFY_CLIENT_SECRET="your_client_secret"
- Dependencies: Regularly update all libraries to prevent vulnerabilities:
- **pip install --upgrade -r requirements.txt** 

- Model Security: Ensure trained models are from trusted sources to prevent adversarial attacks.
## 🚀 Future Security Considerations
### Secure API keys using dotenv or environment variables.
- Implement user authentication for personalized recommendations.
 - Restrict API rate limits to prevent misuse.
## 📝 License
- This project follows the MIT License – see LICENSE for details.

