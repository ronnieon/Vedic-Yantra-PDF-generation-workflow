# 🔒 Security Notice

## API Key Protection

Your API keys are sensitive credentials that should be kept secure. This project follows best practices for API key management:

### ✅ What We Do

1. **Environment Variables**: API keys are stored in `.envrc` (not in code)
2. **Git Ignore**: `.envrc` is automatically ignored by git (won't be committed)
3. **Template File**: `.envrc.example` provides a safe template without real keys
4. **Runtime Loading**: Keys are loaded from environment at runtime, never hardcoded

### ⚠️ Security Best Practices

**DO:**
- ✅ Keep your `.envrc` file private
- ✅ Use different API keys for different projects
- ✅ Rotate your keys regularly
- ✅ Review `.gitignore` before committing

**DON'T:**
- ❌ Share your `.envrc` file
- ❌ Commit API keys to version control
- ❌ Post your keys in public forums/screenshots
- ❌ Use production keys for testing

### 🔍 If You Accidentally Expose a Key

1. **Immediately revoke** the exposed key in your service dashboard
2. **Generate a new key** and update `.envrc`
3. **Check git history** to ensure the key isn't committed
4. If committed to git:
   ```bash
   # Remove from git history (careful!)
   git filter-branch --force --index-filter \
   "git rm --cached --ignore-unmatch .envrc" \
   --prune-empty --tag-name-filter cat -- --all
   ```

### 📍 Where to Manage Your Keys

- **Replicate**: https://replicate.com/account/api-tokens
- **ElevenLabs**: https://elevenlabs.io/app/settings/api-keys
- **Google Gemini**: https://makersuite.google.com/app/apikey

### 🔐 Additional Security Tips

1. **Rate Limiting**: The app includes rate limiting to prevent excessive API usage
2. **Local Processing**: Images are processed locally before PDF generation
3. **No Data Persistence**: Session data is cleared when you close the app
4. **HTTPS Only**: Always use HTTPS when accessing external APIs

---

**Remember**: Treat your API keys like passwords. Keep them secret, keep them safe! 🔒
