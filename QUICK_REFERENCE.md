# 🚀 Quick Reference: AI Project Deployment Checklist

*Based on Advanced Healthcare AI Clinical Trial Screening System (Sept 2025)*

## ⚡ **Fast Track Setup (15 minutes)**

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv ~/venvs/my-ai-project
source ~/venvs/my-ai-project/bin/activate

# Install core dependencies
pip install torch transformers gradio datasets scikit-learn pytest python-dotenv
```

### 2. Project Structure
```bash
# Create essential files
touch app.py core_module.py test_system.py requirements.txt README.md create_space.py .env .gitignore
```

### 3. Security Setup (.gitignore)
```gitignore
.env
__pycache__/
*.pyc
models/
data/
logs/
```

## 🔑 **Critical Issue Prevention**

### ❌ **Common Pitfalls to Avoid**
1. **PEP 668 Error**: Always use virtual environments
2. **HF Token Issues**: Verify namespace with `api.whoami()` first
3. **Git Auth Fails**: Use token in remote URL: `https://USERNAME:TOKEN@huggingface.co/...`
4. **Model Accuracy**: Use domain-specific models (Bio_ClinicalBERT > DistilBERT)
5. **Missing Dependencies**: Pin exact versions in requirements.txt

### ✅ **Pre-Deployment Verification**
```bash
# Test everything works
python test_system.py

# Check HF permissions
python -c "from huggingface_hub import HfApi; print(HfApi().whoami())"

# Verify git status
git status
```

## 🎯 **Deployment Commands**

### GitHub
```bash
git add .
git commit -m "🚀 Initial deployment"
git push origin main
```

### Hugging Face Space
```bash
# Create space
python create_space.py

# Add remote with token auth
git remote add hf-space https://USERNAME:$HUGGINGFACE_TOKEN@huggingface.co/spaces/USERNAME/project-name

# Deploy
git push hf-space main --force
```

## 📋 **Must-Have Files**

- [ ] `app.py` - Gradio interface
- [ ] `core_module.py` - AI implementation  
- [ ] `test_system.py` - 4-test validation
- [ ] `requirements.txt` - Exact versions
- [ ] `README.md` - Professional docs
- [ ] `create_space.py` - HF helper script
- [ ] `.gitignore` - Security

## 🔧 **Copy-Paste Templates**

### Essential create_space.py
```python
import os
from huggingface_hub import HfApi

# Auto-load .env
try:
    from dotenv import load_dotenv; load_dotenv()
except: pass

api = HfApi()
user = api.whoami()['name']
space_id = f"{user}/my-project-name"

api.create_repo(repo_id=space_id, repo_type="space", space_sdk="gradio")
print(f"✅ Created: https://huggingface.co/spaces/{space_id}")
```

### 4-Test Framework
```python
def test_system():
    tests = [("Model Loading", test_model), ("Prediction", test_predict), 
             ("Batch", test_batch), ("Data Gen", test_data)]
    results = []
    for name, test in tests:
        try: test(); print(f"✅ {name}"); results.append(True)
        except Exception as e: print(f"❌ {name}: {e}"); results.append(False)
    print(f"📊 {sum(results)}/{len(results)} passed")
    return all(results)
```

## ⏱️ **Success Timeline**
- **0-30min**: Environment + structure setup
- **30min-2hr**: Core AI implementation  
- **2-4hr**: Testing + documentation
- **4-6hr**: Deployment + verification

## 🎯 **Quality Gates**
1. ✅ All tests pass (`python test_system.py`)
2. ✅ Local Gradio works (`python app.py`)
3. ✅ Clean git status (`git status`)
4. ✅ HF token verified (`api.whoami()`)
5. ✅ Professional README with attribution

---

**💡 Pro Tip**: Save 2+ hours by following this checklist exactly!

*For full details, see PROJECT_LESSONS_LEARNED.md*
