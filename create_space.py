"""
Script to create a Hugging Face Space for this project. Requires HUGGINGFACE_TOKEN in environment.
"""
import os
# Attempt to load environment variables from .env if python-dotenv is available
try:
    # Attempt to load environment variables from .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed; attempting manual .env parsing.")
    # Manual .env parsing
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, val = line.strip().split('=', 1)
                    os.environ.setdefault(key, val)
    else:
        print(".env file not found; ensure HUGGINGFACE_TOKEN is set in environment.")
from huggingface_hub import HfApi, HfFolder

def main():
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
    HfFolder.save_token(token)
    api = HfApi()
    # Determine space ID: use SPACE_ID env var or default to your Hugging Face namespace and project slug
    default_slug = "clinical-trial-screening"
    hf_namespace = os.getenv("HF_NAMESPACE", "ArtemisAI")  # Current authenticated HF namespace
    space_id = os.getenv("SPACE_ID", f"{hf_namespace}/{default_slug}")
    # Create a new Gradio space
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=False
        )
        print(f"Space '{space_id}' created successfully.")
        print(f"\nTo deploy, add a git remote: git remote add hf-space https://huggingface.co/spaces/{space_id}")
        print("Then push your code: git push hf-space main")
    except Exception as e:
        print(f"Error creating space: {e}")

if __name__ == "__main__":
    main()
