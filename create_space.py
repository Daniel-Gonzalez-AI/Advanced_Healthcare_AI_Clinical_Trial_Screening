"""
Script to create a Hugging Face Space for this project. Requires HUGGINGFACE_TOKEN in environment.
"""
import os
# Attempt to load environment variables from .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed; ensure HUGGINGFACE_TOKEN is set in environment.")
from huggingface_hub import HfApi, HfFolder

def main():
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set.")
    HfFolder.save_token(token)
    api = HfApi()
    space_id = "danielgonzalez/clinical-trial-screening"
    # Create a new Gradio space
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            private=False
        )
        print(f"Space '{space_id}' created successfully.")
    except Exception as e:
        print(f"Error creating space: {e}")

if __name__ == "__main__":
    main()
