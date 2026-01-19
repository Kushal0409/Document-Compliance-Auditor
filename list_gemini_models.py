"""
Utility script to list available Gemini models.
Run this to see which models you can use with your API key.
"""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    from google import genai
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in .env file or environment.")
        exit(1)
    
    client = genai.Client(api_key=api_key)
    models = list(client.models.list())
    
    import sys
    # Fix encoding for Windows console
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
    
    print(f"\n=== Available Gemini Models ({len(models)} total) ===\n")
    
    # Filter and categorize models
    text_models = [m for m in models if "gemini" in m.name.lower() and "generate" not in m.name.lower() and "embedding" not in m.name.lower()]
    embedding_models = [m for m in models if "embedding" in m.name.lower()]
    other_models = [m for m in models if m not in text_models and m not in embedding_models]
    
    print("TEXT GENERATION MODELS (for compliance auditing):")
    print("-" * 60)
    for model in sorted(text_models, key=lambda x: x.name):
        # Highlight recommended models
        if "gemini-2.5-pro" in model.name and "preview" not in model.name:
            print(f"  [RECOMMENDED] {model.name}")
        elif "gemini-2.5-flash" in model.name and "preview" not in model.name:
            print(f"  [FAST] {model.name}")
        elif "gemini-3-pro" in model.name:
            print(f"  [PREVIEW] {model.name}")
        else:
            print(f"  - {model.name}")
    
    if embedding_models:
        print(f"\nEMBEDDING MODELS ({len(embedding_models)}):")
        print("-" * 60)
        for model in sorted(embedding_models, key=lambda x: x.name)[:5]:
            print(f"  - {model.name}")
        if len(embedding_models) > 5:
            print(f"  ... and {len(embedding_models) - 5} more")
    
    print(f"\nTIP: Set GEMINI_MODEL in your .env file to one of the models above.")
    print(f"   Example: GEMINI_MODEL=models/gemini-2.5-pro\n")
    
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install -r requirements.txt")
except Exception as exc:
    print(f"ERROR: {exc}")
