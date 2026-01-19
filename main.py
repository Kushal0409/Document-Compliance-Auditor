import os
from typing import List, Tuple

from dotenv import load_dotenv

from auditor import call_llm_with_gemini
from doc_utils import extract_text_from_path, is_supported_file_type


REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference_docs")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")


def prompt_for_document(prompt_label: str) -> str:
    print(f"\n=== {prompt_label} ===")
    print("Choose input method:")
    print("1) Paste text directly")
    print("2) Provide path to a file (.txt, .pdf, .docx)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\nPaste your text below. End with a single line containing only 'EOF':\n")
        lines: List[str] = []
        while True:
            line = input()
            if line.strip() == "EOF":
                break
            lines.append(line)
        return "\n".join(lines).strip()
    elif choice == "2":
        path = input("Enter file path: ").strip().strip('"')
        # Check file type for user-provided document
        is_valid, warning_msg = is_supported_file_type(path)
        if not is_valid:
            print(f"\n{warning_msg}")
            print("Please try again with a supported file type (.txt, .pdf, or .docx).\n")
            return prompt_for_document(prompt_label)
        return extract_text_from_path(path, show_warning=True)
    else:
        print("Invalid choice, please try again.")
        return prompt_for_document(prompt_label)


def prompt_for_rag_contexts() -> List[Tuple[str, str]]:
    print("\n=== RAG CONTEXT DOCUMENTS (Acts / Policies / Regulations) ===")
    contexts: List[Tuple[str, str]] = []

    while True:
        print("\nAdd a new RAG context?")
        print("1) Yes - paste text")
        print("2) Yes - from file (.txt, .pdf, .docx)")
        print("3) No more RAG documents (continue)")
        choice = input("Enter 1, 2, or 3: ").strip()

        if choice == "3":
            break

        label = input("Enter a short label for this RAG source (e.g., 'IT Act Sec 43A'): ").strip()

        if choice == "1":
            print("\nPaste the RAG context text. End with a single line containing only 'EOF':\n")
            lines: List[str] = []
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            contexts.append((label, "\n".join(lines).strip()))
        elif choice == "2":
            path = input("Enter file path: ").strip().strip('"')
            # Check file type for user-provided reference file
            is_valid, warning_msg = is_supported_file_type(path)
            if not is_valid:
                print(f"\n{warning_msg}")
                print("Skipping this file. Please try again with a supported file type (.txt, .pdf, or .docx).\n")
                continue
            try:
                contexts.append((label, extract_text_from_path(path, show_warning=True)))
            except Exception as exc:
                print(f"⚠️ WARNING: Error reading file '{path}': {exc}")
                print("Skipping this file.\n")
        else:
            print("Invalid choice. Please try again.")

    # If user did not provide any RAG contexts, fall back to reference_docs folder.
    # Silently skip unsupported files in reference_docs (no warnings).
    if not contexts:
        if os.path.isdir(REFERENCE_DIR):
            for name in os.listdir(REFERENCE_DIR):
                full_path = os.path.join(REFERENCE_DIR, name)
                if not os.path.isfile(full_path):
                    continue
                try:
                    # Silently skip unsupported files in reference_docs folder
                    text = extract_text_from_path(full_path, show_warning=False)
                except Exception:
                    continue  # Silently skip unsupported or unreadable files
                contexts.append((name, text))

    return contexts


def configure_gemini_model():
    # override=True ensures new keys in .env replace any old environment values
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Please set it in a .env file or environment variable.")

    # If GOOGLE_API_KEY is set in the environment, google-genai may prefer it.
    # We explicitly remove it so GEMINI_API_KEY is always used.
    os.environ.pop("GOOGLE_API_KEY", None)

    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai not installed. Run: pip install -r requirements.txt") from exc

    client = genai.Client(api_key=api_key)
    return client


def main():
    print("=== AI Legal Compliance Auditor & Redrafter (Production - Vector DB + Agentic AI) ===")
    
    # Check database health
    try:
        from database_manager import get_database_manager
        db_manager = get_database_manager()
        health = db_manager.check_database_health()
        
        if health["status"] == "empty":
            print("\n⚠️  Warning: Vector database is empty!")
            print("   Run 'python initialize_database.py' to index reference documents.")
            print("   The system will use provided RAG contexts as fallback.\n")
        elif health["status"] == "healthy":
            print(f"✅ Vector database ready ({health['document_count']} document chunks indexed)\n")
    except Exception as e:
        print(f"⚠️  Warning: Could not check database: {e}")
        print("   The system will use provided RAG contexts.\n")

    user_document = prompt_for_document("USER DOCUMENT TO BE AUDITED")
    rag_contexts = prompt_for_rag_contexts()

    print("\nRunning compliance audit with Gemini... this may take a moment.\n")
    
    # Check if user wants agentic mode
    use_agentic = os.getenv("USE_AGENTIC", "true").lower() == "true"
    use_vector_search = os.getenv("USE_VECTOR_SEARCH", "true").lower() == "true"
    
    if use_agentic:
        print("🤖 Agentic AI mode: ENABLED (multi-step reasoning with tools)")
    else:
        print("📝 Standard mode: Single-shot LLM call")
    
    if use_vector_search:
        print("🔍 Vector search: ENABLED (semantic search from database)")
    else:
        print("📄 Text search: Using provided documents only")
    
    client = configure_gemini_model()
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    try:
        result = call_llm_with_gemini(client, model_name, user_document, rag_contexts, use_agentic=use_agentic)
    except Exception as exc:
        print(f"\nERROR while calling Gemini model '{model_name}': {exc}\n")
        if "NOT_FOUND" in str(exc) or "404" in str(exc) or "not found" in str(exc).lower():
            print("❌ Model not found! The model name you specified doesn't exist.\n")
            print("To see available models, run:")
            print("  python list_gemini_models.py\n")
            print("Recommended models:")
            print("  - models/gemini-2.5-pro (stable, best quality)")
            print("  - models/gemini-2.5-flash (faster, cheaper)")
            print("  - models/gemini-3-pro-preview (latest preview)\n")
            print("Set GEMINI_MODEL in your .env file to one of the available models.\n")
        elif "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc) or "quota" in str(exc).lower():
            print("Your Gemini API key currently has no available quota for this model.\n")
            print("- Enable billing / increase quota for your Gemini API project, or")
            print("- Switch GEMINI_MODEL in your .env to a model you have quota for.\n")
        else:
            print("If this is a model-not-found error, run 'python list_gemini_models.py' to see available models.\n")
        raise

    # The model already includes the humanized summary as part of the JSON.
    # We simply print the JSON and then the Markdown summary.
    print("\n=== RAW JSON RESULT ===")
    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))

    human_md = result.get("humanized_summary_markdown", "")
    if human_md:
        print("\n---\n")
        print(human_md)

    # Optional: ask user whether to save outputs
    save_choice = input("\nDo you want to save the JSON and Markdown to files? (y/n): ").strip().lower()
    if save_choice == "y":
        json_path = input("Enter path for JSON output file (e.g., audit_result.json): ").strip()
        md_path = input("Enter path for Markdown report file (e.g., audit_report.md): ").strip()

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(result, jf, indent=2, ensure_ascii=False)

        with open(md_path, "w", encoding="utf-8") as mf:
            mf.write(human_md)

        print(f"\nSaved JSON to {json_path}")
        print(f"Saved Markdown report to {md_path}")


if __name__ == "__main__":
    main()

