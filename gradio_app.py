import os
import json
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv

from auditor import call_llm_with_gemini
from doc_utils import extract_text_from_path, is_supported_file_type


REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference_docs")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-pro")


def configure_gemini_model():
    """
    Configure and return a Gemini model instance using GEMINI_API_KEY.
    """
    # override=True ensures new keys in .env replace any old environment values
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY")
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


def run_audit(
    user_document_text: str,
    user_document_file: str,
    rag_text: str,
    rag_files: List[str],
) -> Tuple[str, str]:
    """
    Gradio callback to run the compliance audit.

    Returns:
    - JSON string (pretty-printed)
    - Markdown report
    """
    # Resolve main user document text:
    # If a file is uploaded, it overrides the textbox content.
    user_document_text = (user_document_text or "").strip()
    user_document_file = (user_document_file or "").strip()
    
    warnings = []

    if user_document_file:
        # Check file type for user-uploaded document
        is_valid, warning_msg = is_supported_file_type(user_document_file)
        if not is_valid:
            warnings.append(warning_msg)
            return "\n".join(warnings) + "\n\nERROR: Please upload a supported file type (.txt, .pdf, or .docx) or paste text instead.", ""
        
        try:
            user_document = extract_text_from_path(user_document_file, show_warning=True)
        except Exception as exc:
            return f"ERROR reading uploaded user document: {exc}", ""
    else:
        user_document = user_document_text

    if not user_document.strip():
        return "ERROR: Please provide a user document to be audited.", ""

    rag_contexts: List[Tuple[str, str]] = []

    # Check whether user has provided any reference documents (text or files).
    user_has_refs = bool(rag_text and rag_text.strip()) or bool(rag_files)

    if user_has_refs:
        # Use ONLY the references explicitly provided by the user.
        if rag_text and rag_text.strip():
            rag_contexts.append(("Pasted RAG Text", rag_text.strip()))

        if rag_files:
            for path in rag_files:
                # Check file type for user-uploaded reference files
                is_valid, warning_msg = is_supported_file_type(path)
                if not is_valid:
                    warnings.append(warning_msg)
                    continue  # Skip this file
                
                try:
                    label = os.path.basename(path)
                    content = extract_text_from_path(path, show_warning=True)
                    rag_contexts.append((label, content))
                except Exception as exc:
                    warnings.append(f"⚠️ WARNING: Error reading file '{os.path.basename(path)}': {exc}")
    else:
        # No user-provided reference docs: fall back to reference_docs directory.
        # Silently skip unsupported files in reference_docs (no warnings).
        if os.path.isdir(REFERENCE_DIR):
            for name in os.listdir(REFERENCE_DIR):
                full_path = os.path.join(REFERENCE_DIR, name)
                if not os.path.isfile(full_path):
                    continue
                try:
                    # Silently skip unsupported files in reference_docs folder
                    content = extract_text_from_path(full_path, show_warning=False)
                except Exception:
                    continue  # Silently skip unsupported or unreadable files
                rag_contexts.append((name, content))

    # Show warnings if any (before running audit)
    warning_text = ""
    if warnings:
        warning_text = "\n\n".join(warnings) + "\n\n" + "="*50 + "\n\n"
    
    # Check database health (silently)
    db_status = ""
    try:
        from database_manager import get_database_manager
        db_manager = get_database_manager()
        health = db_manager.check_database_health()
        if health["status"] == "healthy" and health["document_count"] > 0:
            db_status = f"✅ Vector DB: {health['document_count']} chunks indexed | "
        elif health["status"] == "empty":
            db_status = "⚠️  Vector DB empty - using provided documents | "
    except:
        pass
    
    try:
        client = configure_gemini_model()
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        use_agentic = os.getenv("USE_AGENTIC", "true").lower() == "true"
        result = call_llm_with_gemini(client, model_name, user_document, rag_contexts, use_agentic=use_agentic)
    except Exception as exc:
        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        msg = f"ERROR while calling Gemini model '{model_name}': {exc}"
        
        if "NOT_FOUND" in str(exc) or "404" in str(exc) or "not found" in str(exc).lower():
            msg += (
                "\n\n❌ **Model not found!** The model name you specified doesn't exist."
                "\n\n**To see available models, run:**"
                "\n```bash"
                "\npython list_gemini_models.py"
                "\n```"
                "\n\n**Recommended models:**"
                "\n- `models/gemini-2.5-pro` (stable, best quality)"
                "\n- `models/gemini-2.5-flash` (faster, cheaper)"
                "\n- `models/gemini-3-pro-preview` (latest preview)"
                "\n\nSet `GEMINI_MODEL` in your `.env` file to one of the available models."
            )
        elif "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc) or "quota" in str(exc).lower():
            msg += (
                "\n\nYour Gemini API key currently has **no available quota** for this model."
                "\n- If you intended to use a paid plan, enable billing for the Gemini API project."
                "\n- Or switch to a different available model in your `.env` via `GEMINI_MODEL=`."
                "\n\n**Tip:** Run `python list_gemini_models.py` to see available models."
            )
        else:
            msg += (
                "\n\n**Tip:** Run `python list_gemini_models.py` to see available models."
            )
        return warning_text + msg, ""

    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    human_md = result.get("humanized_summary_markdown", "")
    
    # Prepend warnings to JSON output if any
    if warnings:
        json_str = warning_text + json_str

    return json_str, human_md


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AI Legal Compliance Auditor & Redrafter") as demo:
        gr.Markdown(
            """
            ## AI Legal Compliance Auditor & Redrafter (Production)

            **🚀 Production Features:**
            - **Vector Database**: Semantic search for regulations
            - **Agentic AI**: Multi-step reasoning with tools
            - **Hybrid Search**: Semantic + keyword search

            **Supported file types:** `.txt`, `.pdf`, `.docx` only
            
            - Upload or paste the **document to be audited** (Word, PDF, or text file).
            - Optionally paste additional **Acts / policies / regulations** in the text box.
            - Or upload one or more **reference documents** (.txt, .pdf, or .docx files).
            - If you don't provide reference documents, the system will use **semantic search** from the vector database.
            - Click **Run audit** to get:
              - A structured **JSON compliance audit**, and
              - A **Markdown Compliance Audit Report** ready to share.
            
            **💡 Tip**: Run `python initialize_database.py` to index reference documents for semantic search.
            """
        )

        with gr.Row():
            user_document_file = gr.File(
                label="Upload user document (.txt, .pdf, .docx) - optional",
                file_count="single",
                type="filepath",
            )

        with gr.Row():
            user_document = gr.Textbox(
                label="User document to be audited (paste text if no file uploaded)",
                placeholder="Paste the full text of the contract / policy / handbook here...",
                lines=20,
            )

        with gr.Row():
            rag_text = gr.Textbox(
                label="RAG context (Acts / Policies / Regulations) - optional",
                placeholder="Paste any legal / regulatory excerpts here...",
                lines=12,
            )

        with gr.Row():
            rag_files = gr.Files(
                label="Reference documents (optional, .txt / .pdf / .docx)",
                file_count="multiple",
                type="filepath",
            )

        run_button = gr.Button("Run audit", variant="primary")

        with gr.Row():
            json_output = gr.Textbox(
                label="JSON audit result",
                lines=20,
            )

        md_output = gr.Markdown(label="Compliance Audit Report (Markdown)")

        run_button.click(
            fn=run_audit,
            inputs=[user_document, user_document_file, rag_text, rag_files],
            outputs=[json_output, md_output],
        )

    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch()

