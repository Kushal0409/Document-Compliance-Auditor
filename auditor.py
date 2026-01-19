import json
from typing import List, Tuple, Dict, Any


SYSTEM_PROMPT = """
You are an advanced AI Legal Compliance Agent specialized in helping MSMEs (Micro, Small, and Medium Enterprises).
Your function is twofold:
1. The Auditor: Rigorous, fact-based checking of documents against specific regulations.
2. The Ghostwriter: A chameleon-like editor capable of writing new clauses that perfectly match the user's specific writing style (tone, vocabulary, and structure).

You receive two inputs:
1. {{USER_DOCUMENT}}: The raw text of the document needing audit.
2. {{RAG_CONTEXT}}: Relevant excerpts from Acts, Policies, or Regulations.

You must perform three phases:

PHASE 1: COMPLIANCE ANALYSIS & SCORING
- Identify specific missing fields (e.g., Dates, CIN, Signatures, Jurisdiction).
- Perform gap analysis: clauses required by law that are missing from the document.
- Identify non-compliance: clauses that exist but violate the statutes in {{RAG_CONTEXT}}.
- Produce a COMPLIANCE RISK SCORE from 0–100 (100 = perfectly compliant).

PHASE 2: CONTEXTUAL STYLE PROFILING
- Analyze the "Voice" of the {{USER_DOCUMENT}}:
  - Tone (e.g., "Strict Legal", "Corporate Professional", "Friendly/Casual").
  - Terminology (e.g., "Vendor/Client", "Company/Employee", "Party A/Party B").
  - Structure (e.g., Roman numerals I, II; decimals 1.1, 1.2; bullet points).
- This profile must be mimicked in Phase 3.

PHASE 3: AGENTIC REDRAFTING (HUMANIZATION LAYER)
- For every gap or error identified, generate a Correction.
- RULE 1 (NO COPY-PASTE): Do NOT copy language verbatim from {{RAG_CONTEXT}}.
- RULE 2 (STYLE ADAPTATION): Redraft in the exact tone, terminology, and structure of {{USER_DOCUMENT}}.
- RULE 3 (CONTEXTUAL FILLING): If the document mentions specific company names or roles, reuse them; avoid placeholders if the name is known.

OUTPUT FORMAT (STRICT)
Return a single JSON object with this structure, and nothing else:
{
  "compliance_score": integer,
  "style_profile": {
    "tone": "string",
    "detected_terminology": "string"
  },
  "audit_findings": [
    {
      "id": 1,
      "type": "MISSING_CLAUSE" | "NON_COMPLIANT_CLAUSE" | "MISSING_FIELD",
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "regulation_reference": "Name of Act/Section from {{RAG_CONTEXT}}",
      "issue_description": "Brief explanation of the gap.",
      "original_text": "Text causing the issue (or null if missing)",
      "suggested_redraft": "The specific, humanized text to insert/replace.",
      "redraft_reasoning": "Why you worded it this way based on the style profile."
    }
  ],
  "humanized_summary_markdown": "A user-friendly Compliance Audit Report in Markdown, following the template below."
}

The field "humanized_summary_markdown" must follow exactly this template, filled in:

## Compliance Audit Report
**Risk Score:** [Score]/100

### 🚨 Critical Issues & Fixes
**1. [Issue Name]**
* **The Problem:** [Simple explanation of why this matters for an MSME, avoiding jargon].
* **The Fix:** I have drafted a new clause for you that matches your document's style.
    > *[Insert suggested_redraft here]*
* **Why this wording?**: [Explain how you adapted the law to their document style].

For multiple issues, continue the numbering 2, 3, etc.

IMPORTANT:
- The overall response MUST be valid JSON.
- Do not include Markdown fences like ```json.
- Do not include any text before or after the JSON.
- Be concise but specific and useful for MSMEs.
"""


def build_user_content(user_document: str, rag_contexts: List[Tuple[str, str]]) -> str:
    """
    Build a single user content string that clearly separates:
    - The user document
    - One or more RAG context documents (with labels)
    """
    parts: List[str] = []
    parts.append("{{USER_DOCUMENT}}:\n")
    parts.append(user_document.strip())
    parts.append("\n\n{{RAG_CONTEXT}}:\n")

    if len(rag_contexts) == 0:
        parts.append("[NO RAG CONTEXT PROVIDED]\n")
    else:
        for idx, (label, text) in enumerate(rag_contexts, start=1):
            parts.append(f"--- RAG SOURCE {idx}: {label} ---\n")
            parts.append(text.strip())
            parts.append("\n\n")

    return "".join(parts).strip()


def call_llm_with_gemini(
    client,
    model_name: str,
    user_document: str,
    rag_contexts: List[Tuple[str, str]],
    use_agentic: bool = True,
) -> Dict[str, Any]:
    """
    Invoke the Gemini model with the system prompt and user content.
    
    Now supports both agentic and non-agentic modes.

    Parameters
    ----------
    client: a configured google-genai client (google.genai.Client)
    model_name: model id (e.g., gemini-3.0-pro)
    user_document: main document string
    rag_contexts: list of (label, text) tuples for reference documents
    use_agentic: If True, uses agentic AI system with tools. If False, uses single-shot LLM call.

    Returns
    -------
    Parsed JSON (Python dict) according to the required schema.
    """
    
    if use_agentic:
        try:
            from agentic_auditor import call_llm_with_agentic_system
            print("\n🤖 Using AGENTIC AI system with multi-step reasoning and tools...")
            return call_llm_with_agentic_system(client, model_name, user_document, rag_contexts)
        except ImportError as e:
            print(f".")
        except Exception as e:
            print(f".")
    
    # Standard single-shot mode 
    user_content = build_user_content(user_document, rag_contexts)

    # google-genai SDK
    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise RuntimeError(
            "google-genai is not installed. Run: pip install -r requirements.txt"
        ) from exc

    response = client.models.generate_content(
        model=model_name,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            max_output_tokens=4096,
            response_mime_type="application/json",
        ),
    )

    
    raw_text = (response.text or "").strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        
        repaired = raw_text

        # 1) Extract JSON substring if extraneous text surrounds it
        first = repaired.find('{')
        last = repaired.rfind('}')
        if first != -1 and last != -1 and last > first:
            repaired = repaired[first : last + 1]
        elif first != -1:
            repaired = repaired[first:]

        # 2) If number of double quotes is odd, append a closing quote (common when truncated)
        if repaired.count('"') % 2 == 1:
            repaired = repaired + '"'

        # 3) Balance braces by appending missing closing braces
        opens = repaired.count('{')
        closes = repaired.count('}')
        if closes < opens:
            repaired = repaired + ('}' * (opens - closes))

        # Try parsing the repaired text
        try:
            result = json.loads(repaired)
            return result
        except json.JSONDecodeError:
            # As a last resort, ask the model to reformat its previous (malformed) output into valid JSON.
            try:
                repair_prompt = (
                    "The response you previously returned was intended to be STRICTLY valid JSON following a known schema, "
                    "but it was malformed. Please re-output ONLY valid JSON (no commentary). Here is the exact previous output to fix:\n\n" + raw_text
                )

                repair_resp = client.models.generate_content(
                    model=model_name,
                    contents=repair_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are a utility that fixes malformed JSON responses. The user will supply a malformed JSON string; "
                            "your job is to output only the corrected, valid JSON and nothing else."
                        ),
                        temperature=0.0,
                        top_p=0.0,
                        max_output_tokens=1024,
                        response_mime_type="application/json",
                    ),
                )

                repaired2 = (repair_resp.text or "").strip()
                result = json.loads(repaired2)
                return result
            except Exception as exc2:
               
                raise ValueError(
                    f"Model output was not valid JSON: {exc}\n\nRaw output:\n{raw_text}\n\nAttempted simple repair (trim/balance/quote):\n{repaired}\n\nRepair attempt error: {exc2}"
                ) from exc

    return result

