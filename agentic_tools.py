"""
Agentic Tools for the Legal Compliance Auditor.
These tools can be called by the AI agent to perform specific tasks.
"""
from typing import List, Tuple, Dict, Any, Optional
import re
from langchain.tools import tool


@tool
def search_regulations(query: str, rag_contexts: List[Tuple[str, str]] = None) -> str:
    """
    Search through regulation documents using semantic search from vector database.
    
    Args:
        query: What to search for (e.g., "data protection", "privacy policy", "signature requirement")
        rag_contexts: Optional list of (label, text) tuples (for backward compatibility)
    
    Returns:
        Relevant excerpts from regulations that match the query
    """
    try:
        # Use vector database for semantic search
        from retrieval import get_retrieval_system
        
        retrieval_system = get_retrieval_system()
        
        # Perform semantic search
        results = retrieval_system.semantic_search(query, k=5)
        
        if not results:
            # Fallback to provided rag_contexts if available
            if rag_contexts:
                return _fallback_keyword_search(query, rag_contexts)
            return f"No matches found for '{query}' in regulation database."
        
        # Format results
        formatted_results = []
        seen_sources = set()
        
        for doc in results:
            source = doc.metadata.get("source", "Unknown")
            if source in seen_sources:
                continue
            seen_sources.add(source)
            
            excerpt = f"From {source}:\n{doc.page_content[:500]}..."
            formatted_results.append(excerpt)
        
        return "\n\n".join(formatted_results)
    
    except Exception as e:
        # Fallback to keyword search if vector DB fails
        if rag_contexts:
            return _fallback_keyword_search(query, rag_contexts)
        return f"Search failed: {str(e)}. No regulation documents available."


def _fallback_keyword_search(query: str, rag_contexts: List[Tuple[str, str]]) -> str:
    """Fallback keyword search when vector DB is not available."""
    if not rag_contexts:
        return "No regulation documents available to search."
    
    query_lower = query.lower()
    results = []
    
    for label, text in rag_contexts:
        text_lower = text.lower()
        if query_lower in text_lower:
            sentences = re.split(r'[.!?]\s+', text)
            relevant_sentences = [
                s.strip() for s in sentences 
                if query_lower in s.lower()
            ]
            if relevant_sentences:
                excerpt = f"From {label}:\n" + "\n".join(relevant_sentences[:3])
                results.append(excerpt)
    
    if not results:
        return f"No matches found for '{query}' in regulation documents."
    
    return "\n\n".join(results)


@tool
def analyze_document_structure(document: str) -> Dict[str, Any]:
    """
    Analyze the structural elements of a document (formatting, numbering, sections).
    
    Args:
        document: The document text to analyze
    
    Returns:
        Dictionary with structure analysis (tone, terminology, formatting style)
    """
    analysis = {
        "tone": "Unknown",
        "terminology": [],
        "formatting_style": "Unknown",
        "section_markers": []
    }
    
    # Detect tone indicators
    formal_indicators = ["hereby", "whereas", "pursuant", "hereinafter", "party of the first part"]
    casual_indicators = ["we", "you", "your", "our", "let's"]
    
    formal_count = sum(1 for word in formal_indicators if word.lower() in document.lower())
    casual_count = sum(1 for word in casual_indicators if word.lower() in document.lower())
    
    if formal_count > casual_count:
        analysis["tone"] = "Strict Legal"
    elif casual_count > formal_count:
        analysis["tone"] = "Friendly/Casual"
    else:
        analysis["tone"] = "Corporate Professional"
    
    # Detect terminology patterns
    if "vendor" in document.lower() or "client" in document.lower():
        analysis["terminology"].append("Vendor/Client")
    if "company" in document.lower() and "employee" in document.lower():
        analysis["terminology"].append("Company/Employee")
    if "party a" in document.lower() or "party b" in document.lower():
        analysis["terminology"].append("Party A/Party B")
    
    # Detect formatting style
    if re.search(r'\b[IVX]+\.', document):
        analysis["formatting_style"] = "Roman Numerals"
    elif re.search(r'\d+\.\d+', document):
        analysis["formatting_style"] = "Decimal Numbering (1.1, 1.2)"
    elif re.search(r'^\s*[-•*]\s+', document, re.MULTILINE):
        analysis["formatting_style"] = "Bullet Points"
    else:
        analysis["formatting_style"] = "Paragraph Style"
    
    return analysis


@tool
def check_missing_fields(document: str, required_fields: List[str]) -> Dict[str, Any]:
    """
    Check if required fields are present in the document.
    
    Args:
        document: The document text to check
        required_fields: List of fields to check for (e.g., ["Date", "CIN", "Signature", "Jurisdiction"])
    
    Returns:
        Dictionary with missing fields and their status
    """
    document_lower = document.lower()
    results = {
        "missing": [],
        "found": [],
        "partial": []
    }
    
    field_patterns = {
        "Date": [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'date', r'dated'],
        "CIN": [r'cin', r'corporate.*identification.*number', r'c\.i\.n\.'],
        "Signature": [r'signature', r'signed', r'sign'],
        "Jurisdiction": [r'jurisdiction', r'governed.*by', r'law.*of'],
        "Company Name": [r'company.*name', r'incorporated', r'ltd', r'llc'],
        "Address": [r'address', r'located.*at', r'residing.*at']
    }
    
    for field in required_fields:
        found = False
        if field in field_patterns:
            for pattern in field_patterns[field]:
                if re.search(pattern, document_lower, re.IGNORECASE):
                    found = True
                    break
        
        if found:
            results["found"].append(field)
        else:
            results["missing"].append(field)
    
    return results


@tool
def compare_with_regulation(document_clause: str, regulation_text: str) -> Dict[str, Any]:
    """
    Compare a document clause against regulation text to check compliance.
    
    Args:
        document_clause: The clause from the user's document
        regulation_text: The relevant regulation text
    
    Returns:
        Dictionary with compliance analysis
    """
    # Simple keyword-based comparison (could be enhanced with semantic similarity)
    regulation_lower = regulation_text.lower()
    clause_lower = document_clause.lower()
    
    # Extract key terms from regulation
    regulation_keywords = set(re.findall(r'\b\w{4,}\b', regulation_lower))
    clause_keywords = set(re.findall(r'\b\w{4,}\b', clause_lower))
    
    overlap = regulation_keywords.intersection(clause_keywords)
    overlap_ratio = len(overlap) / len(regulation_keywords) if regulation_keywords else 0
    
    compliance_status = "COMPLIANT" if overlap_ratio > 0.3 else "NON_COMPLIANT"
    
    return {
        "status": compliance_status,
        "overlap_ratio": overlap_ratio,
        "shared_keywords": list(overlap)[:10],
        "analysis": f"Clause has {overlap_ratio:.1%} keyword overlap with regulation."
    }


@tool
def generate_style_adapted_clause(
    regulation_requirement: str,
    style_profile: Dict[str, Any],
    document_context: str
) -> str:
    """
    Generate a clause that matches the document's style while meeting regulatory requirements.
    
    Args:
        regulation_requirement: What the regulation requires
        style_profile: The style analysis of the document (from analyze_document_structure)
        document_context: Relevant context from the user's document
    
    Returns:
        A style-adapted clause that meets the requirement
    """
    tone = style_profile.get("tone", "Corporate Professional")
    terminology = style_profile.get("terminology", [])
    formatting = style_profile.get("formatting_style", "Paragraph Style")
    
    # This is a placeholder - in a real system, this would call an LLM
    # For now, return a template-based response
    clause_template = f"""
Based on the regulation requirement: {regulation_requirement}

Generated clause (adapted to {tone} tone, using {terminology} terminology):
[This would be generated by an LLM call in the full implementation]
"""
    
    return clause_template.strip()


@tool
def calculate_compliance_score(
    missing_fields: List[str],
    non_compliant_clauses: List[str],
    total_requirements: int
) -> int:
    """
    Calculate a compliance risk score (0-100, where 100 is perfectly compliant).
    
    Args:
        missing_fields: List of missing required fields
        non_compliant_clauses: List of non-compliant clause descriptions
        total_requirements: Total number of compliance requirements checked
    
    Returns:
        Compliance score from 0-100
    """
    if total_requirements == 0:
        return 100
    
    issues = len(missing_fields) + len(non_compliant_clauses)
    score = max(0, 100 - (issues / total_requirements) * 100)
    return int(score)
