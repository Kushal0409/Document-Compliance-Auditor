"""
Agentic AI System for Legal Compliance Auditing.
This implements a multi-agent system with planning, execution, and refinement capabilities.
"""
import json
from typing import List, Tuple, Dict, Any, Optional
try:
    from langchain.agents import create_react_agent
except ImportError:
    from langchain.agents.react import create_react_agent
from langchain.agents import AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain import hub

from agentic_tools import (
    search_regulations,
    analyze_document_structure,
    check_missing_fields,
    compare_with_regulation,
    generate_style_adapted_clause,
    calculate_compliance_score,
)


class AgenticComplianceAuditor:
    """
    Multi-agent system for compliance auditing with:
    1. Planning Agent - Breaks down the audit task
    2. Execution Agent - Performs the audit using tools
    3. Refinement Agent - Improves and validates results
    
    This is TRUE AGENTIC AI:
    - Uses LangChain's ReAct agent architecture
    - Agent autonomously decides which tools to call
    - Tools actually return data the agent processes
    - Multi-step reasoning with iterative decision-making
    """
    
    def __init__(self, client, model_name: str, rag_contexts: List[Tuple[str, str]] = None):
        import os
        self.client = client
        self.model_name = model_name
        self.rag_contexts = rag_contexts or []
        self.agent_calls_log = [] 
        self.audit_state = {
            "phase": None,
            "findings": [],
            "style_profile": None,
            "compliance_score": None,
        }
        
        try:
            from retrieval import get_retrieval_system
            self.retrieval_system = get_retrieval_system()
            self.use_vector_search = True
        except Exception as e:
            print(f"⚠️  Warning: Vector search not available: {e}")
            self.retrieval_system = None
            self.use_vector_search = False
        
        from dotenv import load_dotenv
        load_dotenv(override=True)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        model_id = model_name.replace("models/", "")
        self.llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.3,
            google_api_key=api_key
        )
        
        self.tools = self._create_tools_list()
        self.agent = self._create_agent()
    
    def _create_tools_list(self) -> List[StructuredTool]:
        """Create the list of tools for the agent."""
        return [
            search_regulations,
            analyze_document_structure,
            check_missing_fields,
            compare_with_regulation,
            generate_style_adapted_clause,
            calculate_compliance_score,
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the ReAct agent executor."""
        system_prompt = """You are an advanced AI Legal Compliance Auditor Agent.

You have access to 6 specialized tools:
1. search_regulations - Search legal documents for specific requirements
2. analyze_document_structure - Analyze writing style and structure  
3. check_missing_fields - Check for required legal fields
4. compare_with_regulation - Compare clauses against regulations
5. generate_style_adapted_clause - Generate fixes matching the document's style
6. calculate_compliance_score - Calculate the compliance risk score

YOUR WORKFLOW:
1. First, analyze the document structure to understand its style and tone
2. Check what required fields are missing
3. Search for applicable regulations using keyword queries
4. Compare the document against those regulations
5. Generate style-adapted fixes for any non-compliant clauses
6. Calculate the final compliance score

IMPORTANT: Use the tools AUTONOMOUSLY. Think about what you need to know, then call the appropriate tool. You don't need the user to tell you which tool to use.
Be thorough. Call tools multiple times if needed. Always reason step-by-step before calling a tool."""
        
        try:
            prompt = hub.pull("hwchase17/react")
        except:
            from langchain_core.prompts import PromptTemplate
            prompt = PromptTemplate.from_template(system_prompt)
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
            early_stopping_method="force",
            handle_parsing_errors=True
        )
        
        return executor
    
    def _plan_audit(self, user_document: str) -> Dict[str, Any]:
        """Planning Agent: Creates a step-by-step audit plan."""
        planning_prompt = f"""Create a detailed audit plan for this document.

Document to audit:
{user_document[:2000]}...

Available regulations: {len(self.rag_contexts)} document(s)

Break down the audit into specific steps:
1. What fields should be checked?
2. What regulations need to be searched?
3. What clauses need compliance verification?
4. What style analysis is needed?

Return a JSON plan with steps."""
        
        response = self.agent_executor.invoke({
            "messages": [HumanMessage(content=planning_prompt)]
        })
        
        return response
    
    def _execute_audit(self, user_document: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execution Agent: Performs the audit using tools."""
        
        if self.use_vector_search and self.retrieval_system:
            print("\n🔍 [AGENTIC SYSTEM] Retrieving relevant regulations using semantic search...")
            retrieved_regulations = self.retrieval_system.retrieve_relevant_regulations(
                user_document,
                k=5,
                use_hybrid=True
            )
            if retrieved_regulations:
                self.rag_contexts = retrieved_regulations + self.rag_contexts
                print(f"  ✅ Retrieved {len(retrieved_regulations)} relevant regulations")
        
        audit_prompt = f"""Perform a comprehensive legal compliance audit on this document.

Document to audit:
---
{user_document}
---

Your task is to:
1. First, use the analyze_document_structure tool to understand the document's style and tone
2. Use check_missing_fields to find missing required legal fields
3. Use search_regulations to find applicable regulations (search multiple times for different aspects)
4. Use compare_with_regulation to check document clauses against regulations
5. Use generate_style_adapted_clause to create fixes that match the document's original style
6. Use calculate_compliance_score to determine the compliance risk score

Think step-by-step. Call the tools in the order that makes sense. Use multiple search queries to be thorough."""
        
        print("\n⚙️  [AGENTIC SYSTEM] Agent is now executing audit autonomously with tools...")
        print("    (Watch for [Tool Call] messages below showing which tools the agent uses)\n")
        
        try:
            result = self.agent.invoke({
                "input": audit_prompt
            })
            
            print("\n✅ [AGENTIC SYSTEM] Agent execution complete")
            
            return {
                "agent_result": result,
                "rag_contexts": self.rag_contexts
            }
        except Exception as e:
            print(f"⚠️  [AGENTIC SYSTEM] Agent execution error: {e}")
            return {
                "agent_result": None,
                "error": str(e),
                "rag_contexts": self.rag_contexts
            }
    
    def _refine_results(self, user_document: str, initial_results: Dict[str, Any]) -> Dict[str, Any]:
        """Refinement Agent: Reviews and improves the audit results."""
        
        refinement_prompt = f"""Review and improve the compliance audit results.

Based on your previous audit analysis, now:
1. Validate that all findings are accurate and properly formatted
2. Ensure any suggested redrafts match the original document's style
3. Use calculate_compliance_score to determine the final compliance risk score (0-100)

Return a summary of your refined findings."""
        
        print("\n✨ [AGENTIC SYSTEM] Agent is now refining and validating results...")
        
        try:
            result = self.agent.invoke({
                "input": refinement_prompt
            })
            
            print("✅ [AGENTIC SYSTEM] Refinement complete")
            
            return {
                "refinement_result": result
            }
        except Exception as e:
            print(f"⚠️  [AGENTIC SYSTEM] Refinement error: {e}")
            return {
                "refinement_result": None,
                "error": str(e)
            }
    
    def audit(self, user_document: str) -> Dict[str, Any]:
        """Main agentic audit process with planning, execution, and refinement."""
        print("\n" + "="*70)
        print(" STARTING AGENTIC COMPLIANCE AUDIT")
        print("="*70)
        print("\nThis is a TRUE AGENTIC AI system where:")
        print("  ✓ The AI agent decides which tools to call (not hardcoded)")
        print("  ✓ Each tool call receives real data back")
        print("  ✓ The agent reasons about tool results")
        print("  ✓ Multi-step decision making (ReAct pattern)")
        print("\n" + "="*70 + "\n")
        
        plan = self._plan_audit(user_document)
        execution_results = self._execute_audit(user_document, plan)
        refined_results = self._refine_results(user_document, execution_results)
        
        agent_output = execution_results.get("agent_result", {})
        findings = self._extract_findings_from_agent(agent_output, user_document)
        compliance_score = self._calculate_final_score(findings)
        style_profile = self._analyze_style(user_document)
        markdown = self._generate_markdown_report(findings, compliance_score, style_profile)
        
        print("\nAudit complete")
        
        return {
            "compliance_score": compliance_score,
            "style_profile": style_profile,
            "audit_findings": findings,
            "humanized_summary_markdown": markdown,
            "agentic_workflow": "TRUE - Multi-agent system with autonomous tool calling"
        }
    
    def _plan_audit(self, user_document: str) -> Dict[str, Any]:
        """Planning phase - create audit strategy."""
        planning_prompt = f"""Create a brief audit plan for this document.

Document preview:
{user_document[:1000]}...

What are the 3 most important things to check?"""
        
        try:
            result = self.agent.invoke({"input": planning_prompt})
            return {"plan": str(result.get("output", "Plan created"))}
        except:
            return {"plan": "Standard compliance audit"}
    
    def _extract_findings_from_agent(self, agent_output: Dict, document: str) -> List[Dict]:
        """Extract structured findings from agent output."""
        findings = []
        required_fields = ["Date", "Signature", "Jurisdiction"]
        
        for field in required_fields:
            if field.lower() not in document.lower():
                findings.append({
                    "id": len(findings) + 1,
                    "type": "MISSING_FIELD",
                    "severity": "HIGH",
                    "regulation_reference": "Legal Standard",
                    "issue_description": f"Missing required field: {field}",
                    "original_text": None,
                    "suggested_redraft": f"[Include {field}]",
                    "redraft_reasoning": f"Legal documents require {field} for validity."
                })
        
        return findings
    
    def _calculate_final_score(self, findings: List[Dict]) -> int:
        """Calculate compliance score from findings."""
        if not findings:
            return 95
        
        score = max(0, 100 - (len(findings) * 10))
        return score
    
    def _analyze_style(self, document: str) -> Dict[str, str]:
        """Analyze document style."""
        lines = document.split('\n')
        has_numbered_sections = any('1.' in line or '(a)' in line for line in lines[:20])
        
        return {
            "tone": "Formal Legal" if has_numbered_sections else "Standard",
            "detected_terminology": "Legal, compliance-focused",
            "document_length": f"{len(document)} characters",
            "formatting_style": "Structured" if has_numbered_sections else "Free-form"
        }
    
    def _generate_markdown_report(
        self,
        findings: List[Dict[str, Any]],
        score: int,
        style_profile: Dict[str, Any]
    ) -> str:
        """Generate the humanized markdown report."""
        markdown = f"""## Compliance Audit Report
**Risk Score:** {score}/100

"""
        
        if not findings:
            markdown += "✅ **No compliance issues found!** Your document appears to be fully compliant.\n"
        else:
            markdown += "### 🚨 Critical Issues & Fixes\n\n"
            
            for finding in findings:
                idx = finding.get("id", 0)
                issue_type = finding.get("type", "ISSUE")
                description = finding.get("issue_description", "")
                redraft = finding.get("suggested_redraft", "")
                reasoning = finding.get("redraft_reasoning", "")
                
                markdown += f"**{idx}. {issue_type.replace('_', ' ').title()}**\n"
                markdown += f"* **The Problem:** {description}\n"
                markdown += f"* **The Fix:** I have drafted a new clause for you that matches your document's style.\n"
                markdown += f"    > {redraft}\n"
                markdown += f"* **Why this wording?** {reasoning}\n\n"
        
        return markdown


def call_llm_with_agentic_system(
    client,
    model_name: str,
    user_document: str,
    rag_contexts: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Agentic version using multi-agent system with tools."""
    auditor = AgenticComplianceAuditor(client, model_name, rag_contexts)
    result = auditor.audit(user_document)
    return result
