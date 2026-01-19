# Agentic AI System - Implementation Guide

## Overview

This project has been upgraded to include **true Agentic AI capabilities** with multi-step reasoning, tool calling, and autonomous decision-making.

## What's New: Agentic Features

### 🤖 Multi-Agent Architecture

1. **Planning Agent**: Breaks down the audit into steps
2. **Execution Agent**: Performs the audit using tools
3. **Refinement Agent**: Reviews and improves results

### 🛠️ Available Tools

The agent can autonomously call these tools:

1. **`search_regulations`**: Search through regulation documents for specific requirements
2. **`analyze_document_structure`**: Analyze document tone, terminology, and formatting
3. **`check_missing_fields`**: Check for required fields (Date, CIN, Signature, etc.)
4. **`compare_with_regulation`**: Compare document clauses against regulations
5. **`generate_style_adapted_clause`**: Generate clauses that match document style
6. **`calculate_compliance_score`**: Calculate compliance risk score

### 🔄 Agentic Workflow

```
User Document
    ↓
[PLANNING AGENT] → Creates audit plan
    ↓
[EXECUTION AGENT] → Uses tools to:
    - Search regulations
    - Check compliance
    - Analyze style
    - Find issues
    ↓
[REFINEMENT AGENT] → Reviews and improves:
    - Validates findings
    - Generates style-adapted fixes
    - Calculates final score
    ↓
Final Audit Report
```

## Installation

1. **Install new dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- `langchain` - Agent framework
- `langchain-google-genai` - Gemini integration
- `langchain-core` - Core LangChain components
- `langchain-community` - Community tools

## Configuration

### Enable/Disable Agentic Mode

In your `.env` file:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=models/gemini-2.5-pro
USE_AGENTIC=true  # Set to false to use standard mode
```

### Default Behavior

- **Agentic mode is ENABLED by default** (`USE_AGENTIC=true`)
- Falls back to standard mode if agentic system fails
- You can disable it by setting `USE_AGENTIC=false`

## Usage

### Command Line

```bash
python main.py
```

The system will automatically use agentic mode if enabled.

### Gradio Web UI

```bash
python gradio_app.py
```

Same agentic capabilities available in the web interface.

## How It Works

### Standard Mode (Non-Agentic)
- Single LLM call
- All instructions in one prompt
- No tool calling
- No multi-step reasoning

### Agentic Mode (New!)
- **Planning Phase**: Agent creates a step-by-step audit plan
- **Execution Phase**: Agent autonomously:
  - Decides which tools to use
  - Calls tools as needed
  - Iterates through findings
- **Refinement Phase**: Agent reviews and improves results

## Example Agent Behavior

When you run an audit, the agent will:

1. **Plan**: "I need to check for missing fields, search regulations for data protection requirements, and compare clauses."

2. **Execute**:
   - Calls `check_missing_fields` → Finds missing "Date" field
   - Calls `search_regulations` with query "data protection" → Finds relevant regulations
   - Calls `compare_with_regulation` → Identifies non-compliant clause
   - Calls `analyze_document_structure` → Detects "Corporate Professional" tone

3. **Refine**:
   - Calls `generate_style_adapted_clause` → Creates style-matched fix
   - Calls `calculate_compliance_score` → Computes final score
   - Validates all findings

## Key Differences

| Feature | Standard Mode | Agentic Mode |
|---------|--------------|--------------|
| **Tool Calling** | ❌ No | ✅ Yes |
| **Multi-Step** | ❌ Single prompt | ✅ Planning → Execution → Refinement |
| **Autonomous Decisions** | ❌ No | ✅ Agent decides which tools to use |
| **Iteration** | ❌ One-shot | ✅ Can refine and improve |
| **Error Recovery** | ⚠️ Basic | ✅ Advanced with agent reasoning |

## Troubleshooting

### Agentic System Not Working?

1. **Check dependencies**:
   ```bash
   pip install langchain langchain-google-genai langchain-core langchain-community
   ```

2. **Check API key**: Make sure `GEMINI_API_KEY` is set in `.env`

3. **Fallback**: System automatically falls back to standard mode if agentic fails

4. **Disable agentic**: Set `USE_AGENTIC=false` in `.env` to use standard mode

### Model Compatibility

- Works best with: `models/gemini-2.5-pro` or `models/gemini-3-pro-preview`
- Function calling requires models that support tool use
- Standard mode works with any Gemini model

## Architecture

```
agentic_tools.py      → Tool definitions (search, analyze, check, etc.)
agentic_auditor.py    → Multi-agent system (planning, execution, refinement)
auditor.py            → Main interface (supports both modes)
main.py               → CLI entry point
gradio_app.py         → Web UI entry point
```

## Future Enhancements

Potential improvements:
- [ ] Add more tools (web search, legal database APIs)
- [ ] Multi-agent collaboration (specialist agents)
- [ ] Long-term memory for audit history
- [ ] Self-correction loops
- [ ] External API integrations

## Questions?

The agentic system is designed to be transparent. You'll see:
- `🤖 [AGENT]` messages showing agent activity
- Tool calls in verbose mode
- Step-by-step progress indicators
