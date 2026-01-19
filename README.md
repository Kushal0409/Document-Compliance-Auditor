# AI Legal Compliance Auditor

An intelligent AI-powered system that audits legal documents for regulatory compliance and generates style-adapted corrections. Built for MSMEs (Micro, Small, and Medium Enterprises) to ensure their business documents meet legal requirements.

## 🚀 Features

- **Agentic AI System**: Multi-agent architecture with autonomous tool calling (planning, execution, refinement)
- **Smart Document Analysis**: Analyzes document structure, tone, terminology, and formatting style
- **Compliance Scoring**: Calculates compliance risk scores (0-100) based on identified issues
- **Style-Aware Redrafting**: Generates corrections that match your document's original tone and style
- **Multi-Format Support**: Works with PDF, DOCX, and TXT files
- **Semantic Search**: Uses vector embeddings to find relevant regulations intelligently
- **Dual Interface**: Both CLI and interactive Gradio web UI
- **Custom Regulations**: Inject your own regulations or policy documents for analysis

## 📋 Requirements

- Python 3.8+
- Google Gemini API Key
- Dependencies: See `requirements.txt`

## 🔧 Installation

```bash
# Clone the repository
git clone <repo-url>
cd Auditor_Compliance_Final

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
echo "GEMINI_MODEL=models/gemini-2.5-pro" >> .env
echo "USE_AGENTIC=true" >> .env
```

## 💻 Usage

### CLI Mode
```bash
python main.py
```

### Web UI
```bash
python gradio_app.py
```

Then open your browser to the displayed URL.

## How It Works

1. **Planning Phase**: Agent creates an audit strategy
2. **Execution Phase**: Agent autonomously uses tools to:
   - Search regulations
   - Check missing fields
   - Analyze document structure
   - Compare clauses against regulations
3. **Refinement Phase**: Validates findings and generates compliant redrafts

##  Output

The audit returns:
- **Compliance Score**: 0-100 risk assessment
- **Findings**: Identified gaps and non-compliant clauses
- **Style Profile**: Document tone, terminology, and structure analysis
- **Suggested Fixes**: Style-matched corrections for each issue
- **Markdown Report**: Human-readable summary

##  Tools Available

- `search_regulations` - Find relevant regulations
- `analyze_document_structure` - Extract style profile
- `check_missing_fields` - Identify missing legal fields
- `compare_with_regulation` - Verify clause compliance
- `generate_style_adapted_clause` - Create fixes matching original style
- `calculate_compliance_score` - Compute compliance risk

##  Key Technologies

- **LangChain** - Agent framework
- **Google Gemini** - Large language model
- **Chroma DB** - Vector database for semantic search
- **Sentence Transformers** - Embeddings
- **Gradio** - Web interface

##  Use Cases

- Contract review and compliance
- Policy document auditing
- Regulatory requirement checking
- Document style standardization
- Legal document templates

**Perfect for**: MSMEs, legal teams, compliance officers, and document-heavy businesses
