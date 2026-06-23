# Mortgage Underwriting System

A multi-agent AI pipeline that automates residential mortgage loan analysis using LangGraph and OpenAI.

## Overview

The system routes each loan application through a team of specialized AI agents in sequence:

```
Application → Credit Analyst → Income Analyst → Asset Analyst
           → Collateral Analyst → Critic → Decision Agent → Output
```

Each agent uses retrieval-augmented policies (RAG) and dedicated calculation tools to produce structured, auditable analysis. The final output is a structured credit memo with an APPROVED / DENIED / CONDITIONAL_APPROVAL decision and a 0–100 risk score.

## Features

- **6 specialized agents** — Credit, Income, Asset, Collateral, Critic, Decision
- **Calculation tools** — DTI ratio, LTV ratio, housing expense ratio, reserve coverage, large-deposit detection
- **PII sanitization** — SSN, name, address, and phone redaction before LLM processing
- **Fair Lending compliance** — Automated bias signal detection across all analyses
- **Policy RAG** — Retrieval from built-in policy text (or your own PDF)
- **LangGraph checkpointing** — Full audit trail per case with in-memory state persistence

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/your-username/mortgage-underwriting-system.git
cd mortgage-underwriting-system
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run

```bash
# Run all three built-in test cases
python main.py

# Run a specific test case
python main.py --case TC-001

# Run from a custom JSON file
python main.py --input path/to/my_case.json
```

## Project Structure

```
mortgage_underwriting_system/
├── main.py                  # CLI entry point
├── requirements.txt
├── .env.example
├── data/
│   └── test_cases.json      # Three sample loan applications
└── src/
    ├── state.py             # UnderwritingState TypedDict
    ├── tools.py             # Calculation tools (DTI, LTV, reserves, etc.)
    ├── compliance.py        # PII sanitization & bias detection
    ├── policy_store.py      # Policy RAG (built-in text + optional PDF/Chroma)
    ├── agents.py            # All six agent node functions
    └── workflow.py          # LangGraph graph builder and run helper
```

## Test Cases

Three sample applications are included in `data/test_cases.json`:

| Case | Profile | Expected |
|------|---------|----------|
| TC-001 | Strong applicant — high credit, stable W2 income | APPROVED |
| TC-002 | Borderline — self-employed, moderate credit | CONDITIONAL_APPROVAL |
| TC-003 | Weak applicant — low credit, high DTI, derogatory items | DENIED |

## Using Your Own Policy PDF

If you have an underwriting policy PDF, place it in the project root and set the path in code:

```python
from src.policy_store import create_policy_store
from src.workflow import build_workflow

policy_store = create_policy_store("my_policies.pdf")
graph = build_workflow(policy_store=policy_store)
```

You will also need to install the optional dependencies:

```bash
pip install pypdf chromadb
```

## Custom Case Format

Pass any application via `--input` with a JSON file matching this structure:

```json
{
  "test_cases": [
    {
      "case_id": "MY-001",
      "name": "...",
      "ssn": "...",
      "credit_score": 720,
      "employment": { "type": "W2", "monthly_income": 7000, ... },
      "loan": { "amount": 300000, "estimated_payment": 1750 },
      "debts": { "auto_loan": 400, "credit_cards": 200 },
      "assets": { "checking": 10000, "savings": 30000, "recent_deposits": [] },
      "property": { "appraised_value": 380000, "condition": "Good" },
      "credit_history": { "bankruptcies": 0, "foreclosures": 0, "late_payments_12mo": 0, "collections": [] }
    }
  ]
}
```

## License

MIT
