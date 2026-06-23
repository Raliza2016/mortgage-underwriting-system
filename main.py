#!/usr/bin/env python3
"""
Mortgage Underwriting System
Multi-agent AI pipeline for automated loan application analysis.

Usage:
    python main.py                        # Run all test cases
    python main.py --case TC-001          # Run a specific case by ID
    python main.py --input path/to.json   # Run from a custom JSON file
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.policy_store import create_policy_store
from src.workflow import build_workflow, run_case


def print_header(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(values: dict):
    decision = values.get("final_decision", "PENDING")
    risk_score = values.get("risk_score", "N/A")
    human_review = values.get("human_review_required", False)
    bias_flags = values.get("bias_flags", [])

    decision_symbol = {
        "APPROVED": "✅",
        "DENIED": "❌",
        "CONDITIONAL_APPROVAL": "⚠️",
    }.get(decision, "❓")

    print(f"\n{decision_symbol} Final Decision : {decision}")
    print(f"   Risk Score    : {risk_score}/100")
    print(f"   Human Review  : {'Required' if human_review else 'Not required'}")
    print(f"   Bias Flags    : {len(bias_flags)}")

    print("\nReasoning Chain:")
    for i, step in enumerate(values.get("reasoning_chain", []), 1):
        print(f"  {i}. {step}")

    memo = values.get("decision_memo", "")
    if memo:
        print("\nDecision Memo (excerpt):")
        print("-" * 60)
        print(memo[:800])
        if len(memo) > 800:
            print("[... truncated — full memo saved in state ...]")


def main():
    parser = argparse.ArgumentParser(description="Mortgage Underwriting System")
    parser.add_argument("--case", help="Run a specific case ID from test_cases.json")
    parser.add_argument("--input", help="Path to custom JSON file with case data")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("Copy .env.example to .env and fill in your API key.")
        sys.exit(1)

    print_header("Mortgage Underwriting System — Initializing")
    print("Building workflow and loading policies...")

    policy_store = create_policy_store()
    graph = build_workflow(policy_store=policy_store)
    print("Workflow ready.\n")

    if args.input:
        with open(args.input, "r") as f:
            data = json.load(f)
        cases = data.get("test_cases", [data])
    else:
        with open("data/test_cases.json", "r") as f:
            data = json.load(f)
        cases = data["test_cases"]
        if args.case:
            cases = [c for c in cases if c["case_id"] == args.case]
            if not cases:
                print(f"Case '{args.case}' not found in test_cases.json")
                sys.exit(1)

    for case_data in cases:
        print_header(
            f"Case: {case_data['case_id']}  (Expected: {case_data.get('expected_decision', 'N/A')})"
        )
        print("Running multi-agent pipeline...\n")

        values = run_case(graph, case_data, thread_id=f"{case_data['case_id']}_run")
        print_result(values)
        print()


if __name__ == "__main__":
    main()
