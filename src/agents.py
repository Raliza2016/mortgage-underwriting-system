import json
from langchain_core.messages import SystemMessage, HumanMessage

from .state import UnderwritingState
from .tools import (
    calculate_dti_ratio,
    calculate_ltv_ratio,
    calculate_reserves,
    calculate_housing_expense_ratio,
    check_credit_score_policy,
    check_large_deposits,
    calculate_total_debt_obligations,
)
from .compliance import detect_bias_signals
from .policy_store import retrieve_relevant_policies


def credit_analyst_node(
    state: UnderwritingState, llm, policy_store=None
) -> UnderwritingState:
    """Analyzes borrower's credit profile and payment history."""
    policies = retrieve_relevant_policies(
        "credit score requirements bankruptcies foreclosures late payments",
        policy_store,
    )
    app_data = state["sanitized_data"]
    credit_score = app_data.get("credit_score", 0)
    credit_score_analysis = check_credit_score_policy.invoke(
        {"credit_score": credit_score}
    )

    system_prompt = f"""
You are a Senior Credit Analyst with 15+ years of experience in mortgage underwriting.

RELEVANT POLICIES:
{policies}

Your task is to analyze the borrower's credit profile and provide a detailed assessment.

ANALYSIS FRAMEWORK:
1. Credit Score Assessment - Use provided assessment (DO NOT recalculate)
2. Payment History - Review late payments and patterns
3. Derogatory Items - Evaluate bankruptcies, foreclosures, collections
4. Policy Compliance - Check against credit guidelines
5. Risk Rating - Assign credit risk (Low/Medium/High)
6. Recommendations - Provide conditions or concerns

Be thorough, objective, and policy-compliant. Support conclusions with data.

IMPORTANT: Use the EXACT credit score assessment provided below. Do not recalculate.
Do not use protected characteristics in the analysis.
"""

    user_prompt = f"""
Analyze the credit profile for case {state.get('case_id')}:

CALCULATED CREDIT SCORE ASSESSMENT (ACCURATE - DO NOT RECALCULATE):
{credit_score_analysis}

CREDIT HISTORY DATA:
- Bankruptcies: {app_data.get('credit_history', {}).get('bankruptcies', 0)}
- Foreclosures: {app_data.get('credit_history', {}).get('foreclosures', 0)}
- Late Payments (12mo): {app_data.get('credit_history', {}).get('late_payments_12mo', 0)}
- Collections: {app_data.get('credit_history', {}).get('collections', [])}

Provide your detailed credit analysis based on the ACCURATE assessment above.
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    analysis = response.content
    bias_flags = detect_bias_signals(analysis, app_data)

    return {
        **state,
        "credit_analysis": analysis,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", [])
        + [f"Credit Analyst: Completed credit analysis for {state.get('case_id')}"],
    }


def income_analyst_node(
    state: UnderwritingState, llm, policy_store=None
) -> UnderwritingState:
    """Analyzes borrower's income stability and capacity to repay."""
    policies = retrieve_relevant_policies(
        "employment income verification DTI ratio self-employed", policy_store
    )
    app_data = state["sanitized_data"]
    debts = app_data.get("debts", {})
    proposed_payment = app_data.get("loan", {}).get("estimated_payment", 0)
    monthly_income = app_data.get("employment", {}).get("monthly_income", 0)

    debts_for_sum = {k: v for k, v in debts.items() if k != "total_monthly_debt"}
    total_debt = debts.get("total_monthly_debt", sum(debts_for_sum.values()))

    dti_result = calculate_dti_ratio.invoke(
        {
            "monthly_debt": total_debt + proposed_payment,
            "monthly_income": monthly_income,
        }
    )
    housing_ratio_result = calculate_housing_expense_ratio.invoke(
        {"monthly_payment": proposed_payment, "monthly_income": monthly_income}
    )
    debt_breakdown = calculate_total_debt_obligations.invoke(
        {"debts": debts_for_sum, "proposed_payment": proposed_payment}
    )

    system_prompt = f"""
You are a Senior Mortgage Income Analyst with expertise in repayment capacity and income verification.

RELEVANT POLICIES:
{policies}

ANALYSIS FRAMEWORK:
1. Employment Stability - Review employment type, tenure, and consistency.
2. Income Verification - Assess whether income appears documented and usable.
3. Debt-to-Income Ratio - Use the provided DTI calculation exactly.
4. Housing Expense Ratio - Use the provided housing ratio exactly.
5. Total Monthly Obligations - Use the provided debt breakdown exactly.
6. Capacity to Repay - Determine whether income supports the proposed mortgage payment.
7. Risk Rating - Assign income risk as Low, Medium, or High.
8. Conditions or Concerns - Identify required documentation or compensating factors.

IMPORTANT: Use the calculated ratios provided. Do not recalculate.
Do not use protected characteristics in the analysis.
"""

    user_prompt = f"""
Analyze the income and repayment capacity for case {state.get('case_id')}.

EMPLOYMENT DATA:
{json.dumps(app_data.get('employment', {}), indent=2)}

LOAN DATA:
{json.dumps(app_data.get('loan', {}), indent=2)}

CURRENT DEBTS:
{json.dumps(debts_for_sum, indent=2)}

CALCULATED DTI RESULT:
{dti_result}

CALCULATED HOUSING EXPENSE RATIO:
{housing_ratio_result}

CALCULATED TOTAL DEBT OBLIGATIONS:
{debt_breakdown}

Provide a detailed income analysis based only on the verified data and calculated tool outputs above.
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    analysis = response.content
    bias_flags = detect_bias_signals(analysis, app_data)

    return {
        **state,
        "income_analysis": analysis,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", [])
        + ["Income Analyst: Completed income analysis with DTI calculation"],
    }


def asset_analyst_node(
    state: UnderwritingState, llm, policy_store=None
) -> UnderwritingState:
    """Analyzes borrower's assets and reserves."""
    policies = retrieve_relevant_policies(
        "down payment reserves assets large deposits gift funds", policy_store
    )
    app_data = state["sanitized_data"]
    assets = app_data.get("assets", {})
    loan = app_data.get("loan", {})
    monthly_income = app_data.get("employment", {}).get("monthly_income", 0)

    liquid_assets = assets.get("checking", 0) + assets.get("savings", 0)
    monthly_payment = loan.get("estimated_payment", 0)

    reserves_result = calculate_reserves.invoke(
        {
            "liquid_assets": liquid_assets,
            "monthly_payment": monthly_payment,
            "required_months": 2,
        }
    )
    deposits_result = check_large_deposits.invoke(
        {
            "deposits": assets.get("recent_deposits", []),
            "monthly_income": monthly_income,
        }
    )

    system_prompt = f"""
You are a Senior Mortgage Asset Analyst with expertise in asset verification, reserves, and down payment sourcing.

RELEVANT POLICIES:
{policies}

ANALYSIS FRAMEWORK:
1. Liquid Assets - Review available checking and savings funds.
2. Down Payment Adequacy - Determine whether available funds support the transaction.
3. Reserve Coverage - Use the provided reserve calculation exactly.
4. Large Deposits - Use the provided large deposit review exactly.
5. Gift Funds or Unusual Deposits - Identify any documentation concerns.
6. Asset Documentation - Assess whether bank statements or sourcing documents are needed.
7. Risk Rating - Assign asset risk as Low, Medium, or High.
8. Conditions or Concerns - Identify any required follow-up documents.

IMPORTANT: Use the calculated reserve and large deposit results provided. Do not recalculate.
Do not use protected characteristics in the analysis.
"""

    user_prompt = f"""
Analyze the asset profile for case {state.get('case_id')}.

ASSET DATA:
{json.dumps(assets, indent=2)}

LOAN DATA:
{json.dumps(loan, indent=2)}

MONTHLY INCOME: {monthly_income}

CALCULATED RESERVES RESULT:
{reserves_result}

LARGE DEPOSIT REVIEW:
{deposits_result}

Provide a detailed asset analysis based only on the verified data and calculated tool outputs above.
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    analysis = response.content
    bias_flags = detect_bias_signals(analysis, app_data)

    return {
        **state,
        "asset_analysis": analysis,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", [])
        + ["Asset Analyst: Completed asset analysis and deposit review"],
    }


def collateral_analyst_node(
    state: UnderwritingState, llm, policy_store=None
) -> UnderwritingState:
    """Analyzes property value and condition."""
    policies = retrieve_relevant_policies(
        "appraisal property condition LTV collateral", policy_store
    )
    app_data = state["sanitized_data"]
    property_data = app_data.get("property", {})
    loan = app_data.get("loan", {})

    loan_amount = loan.get("amount", 0)
    appraised_value = property_data.get("appraised_value", 0)

    ltv_result = calculate_ltv_ratio.invoke(
        {"loan_amount": loan_amount, "property_value": appraised_value}
    )

    system_prompt = f"""
You are a Senior Collateral Analyst with expertise in property valuation.

RELEVANT POLICIES:
{policies}

ANALYSIS FRAMEWORK:
1. Appraisal Review - Validate property value.
2. LTV Calculation - Use provided calculation exactly.
3. Property Condition - Evaluate habitability and repair concerns.
4. Marketability - Consider property type and market factors.
5. Collateral Risk Rating - Assign Low, Medium, or High risk.
6. Recommendations - Note required conditions or concerns.

IMPORTANT: Use the EXACT LTV calculation provided. Do not recalculate.
Do not use protected characteristics in the analysis.
"""

    user_prompt = f"""
Analyze property collateral for case {state.get('case_id')}.

PROPERTY DATA:
{json.dumps(property_data, indent=2)}

LOAN DATA:
{json.dumps(loan, indent=2)}

CALCULATED LTV RESULT:
{ltv_result}

Provide your collateral analysis based only on the verified property data and calculated LTV result above.
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    analysis = response.content
    bias_flags = detect_bias_signals(analysis, app_data)

    return {
        **state,
        "collateral_analysis": analysis,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", [])
        + ["Collateral Analyst: Completed property analysis (LTV from tool)"],
    }


def critic_agent_node(
    state: UnderwritingState, llm, policy_store=None
) -> UnderwritingState:
    """Reviews all specialist analyses for consistency and completeness."""
    system_prompt = """
You are a Quality Assurance Critic reviewing underwriting analyses.

Your role is to:
1. Verify all analyses are complete and thorough
2. Identify any contradictions or inconsistencies
3. Ensure policy compliance
4. Flag any missing information
5. Provide a synthesis of key findings

Be critical but fair. Focus on ensuring decision quality.
"""

    user_prompt = f"""
Review all analyses for case {state.get('case_id')}:

CREDIT ANALYSIS:
{state.get('credit_analysis', 'Not completed')}

INCOME ANALYSIS:
{state.get('income_analysis', 'Not completed')}

ASSET ANALYSIS:
{state.get('asset_analysis', 'Not completed')}

COLLATERAL ANALYSIS:
{state.get('collateral_analysis', 'Not completed')}

BIAS FLAGS:
{state.get('bias_flags', [])}

Provide your critical review and synthesis.
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    return {
        **state,
        "critic_review": response.content,
        "reasoning_chain": state.get("reasoning_chain", [])
        + ["Critic: Completed review of all specialist analyses"],
    }


def decision_agent_node(
    state: UnderwritingState, llm, policy_store=None
) -> UnderwritingState:
    """Synthesizes all findings into a final decision and credit memo."""
    import re

    system_prompt = """
You are a Senior Mortgage Underwriting Decision Agent.

Your role is to synthesize the specialist analyses and produce an audit-ready underwriting decision.

Decision options:
- APPROVED
- DENIED
- CONDITIONAL_APPROVAL

Decision guidance:
- APPROVED: all major underwriting areas are acceptable with no material unresolved risk.
- CONDITIONAL_APPROVAL: borrower has material but potentially curable issues (elevated DTI,
  moderate credit weakness, self-employed income documentation, large deposit sourcing,
  low down payment, appraisal variance, or compensating factors requiring human review).
- DENIED: use only when the borrower clearly fails non-curable eligibility requirements,
  has severe unresolved policy violations, insufficient repayment capacity with no
  compensating factors, unacceptable collateral, or disqualifying credit events.

For borderline applicants with compensating factors, prefer CONDITIONAL_APPROVAL over DENIED.

Risk score guidance:
- 0-30: Low risk
- 31-64: Moderate risk
- 65-100: High risk

Your output must include exactly:
1. RISK_SCORE: <number 0-100>
2. DECISION: <APPROVED/DENIED/CONDITIONAL_APPROVAL>
3. CREDIT_MEMO: <comprehensive decision documentation>

Be objective, policy-aware, and compliance-sensitive.
Do not use protected characteristics in the decision.
"""

    user_prompt = f"""
Make final underwriting decision for case {state.get('case_id')}:

CREDIT ANALYSIS SUMMARY:
{state.get('credit_analysis', 'N/A')[:500]}...

INCOME ANALYSIS SUMMARY:
{state.get('income_analysis', 'N/A')[:500]}...

ASSET ANALYSIS SUMMARY:
{state.get('asset_analysis', 'N/A')[:500]}...

COLLATERAL ANALYSIS SUMMARY:
{state.get('collateral_analysis', 'N/A')[:500]}...

CRITIC REVIEW:
{state.get('critic_review', 'N/A')[:500]}...

COMPLIANCE ALERTS:
- Bias Flags: {len(state.get('bias_flags', []))}
- Policy Violations: {len(state.get('policy_violations', []))}

Provide:
1. RISK_SCORE: (number 0-100)
2. DECISION: (APPROVED/DENIED/CONDITIONAL_APPROVAL)
3. CREDIT_MEMO: (comprehensive decision documentation)
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )

    content = response.content
    risk_score = 50
    match = re.search(r"RISK_SCORE:\s*(\d+)", content)
    if match:
        risk_score = int(match.group(1))

    decision = "CONDITIONAL_APPROVAL"
    if re.search(r"DECISION:\s*DENIED", content, re.IGNORECASE):
        decision = "DENIED"
    elif re.search(r"DECISION:\s*APPROVED", content, re.IGNORECASE):
        decision = "APPROVED"
    elif re.search(r"DECISION:\s*CONDITIONAL[_\s]APPROVAL", content, re.IGNORECASE):
        decision = "CONDITIONAL_APPROVAL"

    human_review_required = (
        risk_score >= 65
        or len(state.get("bias_flags", [])) > 0
        or decision in ["DENIED", "CONDITIONAL_APPROVAL"]
    )

    return {
        **state,
        "decision_memo": content,
        "risk_score": risk_score,
        "final_decision": decision,
        "human_review_required": human_review_required,
        "analysis_complete": True,
        "reasoning_chain": state.get("reasoning_chain", [])
        + [f"Decision Agent: Final decision {decision} with risk score {risk_score}"],
    }
