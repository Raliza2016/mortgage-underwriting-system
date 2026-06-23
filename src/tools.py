from langchain_core.tools import tool


@tool
def calculate_dti_ratio(monthly_debt: float, monthly_income: float) -> str:
    """Calculate Debt-to-Income ratio for mortgage qualification."""
    if monthly_income <= 0:
        return "Error: Monthly income must be greater than zero."
    dti = (monthly_debt / monthly_income) * 100
    if dti <= 36:
        assessment = "Excellent - Well within guidelines"
    elif dti <= 43:
        assessment = "Acceptable - Meets conventional loan limits"
    elif dti <= 50:
        assessment = "Elevated - May require compensating factors"
    else:
        assessment = "Exceeds limits - Likely ineligible for conventional financing"
    return (
        f"DTI Ratio: {dti:.1f}%\n"
        f"Monthly Debt: ${monthly_debt:,.2f}\n"
        f"Monthly Income: ${monthly_income:,.2f}\n"
        f"Assessment: {assessment}"
    )


@tool
def calculate_ltv_ratio(loan_amount: float, property_value: float) -> str:
    """Calculate Loan-to-Value ratio for collateral assessment."""
    if property_value <= 0:
        return "Error: Property value must be greater than zero."
    ltv = (loan_amount / property_value) * 100
    if ltv <= 80:
        assessment = "Excellent - No PMI required"
    elif ltv <= 90:
        assessment = "Acceptable - PMI may be required"
    elif ltv <= 97:
        assessment = "High LTV - PMI required, limited programs available"
    else:
        assessment = "Exceeds conventional limits"
    return (
        f"LTV Ratio: {ltv:.1f}%\n"
        f"Loan Amount: ${loan_amount:,.2f}\n"
        f"Property Value: ${property_value:,.2f}\n"
        f"Assessment: {assessment}"
    )


@tool
def calculate_reserves(
    liquid_assets: float, monthly_payment: float, required_months: int = 2
) -> str:
    """Calculate reserve coverage in months."""
    if monthly_payment <= 0:
        return "Error: Monthly payment must be greater than zero."
    months_of_reserves = liquid_assets / monthly_payment
    required = required_months
    if months_of_reserves >= required * 2:
        assessment = f"Strong reserves - {months_of_reserves:.1f} months available"
    elif months_of_reserves >= required:
        assessment = f"Adequate reserves - {months_of_reserves:.1f} months available"
    else:
        assessment = f"Insufficient reserves - {months_of_reserves:.1f} months (minimum {required} required)"
    return (
        f"Reserve Coverage: {months_of_reserves:.1f} months\n"
        f"Liquid Assets: ${liquid_assets:,.2f}\n"
        f"Monthly Payment: ${monthly_payment:,.2f}\n"
        f"Required Reserves: {required} months\n"
        f"Assessment: {assessment}"
    )


@tool
def calculate_housing_expense_ratio(
    monthly_payment: float, monthly_income: float
) -> str:
    """Calculate front-end (housing expense) ratio."""
    if monthly_income <= 0:
        return "Error: Monthly income must be greater than zero."
    ratio = (monthly_payment / monthly_income) * 100
    if ratio <= 28:
        assessment = "Excellent - Well within guidelines"
    elif ratio <= 31:
        assessment = "Acceptable - Meets standard guidelines"
    elif ratio <= 36:
        assessment = "Elevated - Monitor closely"
    else:
        assessment = "Exceeds guidelines - Compensating factors required"
    return (
        f"Housing Expense Ratio: {ratio:.1f}%\n"
        f"Monthly Payment: ${monthly_payment:,.2f}\n"
        f"Monthly Income: ${monthly_income:,.2f}\n"
        f"Assessment: {assessment}"
    )


@tool
def check_credit_score_policy(credit_score: int) -> str:
    """Check if credit score meets policy requirements."""
    if credit_score >= 740:
        tier = "Excellent"
        rate_adjustment = "Best rates available"
    elif credit_score >= 700:
        tier = "Very Good"
        rate_adjustment = "Favorable rates"
    elif credit_score >= 660:
        tier = "Good"
        rate_adjustment = "Standard rates"
    elif credit_score >= 620:
        tier = "Fair"
        rate_adjustment = "Higher rates, may require compensating factors"
    else:
        tier = "Below Minimum"
        rate_adjustment = "Does not meet conventional loan requirements"
    return f"Credit Score: {credit_score} - Tier: {tier} - {rate_adjustment}"


@tool
def check_large_deposits(deposits: list, monthly_income: float) -> str:
    """Identify large deposits requiring sourcing documentation."""
    threshold = monthly_income * 0.25
    large_deposits = []
    for deposit in deposits:
        amount = deposit.get("amount", 0)
        if amount >= threshold:
            large_deposits.append(
                {
                    "amount": amount,
                    "date": deposit.get("date", "Unknown"),
                    "sourcing_required": True,
                }
            )
    if not large_deposits:
        return (
            f"No large deposits identified "
            f"(threshold: ${threshold:,.2f}). All deposits are acceptable."
        )
    result = (
        f"Found {len(large_deposits)} large deposit(s) requiring documentation "
        f"(threshold: ${threshold:,.2f}):\n"
    )
    for i, dep in enumerate(large_deposits, 1):
        result += (
            f"  {i}. ${dep['amount']:,.2f} on {dep['date']} - "
            f"Sourcing documentation required\n"
        )
    return result


@tool
def calculate_total_debt_obligations(debts: dict, proposed_payment: float) -> str:
    """Calculate total monthly debt obligations including proposed loan."""
    current_debt = sum(debts.values())
    total_obligation = current_debt + proposed_payment
    breakdown = "\n".join([f"  - {k}: ${v:,.2f}" for k, v in debts.items()])
    return (
        f"Total Monthly Obligations: ${total_obligation:,.2f}\n"
        f"Current Debt: ${current_debt:,.2f}\n"
        f"{breakdown}\n"
        f"Proposed Payment: ${proposed_payment:,.2f}"
    )
