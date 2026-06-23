import re
from typing import Dict, Any, List


def sanitize_pii(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove or redact personally identifiable information."""
    sanitized = data.copy()

    if "ssn" in sanitized:
        ssn = sanitized["ssn"]
        sanitized["ssn"] = f"***-**-{ssn[-4:]}" if len(ssn) >= 4 else "***-**-XXXX"

    if "name" in sanitized:
        sanitized["name"] = "[APPLICANT_NAME]"

    if "address" in sanitized:
        sanitized["address"] = "[ADDRESS]"

    if "phone" in sanitized:
        phone = sanitized["phone"]
        sanitized["phone"] = f"***-***-{phone[-4:]}" if len(phone) >= 4 else "***-***-XXXX"

    return sanitized


def detect_bias_signals(analysis: str, applicant_data: Dict[str, Any]) -> List[str]:
    """Check for potential Fair Lending bias signals."""
    flags = []

    protected_terms = [
        "race", "color", "religion", "national origin",
        "sex", "marital status", "age", "gender",
        "disability", "familial status"
    ]

    analysis_lower = analysis.lower()

    for term in protected_terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, analysis_lower):
            flags.append(f"Analysis mentions protected characteristic: {term}")

    if "zip" in applicant_data or "zipcode" in applicant_data:
        if re.search(r"\b(neighborhood|area)\b", analysis_lower):
            flags.append("Potential geographic bias - review for Fair Lending compliance")

    return flags
