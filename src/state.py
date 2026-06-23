from typing import TypedDict, List, Dict, Any, Optional


class UnderwritingState(TypedDict):
    case_id: str
    applicant_data: Dict[str, Any]
    sanitized_data: Dict[str, Any]
    credit_analysis: Optional[str]
    income_analysis: Optional[str]
    asset_analysis: Optional[str]
    collateral_analysis: Optional[str]
    critic_review: Optional[str]
    decision_memo: Optional[str]
    final_decision: Optional[str]
    risk_score: Optional[int]
    bias_flags: List[str]
    policy_violations: List[str]
    human_review_required: bool
    human_review_completed: bool
    analysis_complete: bool
    next_agent: Optional[str]
    reasoning_chain: List[str]
    timestamp: Optional[str]
