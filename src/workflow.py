import os
from datetime import datetime
from functools import partial

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import UnderwritingState
from .compliance import sanitize_pii
from .policy_store import create_policy_store
from .agents import (
    credit_analyst_node,
    income_analyst_node,
    asset_analyst_node,
    collateral_analyst_node,
    critic_agent_node,
    decision_agent_node,
)


def build_workflow(llm=None, policy_store=None):
    """
    Build and compile the complete multi-agent underwriting workflow.

    Args:
        llm: LangChain LLM instance. Defaults to ChatOpenAI with env vars.
        policy_store: Optional vector store for policy retrieval.

    Returns:
        Compiled LangGraph workflow.
    """
    if llm is None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        )

    def initialize_application(state: UnderwritingState) -> UnderwritingState:
        sanitized = sanitize_pii(state["applicant_data"])
        return {
            **state,
            "sanitized_data": sanitized,
            "analysis_complete": False,
            "human_review_required": False,
            "human_review_completed": False,
            "bias_flags": [],
            "policy_violations": [],
            "reasoning_chain": [f"Application {state.get('case_id')} initialized"],
            "timestamp": datetime.now().isoformat()
        }

    def supervisor_node(state: UnderwritingState) -> UnderwritingState:
        analyses_done = {
            "credit": state.get("credit_analysis") is not None,
            "income": state.get("income_analysis") is not None,
            "asset": state.get("asset_analysis") is not None,
            "collateral": state.get("collateral_analysis") is not None,
        }
        if not analyses_done["credit"]:
            next_agent = "credit"
        elif not analyses_done["income"]:
            next_agent = "income"
        elif not analyses_done["asset"]:
            next_agent = "asset"
        elif not analyses_done["collateral"]:
            next_agent = "collateral"
        else:
            next_agent = "critic"

        return {
            **state,
            "next_agent": next_agent,
            "analysis_complete": all(analyses_done.values()),
        }

    def should_continue_to_agents(state: UnderwritingState) -> str:
        if state.get("analysis_complete", False):
            return "critic"
        return state.get("next_agent", "credit")

    def _credit(state):
        return credit_analyst_node(state, llm, policy_store)

    def _income(state):
        return income_analyst_node(state, llm, policy_store)

    def _asset(state):
        return asset_analyst_node(state, llm, policy_store)

    def _collateral(state):
        return collateral_analyst_node(state, llm, policy_store)

    def _critic(state):
        return critic_agent_node(state, llm, policy_store)

    def _decision(state):
        return decision_agent_node(state, llm, policy_store)

    workflow = StateGraph(UnderwritingState)

    workflow.add_node("initialize", initialize_application)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("credit", _credit)
    workflow.add_node("income", _income)
    workflow.add_node("asset", _asset)
    workflow.add_node("collateral", _collateral)
    workflow.add_node("critic", _critic)
    workflow.add_node("decision", _decision)

    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        should_continue_to_agents,
        {
            "credit": "credit",
            "income": "income",
            "asset": "asset",
            "collateral": "collateral",
            "critic": "critic",
        }
    )

    workflow.add_edge("credit", "supervisor")
    workflow.add_edge("income", "supervisor")
    workflow.add_edge("asset", "supervisor")
    workflow.add_edge("collateral", "supervisor")
    workflow.add_edge("critic", "decision")
    workflow.add_edge("decision", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def run_case(graph, case_data: dict, thread_id: str = None) -> dict:
    """
    Run a single underwriting case through the full workflow.

    Args:
        graph: Compiled workflow graph.
        case_data: Application data dictionary.
        thread_id: Unique thread ID for checkpointing (defaults to case_id).

    Returns:
        Final state values dictionary.
    """
    if thread_id is None:
        thread_id = case_data.get("case_id", "default")

    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "case_id": case_data["case_id"],
        "applicant_data": case_data,
    }

    for event in graph.stream(inputs, config):
        for node_name, _ in event.items():
            if node_name not in ["__start__", "supervisor"]:
                print(f"  ✓ {node_name.replace('_', ' ').title()} completed")

    final_state = graph.get_state(config)
    return final_state.values
