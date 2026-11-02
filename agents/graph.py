"""
Phase 5: LangGraph orchestration.

This is the node-by-node implementation of the architecture diagram
from the project plan:

    parse_job -> retrieve -> rerank -> apply_rules
                                            |
                          (real conditional branch on whether
                           any candidate actually qualified)
                                            |
                 qualified? --yes--> collect_evidence -> deep_evaluate --+
                     |no                                                 |
                     +---------------------> explain_rejected <----------+
                                                    |
                                              generate_report -> END

Why LangGraph and not just a Python function calling things in order:
apply_rules is a genuine branch point — if a shortlist has zero
qualified candidates, the whole evidence-collection + LLM-explanation
path is skippable, and that's expressed here as a real conditional
edge rather than an if-statement buried in one big function. As the
workflow grows (adding a job-description-parsing LLM step, a
human-in-the-loop review step, retries on low-confidence evidence,
etc.) each becomes another node/edge instead of more nested logic.
"""

from langgraph.graph import StateGraph, END

from agents.state import ScreeningState
from rag.rule_engine import evaluate_candidate


def build_graph(retriever, reranker, explainer, candidate_profiles: dict, candidate_texts: dict, top_n: int = 15):
    """
    candidate_profiles: candidate_id -> {"candidate_id", "skills", "education", "experience_years"}
    candidate_texts:    candidate_id -> {"summary_text": str, "sections": {section: text}}
    """

    def parse_job(state: ScreeningState) -> dict:
        job_rule = state["job_rule"]
        query = ", ".join(job_rule["required_skills"] + job_rule.get("preferred_skills", []))
        return {"query": query}

    def retrieve(state: ScreeningState) -> dict:
        hits = retriever.search(state["query"], top_k=150)
        seen, ordered_candidate_ids = set(), []
        for h in hits:
            cid = h["id"].split("__")[0]
            if cid not in seen:
                seen.add(cid)
                ordered_candidate_ids.append(cid)
        return {"retrieved_candidate_ids": ordered_candidate_ids}

    def rerank(state: ScreeningState) -> dict:
        candidates = [
            {"id": cid, "text": candidate_texts.get(cid, {}).get("summary_text", "")}
            for cid in state["retrieved_candidate_ids"]
            if cid in candidate_texts
        ]
        reranked = reranker.rerank(state["query"], candidates, top_n=top_n)
        return {"reranked_candidate_ids": [c["id"] for c in reranked]}

    def apply_rules(state: ScreeningState) -> dict:
        job_rule = state["job_rule"]
        qualified, rejected, results = [], [], {}
        for cid in state["reranked_candidate_ids"]:
            profile = candidate_profiles.get(cid)
            if not profile:
                continue
            result = evaluate_candidate(job_rule, profile)
            results[cid] = result.to_dict()
            (qualified if result.status == "QUALIFIED" else rejected).append(cid)
        return {"qualified_ids": qualified, "rejected_ids": rejected, "screening_results": results}

    def route_after_rules(state: ScreeningState) -> str:
        # The real conditional edge: skip evidence collection + LLM
        # explanation entirely when nothing in the shortlist qualified.
        return "deep" if state["qualified_ids"] else "skip"

    def collect_evidence(state: ScreeningState) -> dict:
        job_rule = state["job_rule"]
        evidence = {}
        for cid in state["qualified_ids"]:
            sections = candidate_texts.get(cid, {}).get("sections", {})
            matched_skills = state["screening_results"][cid]["required_skills_matched"]
            snippets = {}
            for skill in matched_skills:
                for section_text in sections.values():
                    if skill.lower() in section_text.lower():
                        snippets[skill] = section_text
                        break
            evidence[cid] = snippets
        return {"evidence": evidence}

    def deep_evaluate(state: ScreeningState) -> dict:
        job_rule = state["job_rule"]
        explanations = dict(state.get("explanations", {}))
        for cid in state["qualified_ids"]:
            explanations[cid] = explainer.explain(
                job_rule, state["screening_results"][cid], state["evidence"].get(cid, {})
            )
        return {"explanations": explanations}

    def explain_rejected(state: ScreeningState) -> dict:
        job_rule = state["job_rule"]
        explanations = dict(state.get("explanations", {}))
        for cid in state["rejected_ids"]:
            explanations[cid] = explainer.explain(job_rule, state["screening_results"][cid], {})
        return {"explanations": explanations}

    def generate_report(state: ScreeningState) -> dict:
        report = []
        for cid in state["qualified_ids"] + state["rejected_ids"]:
            result = state["screening_results"][cid]
            report.append({
                "candidate_id": cid,
                "status": result["status"],
                "score": result["final_score"],
                "explanation": state.get("explanations", {}).get(cid, ""),
                "evidence": state.get("evidence", {}).get(cid, {}),
            })
        report.sort(key=lambda r: r["score"], reverse=True)
        return {"final_report": report}

    graph = StateGraph(ScreeningState)
    graph.add_node("parse_job", parse_job)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("apply_rules", apply_rules)
    graph.add_node("collect_evidence", collect_evidence)
    graph.add_node("deep_evaluate", deep_evaluate)
    graph.add_node("explain_rejected", explain_rejected)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("parse_job")
    graph.add_edge("parse_job", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "apply_rules")
    graph.add_conditional_edges(
        "apply_rules", route_after_rules, {"deep": "collect_evidence", "skip": "explain_rejected"}
    )
    graph.add_edge("collect_evidence", "deep_evaluate")
    graph.add_edge("deep_evaluate", "explain_rejected")
    graph.add_edge("explain_rejected", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
