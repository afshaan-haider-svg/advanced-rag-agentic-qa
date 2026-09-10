"""Citation and grounding verifier."""
import re
from typing import Dict, List

def verify_answer(answer: str, context: str, citations: List[dict], document_related: bool = True) -> Dict:
    if not document_related:
        cleaned = re.sub(r"\s*\[\d+\]", "", answer).strip()
        return {"citation_check_passed": True, "unsupported_claims": [], "verified_answer": cleaned}
    valid = {int(c["citation_number"]) for c in citations if "citation_number" in c}
    used = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    invalid = sorted(used - valid)
    unsupported = [f"Invalid citation [{n}]" for n in invalid]
    if invalid:
        return {"citation_check_passed": False, "unsupported_claims": unsupported,
                "verified_answer": "I could not verify the generated answer against the retrieved sources, so I won't present it as grounded."}
    if citations and not used:
        return {"citation_check_passed": False, "unsupported_claims": ["Answer contains no source citation."],
                "verified_answer": "I found relevant context, but could not produce a citation-verified answer."}
    return {"citation_check_passed": True, "unsupported_claims": [], "verified_answer": answer}
