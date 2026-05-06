"""
Faithfulness self-check (lightweight, no extra model required).

Algorithm
---------
1. Split the generated answer into individual claims (sentences).
2. For each claim compute token-level overlap with the concatenated context.
3. Flag claims with overlap below `min_overlap` as potentially unsupported.
4. Compute both a mean overlap score AND an ungrounded-claim ratio.
5. Verdict rules (two-signal approach):
     pass  — mean >= pass_threshold  AND  ungrounded_ratio < warn_ratio
     warn  — mean >= fail_threshold  OR   ungrounded_ratio >= warn_ratio
     fail  — mean < fail_threshold   AND  ungrounded_ratio >= fail_ratio
   The dual signal catches partial grounding (some claims hallucinated,
   some well-supported) which the mean score alone would mask.

Failure modes (honest):
- High-overlap answers can still be wrong if the context is wrong.
- Very short claims (e.g. "Yes.") always score 0 — they are filtered out.
- Token overlap is a rough proxy; a proper faithfulness model (e.g. TRUE,
  MiniCheck) would be more accurate at the cost of an extra inference call.
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 15]


def _token_overlap(claim: str, reference: str) -> float:
    def tok(t):
        return set(re.sub(r"[^\w\s]", " ", t.lower()).split())

    ct, rt = tok(claim), tok(reference)
    if not ct:
        return 0.0
    return len(ct & rt) / len(ct)


class SelfChecker:
    def __init__(
        self,
        min_overlap: float = 0.12,
        pass_threshold: float = 0.30,
        fail_threshold: float = 0.12,
        warn_ratio: float = 0.40,   # ≥40% ungrounded claims → at most "warn"
        fail_ratio: float = 0.80,   # ≥80% ungrounded claims → "fail"
    ):
        self.min_overlap = min_overlap
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold
        self.warn_ratio = warn_ratio
        self.fail_ratio = fail_ratio

    def check(self, answer: str, contexts: List[dict]) -> dict:
        """
        Returns:
            score            – mean per-claim overlap in [0, 1]
            ungrounded_ratio – fraction of claims below min_overlap
            grounded         – bool
            issues           – list of low-overlap claim snippets (max 3)
            verdict          – "pass" | "warn" | "fail"
            need_regen       – bool — set True to trigger a second generation pass
            claim_count      – number of claims evaluated
        """
        if not answer:
            return _verdict(0.0, 1.0, ["empty answer"], 0)

        reference = " ".join(
            c.get("parent_text", c.get("text", "")) for c in contexts
        )
        if not reference.strip():
            return _verdict(0.0, 1.0, ["no context provided"], 0)

        claims = _sentences(answer)
        if not claims:
            return _verdict(1.0, 0.0, [], 0)  # too short to evaluate

        scores, issues = [], []
        ungrounded = 0
        for claim in claims:
            ov = _token_overlap(claim, reference)
            scores.append(ov)
            if ov < self.min_overlap:
                ungrounded += 1
                issues.append(f"[ov={ov:.2f}] {claim[:70]}")

        avg = sum(scores) / len(scores)
        ung_ratio = ungrounded / len(claims)
        return _verdict(avg, ung_ratio, issues[:3], len(claims),
                        self.pass_threshold, self.fail_threshold,
                        self.warn_ratio, self.fail_ratio)

    def grounding_reminder(self) -> str:
        return (
            "STRICT MODE: Only answer using information explicitly present in the "
            "CONTEXT above. If you cannot find the answer, respond exactly: "
            "'The provided context does not contain enough information to answer this question.'"
        )


def _verdict(
    score: float,
    ungrounded_ratio: float,
    issues: List[str],
    claim_count: int,
    pass_threshold: float = 0.30,
    fail_threshold: float = 0.12,
    warn_ratio: float = 0.40,
    fail_ratio: float = 0.80,
) -> dict:
    """
    Two-signal verdict:
      fail  — score < fail_threshold  AND ung_ratio >= fail_ratio
      warn  — (score < pass_threshold)  OR  (ung_ratio >= warn_ratio)
      pass  — score >= pass_threshold  AND ung_ratio < warn_ratio
    """
    if score < fail_threshold and ungrounded_ratio >= fail_ratio:
        verdict, grounded, need_regen = "fail", False, True
    elif score < pass_threshold or ungrounded_ratio >= warn_ratio:
        verdict, grounded, need_regen = "warn", True, False
    else:
        verdict, grounded, need_regen = "pass", True, False
    return {
        "score": round(score, 3),
        "ungrounded_ratio": round(ungrounded_ratio, 3),
        "grounded": grounded,
        "issues": issues,
        "verdict": verdict,
        "need_regen": need_regen,
        "claim_count": claim_count,
    }
