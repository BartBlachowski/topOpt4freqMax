---
name: evaluation_schema
description: Defines the structured output that all evaluators must produce before report generation
---

The evaluator MUST output a structured object with the following fields.

## EXECUTIVE
- verdict: one of [ACCEPT, MINOR_REVISION, MAJOR_REVISION, REJECT]
- confidence: LOW | MEDIUM | HIGH
- summary: 3–5 sentences

## TOP_ISSUES (max 5)
Each issue:
- title
- severity: CRITICAL | MAJOR | MINOR
- description
- evidence
- required_fix

## SECTION_STATUS
For each: abstract, introduction, related_work, methodology, results, discussion, conclusion

Fields:
- status: READY | BORDERLINE | NOT_READY
- key_problem
- suggested_action

## CLAIM_VALIDATION
- claims_supported: YES | PARTIAL | NO
- unsupported_claims: list
- missing_evidence: list

## NOVELTY
- novelty_type: METHOD | APPLICATION | INTEGRATION | INCREMENTAL
- gap_valid: YES | WEAK | INVALID
- risk_of_rejection: LOW | MEDIUM | HIGH

## METHODOLOGY
- reproducibility: YES | PARTIAL | NO
- code_alignment: YES | PARTIAL | NO
- hidden_assumptions: list

## RESULTS
- sufficient_for_claims: YES | PARTIAL | NO
- missing_experiments: list
- comparison_quality: STRONG | WEAK | NONE

## CONSISTENCY
- internal_consistency: YES | MINOR_ISSUES | BROKEN
- contradictions: list

## FINAL_RECOMMENDATION
- minimal_acceptance_path
- must_fix
- should_fix
- optional
