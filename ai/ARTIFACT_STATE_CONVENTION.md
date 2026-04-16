---
name: ARTIFACT_STATE_CONVENTION
description: Standard file, state, and dependency convention for the AI paper writing framework
---

ROLE:
Define how prompts and runners read, write, and track artifacts.

CORE PRINCIPLES:
- Each stage writes explicit artifacts
- Each section has an authoritative current file
- State is stored centrally and per section
- Runners read latest authoritative artifacts by default
- Failures are recorded explicitly, not silently ignored

DIRECTORY STRUCTURE:
ai/out/
  state/
  literature/
  abstract/
  introduction/
  related_work/
  methodology/
  results/
  positioning/
  final/

SECTION ARTIFACT PATTERN:
- *_draft.md
- *_audit.md
- *_issues.md
- *_filter.md
- *_revised.md
- *_final.md
- *.meta.json

GLOBAL STATE:
ai/out/state/pipeline_state.json

SECTION STATUS VOCABULARY:
- NOT_RUN
- IN_PROGRESS
- READY
- BORDERLINE
- NOT_READY
- BLOCKED

READ PRIORITY:
1. *_final.md
2. *_revised.md
3. *_draft.md
4. generate from source

WRITE RULES:
- runners must update both artifact files and state metadata
- failures must produce explicit blocked artifacts
- final files are authoritative outputs

PROMOTION RULE:
Only promote to *_final.md when:
- no critical issues remain
- section status is READY or accepted BORDERLINE
- promotion is explicit

DEPENDENCY RULE:
If a required upstream artifact is missing:
- stop
- record BLOCKED status
- do not fabricate downstream outputs

INLINE-ONLY EXECUTION IS INVALID.

If a stage produces conclusions in context but does not write its required artifacts:
- status = FAILED
- do not mark the stage complete
- do not update downstream readiness

## NON-NEGOTIABLE EXECUTION RULE

All stages MUST write their declared outputs to the ai/out/ directory on the local machine.

The filesystem is the single source of truth for pipeline state. This framework is designed for reproducibility, debugging, partial reruns, audit trails, modular validation, and resuming after interruption — not for one-shot inline generation.

---

## COMPLETION CRITERIA

A stage is considered COMPLETE only if:

1. The expected output file exists in ai/out/
2. The file is non-empty and contains the generated content
3. (If applicable) metadata/state files are updated consistently

---

## INVALID EXECUTION

Producing results only in the model context (inline) without writing to ai/out/ is INVALID and MUST NOT be treated as a completed stage.

---

## FAILURE HANDLING

If a stage does not write its required output file:

- status MUST be set to FAILED (not COMPLETE)
- downstream stages MUST NOT proceed
- the stage MUST be considered not executed

---

## EXECUTION VALIDATION (MANDATORY)

After each stage execution, the system MUST verify:

- Does the expected file exist in ai/out/?
- Is the file populated with content?

If verification fails → enforce FAILURE HANDLING

---

## SOURCE OF TRUTH

Pipeline correctness is determined exclusively by filesystem artifacts, not by:
- model memory
- conversation history
- inline outputs

