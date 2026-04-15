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