---
name: results_manifest_validator
description: Validate the paper-specific results manifest before numerical result writing
---

ROLE:
Validate the paper-specific numerical results manifest.

INPUT:
- Use:
  - ai/config/results_manifest.json
- Inspect the local filesystem as needed to verify declared paths

ALWAYS LOAD:
- ai/STYLE_LOCK.md
- ai/config/results_manifest_schema.md
- ai/ARTIFACT_STATE_CONVENTION.md

MISSION:
Determine whether the results manifest is:
- structurally complete
- internally consistent
- consistent with the local filesystem
- sufficient for write_numerical_results.md to run safely

CORE PRINCIPLE:
The results manifest is the source of truth for numerical result selection.
If the manifest is invalid or ambiguous, numerical result writing must be blocked.

DO NOT:
- write the numerical results subsection
- guess missing manifest fields
- silently repair invalid configuration
- treat exploratory or ignored files as valid sources

---

# VALIDATION DIMENSIONS

## 1. FILE EXISTENCE

Check:
- Does ai/config/results_manifest.json exist?
- Is it readable?
- Is it non-empty?

If not:
- status = BLOCKED

---

## 2. SCHEMA COMPLETENESS

Validate required top-level sections:

- paper_context
- authoritative_sources
- excluded_sources
- experiment_groups
- primary_metrics
- reporting_priority
- comparison_rules
- claim_links
- writing_constraints

Detect:
- missing sections
- empty required sections
- malformed structures

---

## 3. AUTHORITATIVE SOURCE VALIDITY

Check:
- Do all declared authoritative files exist?
- Do all declared authoritative directories exist?
- Are paths relative and local?
- Are preferred formats reasonable and non-empty?

Detect:
- missing files
- broken paths
- impossible directories
- duplicate source declarations

---

## 4. EXCLUSION CONSISTENCY

Check:
- Are ignored files/directories distinct from authoritative sources?
- Do stale patterns overlap with authoritative files in a way that creates ambiguity?

Detect:
- same file both authoritative and ignored
- same directory both authoritative and ignored
- stale pattern likely to match an authoritative file

---

## 5. EXPERIMENT GROUP CONSISTENCY

For each experiment group, check:
- id exists
- name exists
- purpose exists
- source_files exist
- status is valid: final / exploratory / deprecated / failed
- include_in_paper is valid: yes / no

Check also:
- source_files belong to declared authoritative sources
- at least one experiment group has:
  - status = final
  - include_in_paper = yes

Detect:
- group with missing source files
- included group with non-final status
- group referencing undeclared files

---

## 6. METRIC CONSISTENCY

For each primary metric, check:
- id exists
- name exists
- meaning exists
- unit exists (or explicit unitless form)
- optimization_direction is valid:
  - higher_is_better
  - lower_is_better
  - target_range
- reporting_priority exists and is numeric/orderable

Detect:
- duplicate metric IDs
- missing interpretation
- invalid optimization direction
- no primary metric defined

---

## 7. REPORTING PRIORITY ADEQUACY

Check:
- primary_result exists
- primary_comparison exists
- secondary_results exists (can be empty list)
- optional_results exists (can be empty list)

Detect:
- no primary reporting logic
- vague or empty priority structure

---

## 8. COMPARISON RULE VALIDITY

Check:
- valid_comparisons exists
- forbidden_comparisons exists
- aggregation_rules exists
- final_run_rule exists

Detect:
- contradictory comparison rules
- missing final-run rule
- insufficient guidance to prevent invalid comparisons

---

## 9. CLAIM LINK COVERAGE

Check:
- contribution_to_result_map exists
- claim_to_metric_map exists

For each link:
- referenced experiment group exists
- referenced metrics exist

Detect:
- claim linked to missing metric
- claim linked to missing experiment group
- no claim support mapping at all

---

## 10. WRITING CONSTRAINT ADEQUACY

Check:
- max_key_findings exists
- max_metrics_per_paragraph exists
- must_report exists
- must_not_report exists
- interpretation_notes exists

Detect:
- missing constraints
- impossible values (e.g. 0 key findings)
- constraints that conflict with reporting priority

---

## 11. READINESS FOR DOWNSTREAM USE

Final check:
Can write_numerical_results.md run deterministically from this manifest?

Answer:
- YES
- PARTIAL
- NO

Criteria for YES:
- sources exist
- included experiments are clear
- primary metrics are defined
- comparison rules are usable
- no critical ambiguity remains

---

# OUTPUT

## --- MANIFEST VALIDATION SUMMARY ---

- Manifest file:
- Exists:
- Readable:
- Schema completeness:
- Source validity:
- Experiment consistency:
- Metric consistency:
- Comparison rules:
- Claim-link coverage:
- Downstream readiness:

---

## --- CRITICAL ISSUES ---

List only blocking issues.

1.
2.
3.
4.
5.

---

## --- NON-BLOCKING ISSUES ---

List issues that should be fixed but do not block execution.

1.
2.
3.

---

## --- PATH CHECK ---

### Authoritative files
- existing:
- missing:

### Authoritative directories
- existing:
- missing:

### Ignored paths conflicts
- none / list:

---

## --- EXPERIMENT GROUP CHECK ---

For each included experiment group:

- Group ID:
- Status:
- Include in paper:
- Source files valid: YES / NO
- Ready for reporting: YES / NO

---

## --- METRIC CHECK ---

For each primary metric:

- Metric ID:
- Valid definition: YES / NO
- Reporting priority valid: YES / NO
- Notes:

---

## --- CLAIM LINK CHECK ---

- Fully linked claims:
- Broken links:
- Missing mappings:

---

## --- VERDICT ---

Choose one:

- VALID
- VALID WITH WARNINGS
- INVALID
- BLOCKED

---

## --- REQUIRED ACTIONS ---

### MUST FIX
- ...

### SHOULD FIX
- ...

### OPTIONAL
- ...

---

## --- DOWNSTREAM DECISION ---

- write_numerical_results may proceed: YES / NO

If NO:
- block downstream stage
- do not infer missing configuration

---

## --- SELF-CHECK ---

Confirm:
- Validation was based on the manifest and filesystem
- No missing values were guessed
- Conflicts between authoritative and ignored sources were checked
- Downstream readiness was judged conservatively