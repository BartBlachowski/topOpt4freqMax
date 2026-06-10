---
name: results_manifest_schema
description: Defines the required structure for paper-specific numerical results manifests
---

ROLE:
This schema defines how a paper specifies its authoritative numerical result sources.

PURPOSE:
The results manifest tells the AI system:
- where the valid numerical results are
- which experiments matter
- which metrics are primary
- which files must be ignored
- how results should be compared and reported

CORE PRINCIPLE:
The manifest is the source of truth for result selection.
If a file or experiment is not authorized by the manifest, it must not be used in the numerical results subsection.

---

# REQUIRED SECTIONS

## 1. PAPER CONTEXT

Fields:
- paper_id
- paper_title
- domain
- result_scope

Purpose:
Identify which paper the manifest belongs to and what kind of numerical results are expected.

Example:
- paper_id: arm_z_topopt
- domain: topology optimization / robotics
- result_scope: numerical structural optimization results

---

## 2. AUTHORITATIVE SOURCES

Fields:
- authoritative_files
- authoritative_directories
- preferred_formats

Purpose:
List the files and folders that contain valid numerical results.

Rules:
- paths must be relative to repository root
- only these paths may be used by write_numerical_results.md unless explicitly expanded later

---

## 3. EXCLUDED SOURCES

Fields:
- ignored_files
- ignored_directories
- stale_patterns

Purpose:
Prevent use of:
- debug outputs
- archived results
- preliminary runs
- temporary files
- manually superseded versions

Example stale patterns:
- "*debug*"
- "*tmp*"
- "*old*"
- "*draft*"
- "*v1*" when replaced by final validated output

---

## 4. EXPERIMENT GROUPS

Each experiment group must define:
- id
- name
- purpose
- source_files
- status
- include_in_paper

Allowed status values:
- final
- exploratory
- deprecated
- failed

Allowed include_in_paper values:
- yes
- no

Purpose:
Group files into interpretable experiment blocks such as:
- baseline comparison
- ablation study
- sensitivity analysis
- robustness check

---

## 5. PRIMARY METRICS

For each metric define:
- id
- name
- meaning
- unit
- higher_is_better / lower_is_better / target_range
- reporting_priority

Purpose:
Tell the system which metrics matter most and how to interpret them.

Examples:
- volume_reduction [%]
- max_stress [MPa]
- tip_displacement [mm]
- buckling_factor [-]
- mIoU [%]
- Dice [-]

---

## 6. REPORTING PRIORITY

Fields:
- primary_result
- primary_comparison
- secondary_results
- optional_results

Purpose:
Tell write_numerical_results.md what to report first.

Example:
1. main engineering outcome
2. strategy comparison
3. robustness result

---

## 7. COMPARISON RULES

Fields:
- valid_comparisons
- forbidden_comparisons
- aggregation_rules
- final_run_rule

Purpose:
Prevent invalid or misleading comparisons.

Examples:
- compare only experiments with same mesh resolution
- compare only final validated runs
- do not average across different random seeds unless specified
- do not compare exploratory runs to final reported runs

---

## 8. CLAIM LINKS

Fields:
- contribution_to_result_map
- claim_to_metric_map

Purpose:
Connect result reporting to the actual paper claims.

Example:
- claim: envelope approach is more conservative
- supporting metric: material volume / stress distribution
- experiment group: strategy_comparison

---

## 9. WRITING CONSTRAINTS

Fields:
- max_key_findings
- max_metrics_per_paragraph
- must_report
- must_not_report
- interpretation_notes

Purpose:
Control the style and selectivity of the generated numerical results subsection.

---

# VALIDATION RULES

## Required:
- at least one authoritative source
- at least one experiment group with include_in_paper = yes
- at least one primary metric
- at least one reporting priority entry

## Forbidden:
- empty authoritative source list
- using ignored paths as authoritative
- experiment groups with source files outside declared authoritative sources

---

# OUTPUT EXPECTATION

A valid manifest must enable:
- deterministic selection of result files
- deterministic exclusion of stale/debug files
- prioritized reporting of findings
- safe comparison logic

---

# IMPLEMENTATION NOTE

The manifest should be stored as:

- ai/config/results_manifest.json

Optional human-readable companion:
- ai/config/results_manifest.md