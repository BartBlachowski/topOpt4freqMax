ISSUE SCHEMA

For every issue, report:

- ID:
- Location:
- Section:
- Type:
  - scientific_accuracy
  - unsupported_claim
  - inconsistency
  - quantitative_overload
  - readability
  - style
  - reviewer_risk
  - structure
  - redundancy
- Severity:
  - critical
  - major
  - minor
- Description:
- Why it matters:
- Evidence:
- Recommended action:
  - fix_now
  - fix_if_time
  - report_only
  - do_not_change
- Safe patch:
- Confidence:
  - high
  - medium
  - low

  MAPPING:

CRITICAL → fix_now → ESSENTIAL
MAJOR → fix_if_time → OPTIONAL
MINOR → report_only → OPTIONAL

HARMFUL → always remove (override)