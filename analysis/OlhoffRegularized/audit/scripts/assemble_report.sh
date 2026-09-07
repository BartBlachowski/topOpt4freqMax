#!/bin/bash
# Assemble AUDIT_REPORT.md from its parts.  Sections 0-8 live in the report
# itself; 9-11 and 12-15 + appendices are appended from audit/results/.
set -e
cd "$(dirname "$0")/../../"      # analysis/OlhoffRegularized
R=AUDIT_REPORT.md
A=audit/results
[ -f "$A/report_wp9_11.md" ] || { echo "missing $A/report_wp9_11.md"; exit 1; }
python3 - "$R" "$A/report_wp9_11.md" "$A/report_wp6_partial.md" "$A/report_tail.md" <<'PY'
import sys,re
rep,wp9,wp6,tail=sys.argv[1:5]
s=open(rep).read()
# drop anything previously appended after section 8
i=s.find('\n## 9. ')
if i>0: s=s[:i]
s=s.rstrip('\n')+'\n'+open(wp9).read().rstrip('\n')+'\n'+open(wp6).read().rstrip('\n')+'\n'+open(tail).read()
open(rep,'w').write(s)
print("assembled:",rep, len(s.splitlines()), "lines")
PY
