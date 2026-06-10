---
name: issue_filter
description: Classify issues into essential/optional/harmful
---

Using the current issue list, classify each issue as:
- ESSENTIAL
- OPTIONAL
- HARMFUL TO CHANGE

Rules:
- ESSENTIAL = correctness, misleading framing, major readability break, likely reviewer attack
- OPTIONAL = polish only
- HARMFUL TO CHANGE = would risk style degradation, over-editing, voice loss, or unnecessary expansion

Output:
--- CHANGE FILTER ---
--- SAFE CHANGE SET ---