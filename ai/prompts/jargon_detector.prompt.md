TASK: Detect and reduce domain-specific jargon that harms readability for non-AI experts.

CONTEXT:
You are reviewing a scientific paper in civil engineering / structural health monitoring that uses AI methods.
The target reader is NOT a machine learning specialist.

STYLE LOCK:
Follow ai/STYLE_LOCK.md strictly.
This includes:

* minimal edits only
* preserve meaning exactly
* do not increase verbosity
* do not rewrite the whole text

---

STEP 1 — IDENTIFY JARGON

Scan the text and list phrases that are:

* highly specific to machine learning
* unlikely to be understood by a non-AI domain expert
* compress multiple technical ideas into one phrase

Flag especially:

* multi-word technical constructs (e.g., "conditioning signal type")
* internal training mechanics (e.g., "gradient coupling")
* experimental shorthand (e.g., "five-variant ablation")
* invented or non-standard terms (e.g., "oracle-filter")

For each flagged phrase, provide:

[PHRASE]
[WHY IT IS JARGON] (1 sentence)

--------------------------------------------------

EXAMPLES (CALIBRATION)

EXAMPLE — BAD:
"A five-condition ablation isolates conditioning signal type..."

→ contains ML-internal jargon and unclear meaning → HIGH severity

EXAMPLE — GOOD:
"Controlled experiments evaluate how component information affects damage detection."

→ clear, functional, domain-accessible

---

STEP 2 — CLASSIFY SEVERITY

For each phrase, assign:

* LOW → acceptable in abstract (e.g., "ablation", "mIoU")
* MEDIUM → understandable but should be simplified
* HIGH → unclear to non-AI readers → must be rewritten

---

STEP 3 — SUGGEST REWRITES

For each MEDIUM or HIGH item:

Provide a replacement that:

* preserves exact meaning
* expresses FUNCTION, not mechanism
* reduces cognitive load
* avoids introducing new jargon

Format:

ORIGINAL:
...

REWRITE:
...

---

STEP 4 — CHECK SENTENCE OVERLOAD

Identify sentences that:

* contain ≥2 jargon phrases
* or mix multiple technical concepts

For each:

[ORIGINAL SENTENCE]

[ISSUE] (why it's overloaded)

[SIMPLIFIED VERSION]

---

STEP 5 — OUTPUT RULES

* Do NOT rewrite the entire paragraph
* Do NOT introduce new claims
* Do NOT remove scientific content
* Keep edits minimal and local
* Prefer clarity over technical completeness

GOAL:
A domain expert (civil engineer) should understand each sentence on first read.
