---
name: write_methodology
description: Write a precise, implementation-aligned Methodology section grounded in actual code and manuscript
---

ROLE:
Write the Methodology section of the paper with strict alignment to the actual implementation.

INPUT:
- Use the currently focused manuscript
- Use the project codebase (full access via VSCode)
- Use references/ when needed for attribution or comparison

ALWAYS LOAD:
- ai/STYLE_LOCK.md

MISSION:
Produce a Methodology section that is:
- technically precise
- consistent with implementation
- reproducible
- reviewer-verifiable

DO NOT:
- invent methods not present in code
- describe algorithms abstractly if concrete implementation exists
- oversimplify critical steps
- include irrelevant implementation details
- repeat Introduction or Results

CORE PRINCIPLE:
Every methodological claim must be traceable to either:
- code
- equations
- or clearly stated assumptions

---

## OPTIMIZER DISCLOSURE RULE

If the method uses standard algorithms (e.g., OC, MMA, Adam, SGD):

You MUST:

- describe them at a high level only
- state their role as numerical solvers
- explicitly clarify whether they are modified or not

You MUST NOT:

- include full equations of standard algorithms
- present textbook update rules as part of the method
- imply novelty in the optimizer if none exists

---

### Allowed:

“The optimization is performed using standard gradient-based methods such as OC or MMA, which are used here as numerical backends.”

---

### Allowed (if needed):

“A standard OC update scheme is used without modification.”

---

### Forbidden:

- full OC update equation
- full MMA formulation
- detailed derivation of standard optimizers

---

### Exception:

You MAY include optimizer equations ONLY IF:

- the optimizer is modified
- OR the optimizer itself is a contribution

In that case:
- clearly state what is modified
- highlight the difference from the standard form

---

# CORE TASKS

## 1. METHOD IDENTIFICATION

From manuscript + code:

Identify:
- main method / framework
- key components
- computational flow
- inputs / outputs

Output internally:
- method structure map

---

## 2. CODE ALIGNMENT

Inspect implementation:

- main functions / entry points
- solver structure
- data flow
- parameters and defaults
- special mechanisms (e.g., filtering, aggregation, constraints)

Check:
- what is actually implemented vs described

Flag internally:
- mismatches
- hidden steps
- implicit assumptions

---

## 3. METHOD DECOMPOSITION

Break method into logical components:

Example:

- problem formulation
- model representation
- objective / constraints
- numerical procedure
- special mechanisms (e.g., aggregation, stabilization)

Each component must:
- have clear role
- connect to code

---

## 4. MATHEMATICAL FORMULATION

Where appropriate:

- define variables
- define objective function
- define constraints
- define key operators

Rules:
- only include math that is actually used
- match notation with code logic
- avoid symbolic decoration

---

## 5. ALGORITHMIC FLOW

Describe:

- sequence of operations
- interaction between components
- iteration / convergence logic

Avoid:
- pseudo-code overload unless necessary
- trivial steps

---

## 6. SPECIAL MECHANISMS

Explicitly describe:

- aggregation strategies (e.g., envelope vs average)
- filtering / regularization
- stabilization techniques
- coupling between models

These are often:
👉 the real contribution

---

## 7. IMPLEMENTATION CLARITY

Include:

- key parameter definitions
- important assumptions
- boundary conditions (if relevant)
- model simplifications

Avoid:
- low-level code syntax
- unnecessary constants

---

## 8. CONSISTENCY CHECK

Ensure:

- terminology matches Introduction
- method matches contributions
- no contradiction with Results

---

# STRUCTURE

Write 4–6 subsections:

## 1. Problem formulation
- define problem clearly
- connect to engineering context

## 2. Model representation
- describe system (e.g., frame + solid)
- define abstraction levels

## 3. Optimization / solution strategy
- objective and constraints
- main solver logic

## 4. Multi-case / multi-configuration handling
- how variability is handled
- aggregation strategies

## 5. Numerical implementation
- key algorithmic details
- important parameters

## 6. (Optional) Additional mechanisms
- buckling handling
- stabilization
- coupling

---

# STYLE RULES

- precise and technical
- avoid storytelling
- avoid marketing language
- avoid “we simply”
- avoid unnecessary adjectives

---

# OUTPUT

## --- METHODOLOGY ---
<final section>

---

## --- STRUCTURE MAP ---

- subsection 1:
- subsection 2:
- subsection 3:
- subsection 4:
- subsection 5:
- optional:

---

## --- CODE TRACEABILITY MAP ---

For each major component:

- Component:
- Code location:
- Role in method:
- Mentioned in text: YES / NO

---

## --- SIMPLIFICATIONS / ASSUMPTIONS ---

- ...

---

## --- POTENTIAL MISMATCHES ---

- paper says:
- code does:
- risk level:

---

## --- REVIEWER RISK ---

- unclear steps:
- missing definitions:
- likely objections:

---

## --- SELF-CHECK ---

Confirm:
- All major steps are grounded in implementation
- No invented algorithmic elements
- Mathematical formulation matches code logic
- Method is reproducible from description