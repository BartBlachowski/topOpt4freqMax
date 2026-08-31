# Phase 2H controlled refreeze record

Date: 2026-08-31 (Europe/Warsaw)  
Starting branch: `benchmark-methodology-r2`  
Starting HEAD: `632e9b01811845709de33f93051fd853373ed5e1`

Decision: Candidate C is implemented and formally refrozen for qualification. Production
is not authorized.

## Frozen identities

- Contract: `cc900b4ad4cae18b0bcd9b7a559f51e04e5167db587f64180b371d3c399bf95b`
- Evaluator: `e14a21efe0bb2d9b9d7f3187b4c3f671ec089f6ff96773074b8f3b56cacd79e9`
- Normative manifest: `ceb55dd650f9751d499c19da316571a7ab0c34b3ef2d943b657a817575194f2d`
- Candidate/classifier: `C` / `candidate_c_unanimous_v1`
- Principal/secondary routes: `Olhoff-LP` / `Olhoff-MMA`

The protected native source hashes remain:

- Proposed: `6d9ea66fcc27f63b7380708b5735552b5d9f2885d3e65714af572daccdae72b2`
- Yuksel: `5afc3d16b4ed6af05793df461b541ed3b2ea62a6da8836f38301a9a3917e6ba2`
- Olhoff stabilized LP: `95240cf60f82b40f8e5e892b9eea9b20a8fd3744b5eca6fdfc8dde2698d82aec`
- LP backend: `7724753c02f84d6009c3998f758d5b3f9c5144ad39ca6f470584a2c99e089465`
- Nested-MMA source: `22d4e04e4afde4e3f88b81f88a71a03a0fd0b6b313b022692dbddd48469fbe5e`

## Freeze conditions

The evaluator, contract, normative documents, and preflight are bound by SHA-256. Candidate
C tests and stored-evidence regressions pass. Binary D is excluded from Q. The hard topology
gate and all prior constants are unchanged. The non-numerical Olhoff selector produces
separate route rows and accounting.

Precision, cross-method, and reference-length qualifications are explicitly not executed.
Preflight must therefore fail closed. A future qualification may change only the three
status fields from absent/fail to pass through correctly bound evidence; it may not alter
this scientific definition silently.
