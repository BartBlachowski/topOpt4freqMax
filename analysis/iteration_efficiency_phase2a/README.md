# Iteration-efficiency Phase 2A harness

This directory is the isolated implementation namespace for the frozen iteration-efficiency
study. It does not contain production scientific results and cannot write into
`examples/Performance/final_campaign` or any methodology/audit directory.

The authoritative contract is `iteration_efficiency_contract.json`. Run the no-production
test suite from MATLAB R2025b with:

```matlab
cd analysis/iteration_efficiency_phase2a
run_phase2a_tests
```

`iteration_efficiency_campaign.m` is the eventual manual entry point. It intentionally
fails closed until a pre-production review supplies the explicit authorization token in the
script and every preflight check passes.

