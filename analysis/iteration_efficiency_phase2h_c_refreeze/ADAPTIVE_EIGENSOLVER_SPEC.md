# Adaptive eigensolver specification

Start with the lowest 3 modes. If no unanimously valid structural mode is found, request
6, 12, 24, 48, and so on. Stop only when a valid mode is selected or a technical matrix,
resource, or eigensolver limit prevents continuation. There is no scientific mode ceiling.

Modes are sorted by eigenvalue after each solve. The implementation uses a deterministic
initial vector and validates every eigenpair. `modes_requested_final` and
`escalation_count` are evidence fields. A technical test-only cap may be injected to test
failure behavior; it is not a scientific acceptance parameter.

