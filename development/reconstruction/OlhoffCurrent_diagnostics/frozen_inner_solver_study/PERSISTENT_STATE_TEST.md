# Persistence test

PERSISTENT_STATE_NO_MATERIAL_EFFECT

Source inspection proves all required MMA approximation history persists within
B0. S1 serializes x, xold1, xold2, low, upp and the counter at call 19, restores
them, and continues the same fixed problem. Its call-50 vector matches B0
bit-for-bit. The state transfer is an identity, not an improvement.

Across new outer problems production resets this state. That is a separate,
untested warm-start question and cannot explain a reset between calls that does
not occur. The Newton workspace inside subsolv is reset every approximate solve,
as in the official algorithm; approximate-solve accuracy is studied separately.
innerLoopRho cannot isolate persistence because it changes coordinates and omits
dOff. See the pre-experiment semantic diff in RECONSTRUCTION_CHOICES.md.
