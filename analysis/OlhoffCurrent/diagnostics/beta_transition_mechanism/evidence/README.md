# Evidence for this study

This audit ran no optimisation and produced no new raw trajectories.

Its conclusions nevertheless depend on element-level evidence — the bound-active
population at 400x50 is computed from `DRHO`, which no CSV contains. That
evidence lives in the single durable evidence root,

    analysis/OlhoffCurrent/evidence/move_activity_400/

and is declared `required` in this study's `EVIDENCE.json`, so
`olhoffcurrent_evidence_gate` fails **this** study as well if either file goes
missing or is altered. The files are referenced, not duplicated: copying a
138 MB artifact per consuming study would multiply the thing most likely to
drift.
