# mma_published/ -- Svanberg's PUBLISHED September-2007 constants

`mma/mmasub.m` as received in this project is NOT byte-identical to Svanberg's
published "Version September 2007 (and a small change August 2008)" code: it
carries two local modifications of unknown origin (present in every other copy
in this user's project lineage, so inherited from a common ancestor):

    as-found (mma/)          published (this folder)
    move    = 1.0            move    = 0.5
    asyinit = 0.01           asyinit = 0.5

Verified against two independent verbatim copies of the published file
(gistmeto/EduTO `mmasub.m`, and arjendeetman's port of the smoptit.se
distribution). Everything else is identical; `subsolv.m` is an unmodified copy
of `mma/subsolv.m` (which matches the published version up to one added
commented-out line).

Per the user's decision (2026-09-03) `mma/` stays byte-identical as received;
the config switch `cfg.mmaVariant` ('published' = reproduction baseline |
'asfound') selects which copy `olhoffOpt` puts on the path. The variant and the
resolved path of `mmasub` are recorded in every run's `res`.

Same licence terms as `mma/`: keep local, do not commit to a public repo.
