# WP2 — Filter-radius semantics
NESTED-MMA ROUTE AUDIT — READ-ONLY

## The question

`BASE_mma_160x20.log` prints, in its configuration echo:

    rminEl: 3
    rminPhys: 0.0600

Given that filter scale is the strongest isolated basin selector identified by the
`OLHOFFEXACT_FAILURE_POSTMORTEM`, these two values must not be left ambiguous. Taken at face
value the header appears to say the run used a three-element filter — which would place it in
the *failing* large-radius family (`rmin = 2.5` gives `omega1 = 159.49`, no coalescence).

## Resolution, from source

`Matlab/reproduction2007/algo/olhoffOpt.m`, immediately after `model2D`:

    if isfield(cfg,'rminPhys') && ~isempty(cfg.rminPhys) && cfg.rminPhys > 0
        dyEl = cfg.b/cfg.nely;
        cfg.rminEl = cfg.rminPhys/dyEl;
    end
    flt = prepFilter(cfg.nelx, cfg.nely, cfg.rminEl);

So:

| field | meaning |
|---|---|
| `rminEl` | filter radius in **element units**. This is the ONLY quantity that reaches the filter: `prepFilter(nelx, nely, rmin)` documents "rmin is in ELEMENT units" and computes weights `max(0, rmin - sqrt((i1-i2)^2+(j1-j2)^2))` on the element index lattice. |
| `rminPhys` | filter radius in **physical units**. It is not used directly. When set and positive it **overrides** `rminEl` by dividing by the element height `dyEl = b/nely`. |

For this run: `dyEl = 1/20 = 0.05`, so

    rminEl_effective = 0.0600 / 0.05 = 1.2 elements

**The `rminEl: 3` in the log header is a stale display value that was never used.** It is
printed by `run_case.m`, which does `disp(cfg)` *before* calling `olhoffOpt`, so the echo
shows the pre-override configuration. The override happens inside `olhoffOpt` on a local
copy.

## Independent empirical confirmation

`fm_mma_diag.mat` was run with the same inner solver, filter mode, move, multiplicity
tolerance, mass law, mesh and supports, and its **saved** `res.cfg` records:

    rminEl   = 1.200
    rminPhys = []          (emptied)

Its trajectory is **bit-identical** to BASE_mma over all 400 shared outer iterations
(`nInner`, `cumInner` and `N` match exactly; `omega1` and `max|drho|` match to the log's
printed precision). Two runs, one storing the pre-override representation and one the
post-override representation, producing the same numbers, is direct evidence that the
effective radius is 1.2 and not 3.

## Consequences

1. **BASE_mma sits in the successful bimodal filter band.** The controlled LP sweep at
   160x20 gives coalescence for `rmin` in 1.1–1.5 and no coalescence at 2.0 and above. At
   1.2 the matched LP run reaches `omega1 = 168.240` with a 0.217% gap. BASE_mma's filter is
   the same.
2. **This is a principal reason BASE_mma behaves better than the legacy nested-MMA
   failures**, which used `rmin = 2.5`. It is not "MMA was fixed".
3. **The log is misleading and must not be quoted as configuration evidence.** Any table
   reporting this run's filter radius must state 1.2 elements (0.06 physical), citing
   `res.cfg` or the override rule, never the header echo.
4. Internal consistency: the two printed fields are mutually inconsistent as displayed
   (3 elements would be 0.15 physical at this mesh, not 0.06). The inconsistency is a
   display defect in `run_case.m`, not a numerical defect in `olhoffOpt.m`.

**Recommendation (not implemented here):** `run_case.m` should echo the configuration
*after* the override, or `olhoffOpt.m` should perform the override before the echo. This is
a logging change only and does not affect any recorded numerical result.
