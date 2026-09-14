# Controller generalization

**THREE_RUNG_CONTROLLER_CROSS_MESH_INCONCLUSIVE.** None of the nine campaign jobs executed the controller under test. Neither generalization nor its failure can be inferred from this campaign. The first problematic mesh is 160×20 by policy identity; among legacy refinements the first loss of even reaching move=0.01 is 320×40.

Actual legacy paths: 160/240 end at stage 3, move 0.01; the remaining seven end at stage 2, move 0.02. Stage-1 start is 1 for each run. The logs identify final stage starts at outer 90,103,130,138,163,189,198,222,169 respectively, followed immediately by stopping at 91,104,131,139,164,190,199,223,170. Earlier beta declarations on 160/240 cannot be reconstructed from the final archive. They must not be labelled S1/S2 E events. A/B histories, S1/S2/S3 declarations, and first terminal persistent E are **not applicable / not recorded**, not zero.

## What survives in historical evidence

Independent offline replay from stored RHO and DRHO gives:

| Mesh | Stage | Start | E declaration | Duration | Branch | Amplitude/tol | Median cosine | Median net/path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160x20 | 1 | 1 | 102 | 102 | A | 12.59299405822477 | -0.906580232393946 | 0.1329617713293757 |
| 160x20 | 2 | 103 | 141 | 39 | A | 2.189388817418995 | -0.8567335705633676 | 0.22610186014893702 |
| 160x20 | 3 | 142 | 180 | 39 | B | 0.10355194206462318 | 0.5681067723829708 | 0.8703066340589276 |
| 240x30 | 1 | 1 | 206 | 206 | B | 0.5092684413691927 | 0.9900339146129635 | 0.9404167100978069 |
| 240x30 | 2 | 207 | 245 | 39 | B | 0.14353654336026844 | 0.9970628571652578 | 0.9874428461762422 |
| 240x30 | 3 | 246 | 284 | 39 | B | 0.047710572371152525 | 0.78613360885388 | 0.8920278532743366 |
| 320x40 | 1 | 1 | 274 | 274 | A | 1.031261433441822 | -0.6538842699968497 | 0.4211376008333546 |
| 320x40 | 2 | 275 | 313 | 39 | B | 0.10803331097620686 | 0.9999649239133963 | 0.9997497672502524 |
| 320x40 | 3 | 314 | 352 | 39 | B | 0.03273705295769494 | 0.5941707234454476 | 0.7527244130517496 |
| 400x50 | 1 | 1 | 388 | 388 | B | 0.6398205937160605 | 0.9981070691081662 | 0.9934647291640188 |
| 400x50 | 2 | 389 | 427 | 39 | B | 0.144496976535398 | 0.9996382946631228 | 0.9987870495268008 |
| 400x50 | 3 | 428 | 466 | 39 | B | 0.06328797679442143 | 0.19069843764054326 | 0.5666592161767758 |


Every per-iteration A, B, nA and nB matches the historical CSV exactly. Definitions are A: median20 cosine<0 AND median20 net/path<0.5 AND ||drho||≥tol; B: ||drho||<tol AND median20 cosine>0. Each branch must persist for 20 consecutive iterations; alternating A/B is not sufficient. Windows are reset locally at a stage transition. Net/path spans 10 increments. The normalized threshold tol/sqrt(NE)=0.00088388347648 is mesh-independent in RMS units; it is not numerically relaxed under refinement, although the number/spatial concentration of moving elements can alter its effectiveness.

Coarse historical stage-1 declarations grow 102→206→274→388; branch sequence is A/B/A/B. C320 S1 lies only 3.13% above the amplitude boundary, a real near-threshold observation. At all historical S3 endpoints B fires with amplitude only 3.27–10.36% of tol, while median cosine ranges 0.191–0.786: mature in amplitude, but not a stationarity proof. A-terminal stopping would explicitly accept persistent nonzero cancellation, not conventional design convergence.

Every historical stage 2 and stage 3 lasts exactly **39 updates**, the minimum from a 20-step median plus 20-step persistence. Consequently the existing four meshes do not establish that adaptive late-stage declaration timing buys anything over two fixed 39-update dwells after the validated S1 event. They also do not establish that those dwells work at 480–800. Fine-mesh low-amplitude cancellation holes, threshold trends, accidental terminal events and asymptotic coherence cannot be tested without fine-mesh trajectories of the intended policy.

Figures [F08](figures/F08_historical_E_declarations.png) and [F09](figures/F09_historical_branch_map.png) deliberately label historical data and absent fine data. No E events were invented for the campaign.
