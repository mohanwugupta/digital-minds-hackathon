# Track A — factorial incentive trajectory across layers

The analysis contains **1187 complete states**, **48 episode clusters**, and **14244 factorial rows**.
All projections use the independently trained, frozen persistence probe from the corresponding layer; no probe was retrained on factorial data.
Source coverage: **14244/14244 cells** and **1187/1187 states**.


## Detectability

- STOP first has the expected negative sign with a nominal two-sided 95% clustered interval excluding zero at layer **3**; the first Holm-corrected layer is **4** and the first sustained nominal layer is **23**.
- CONTINUE first has the expected positive sign with a nominal two-sided 95% clustered interval excluding zero at layer **0**; the first Holm-corrected layer is **0** and the first sustained nominal layer is **6**.
- Holm–Bonferroni correction is applied separately to the 32 layer tests for each incentive effect.

## Evolution toward the final layer

- At layer 31, the STOP raw slope is **-0.0901045** and its magnitude is **0.867×** the behavioral persistence-logit slope.
- At layer 31, the CONTINUE raw slope is **0.0962823** and its magnitude is **0.898×** the behavioral persistence-logit slope.
- Relative incentive reaches 25%, 50%, and 75% of its layer-31 R² at layers **13**, **15**, and **15**. Final within-state R² is **0.751**.

These onset layers identify where incentive-induced differences become expressed along frozen persistence directions. They do not identify where persistence is computed.
