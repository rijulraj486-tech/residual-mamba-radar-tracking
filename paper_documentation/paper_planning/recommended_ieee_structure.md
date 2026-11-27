# Recommended six-page IEEE structure

| Section | Purpose and content | Equations / evidence | Figures / tables | Avoid | Approx. pages |
|---|---|---|---|---|---:|
| I. Introduction | Long-sequence radar state estimation problem and scoped contributions | concise state/observation context | pipeline or result preview | unsupported novelty/real-world robustness | 0.7 |
| II. Related Work | Causal sequence models, Mamba/state-space models, learned filtering, EKF/UKF | cite verified sources; none currently inventoried | none | claiming exhaustive review before bibliography audit | 0.5 |
| III. Problem Formulation | State order, CA F/Q, range-bearing h, noise and initializer | exact equations | radar schematic | stationary/realistic claims | 0.8 |
| IV. Proposed Causal Residual Mamba | feature construction, causal blocks, residual head, M0/M1 normalization | normalization formula | architecture diagram | calling implementation bidirectional | 0.9 |
| V. Experimental Setup | datasets, splits, training selection, metrics, baseline information classes | metric definitions | compact setup table | mixing screening/confirmation or validation/test | 1.0 |
| VI. Results and Discussion | Stage 4 transfer, Stage 6 primary comparison, Stage 7 exploratory statistics, Stage 8 baseline context, limitations integrated | paired changes/CIs and qualifications | primary result + temporal or paired figure; baseline table | universal superiority; treating CPU/GPU timings as fair | 1.7 |
| VII. Conclusion | concise supported outcome and next validation needs | none | none | future work framed as completed | 0.4 |

A separate limitations section likely consumes too much of six pages; integrate a compact limitations paragraph/subsection into Results and Discussion and provide full detail in supplement. Page allocation is a planning estimate, not a template guarantee.
