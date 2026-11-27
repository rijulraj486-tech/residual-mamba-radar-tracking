# Limitations and threats to validity

1. Evidence uses one synthetic motion/noise profile and one simulator family; no real-world radar dataset validation.
2. Long-sequence results use fixed-reference simulation. Increasing T accumulates motion and shifts state/range distributions; T=400 reaches physically extreme simulated speeds/ranges.
3. Measurement noise is fixed with range (30 m range SD and 1° bearing SD). The implied geometry and observation quality are not a realistic full radar channel model at multi-hundred-kilometre to million-metre range.
4. Only single-target state estimation is evaluated in this paper scope. No multi-target evaluation supports claims here.
5. Five training seeds provide limited power. Stage 7 t-tests are assumption-dependent exploratory evidence; exact Wilcoxon two-sided p-values are ≥0.0625 and none are <.05. No trajectory-level bootstrap was performed because per-trajectory error contributions/predictions were not retained. Seed CIs condition on fixed test sets and do not represent new-test-set uncertainty.
6. EKF/UKF know exact F and h; oracle variants also use simulator Q/R. Train-estimated-Q/R filters still know dynamics. These model-informed results must be separated from learned comparisons.
7. KalmanNet is trained at T=100 and transferred to T=200/400; it is not matched-length trained. Stage 8 lacks internal sanitization/clipping invocation counts, so finite outputs do not prove internal numerical stability.
8. Stage 8 evaluation provenance points at a Stage 6 Markdown hash in one field while a separate audit provenance points at the JSON hash; no score discrepancy was found, but the pointer defect must be disclosed.
9. No target-population uncertainty, hardware-independent runtime comparison, or multi-regime generalization is established.
