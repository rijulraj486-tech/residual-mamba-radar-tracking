# Final manuscript preparation checklist

## Authors and front matter

- [ ] Confirm author names, author order, affiliations, corresponding author, and contact details.
- [ ] Confirm IEEE conference template, page limit, keywords, funding, and acknowledgments requirements.

## Technical content and evidence

- [ ] Supply and verify the original bibliography; check every citation and map it to the claim it supports.
- [ ] Freeze the final state, transition, process-noise, radar-observation, measurement-noise, and initialization equations against the implementation.
- [ ] Select the architecture diagram and verify the caption reflects the forward-only implementation and M1 normalization location.
- [ ] Prepare a compact dataset and training table covering T=100/T=200/T=400, splits, seeds, fixed-reference assumptions, and Stage 5A versus Stage 5B.
- [ ] Include Stage 4 frozen T=100 extrapolation results and label them as cross-length transfer.
- [ ] Include Stage 6 final held-out matched-length test results, keeping five seeds and paired comparisons intact.
- [ ] Include Stage 7 statistical results with n=5 limitations, exploratory t-test language, exact Wilcoxon values, and no trajectory-bootstrap claim.
- [ ] Present Stage 8 baselines by information access: matched-length M0/M1, T=100-trained KalmanNet transfer, and model-informed EKF/UKF; identify oracle Q/R privileges.
- [ ] Add the limitations paragraph: synthetic fixed profile, long-horizon distribution shift/extreme simulated speeds and ranges, fixed measurement noise, five seeds, no real radar validation, single target, and no multi-target evaluation.
- [ ] Disclose the Stage 8 provenance pointer discrepancy exactly as recorded; do not edit historical provenance files.
- [ ] Avoid comparative Stage 8 runtime claims; do not equate CPU and CUDA timing values.

## Publication assembly and final checks

- [ ] Assign final figure and table numbers in order of first citation; update all in-text cross-references.
- [ ] Use one caption per figure, confirm units and source-data paths, and do not imply the unavailable trajectory or runtime figures exist.
- [ ] Ensure table values match authoritative CSV/JSON sources and separate validation from final test results.
- [ ] Compile the complete IEEE manuscript and check figure legibility, table width, references, equations, and page count.
- [ ] Archive the manuscript source, verified bibliography, final figures/tables, and reproducibility/provenance notes together.
