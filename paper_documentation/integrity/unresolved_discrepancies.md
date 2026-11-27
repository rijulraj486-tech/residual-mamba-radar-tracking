# Unresolved discrepancies and unavailable evidence

- **Stage 8 provenance pointer:** `evaluation_provenance.json` references Stage 6 Markdown report SHA-256 (`25120873…`) under `stage6_report_sha256`, while the Stage 6 JSON report SHA-256 is `6bac932e…`; separate Stage 8 `provenance.json` correctly points to JSON. Stage 9 finds no M0/M1 score disagreement. Historical artifacts remain unchanged; cite and disclose this pointer error.
- **T=100 distribution report:** no equivalent pooled quantile report found. `NOT VERIFIED FROM AVAILABLE PROJECT ARTIFACTS`.
- **Citation source:** no BibTeX/LaTeX manuscript found, so full citation metadata and citation-use matrix cannot be completed.
- **Training environment:** GPU is reported as RTX 3060 for evaluation and Stage 2 manifests record CUDA/PyTorch; exact CPU model and some actual package versions at training are absent.
- **Trajectory examples/predictions:** current Python environment has no PyTorch (`ModuleNotFoundError` on read-only dataset inspection). A Stage 2 test trajectory figure was therefore not generated. No saved per-trajectory model predictions were retained, preventing model-vs-truth trajectory visualization and trajectory bootstrap.
- **Exact historical commands:** scripts are present, but shell command history was not found; command reference labels entry points rather than asserting past invocations.
- **Checkpoint binary inventory:** this checkout lists 103 checkpoint-like files via file search count in Stage 9/Stage 8 records, but `checkpoint_inventory.csv` inventories current files; files omitted by `.gitignore` may not be present in this repository snapshot.
