# Experiment command reference

Scripts are verified from source; actual shell command histories are not committed, so commands below are **entry-point references**, not claims of exact invocation. Do not rerun these against frozen output directories.

- Stage 2 generator: `python experiments/paper_v4_long_sequence/generate_stage2.py` (inspect script for flags/defaults before reproduction).
- Stage 3 preflight: `python experiments/paper_v4_long_sequence/stage3_frozen_preflight.py`.
- Stage 4 evaluation: `python experiments/paper_v4_long_sequence/stage4_frozen_test_evaluation.py`.
- Stage 5A screen: `python experiments/paper_v4_long_sequence/stage5a_matched_length_screen.py`.
- Stage 5B confirmation: `python experiments/paper_v4_long_sequence/stage5b_matched_length_confirmation.py`.
- Stage 6 held-out evaluation: `python experiments/paper_v4_long_sequence/stage6_matched_length_test_evaluation.py`.
- Stage 7 statistics: `python experiments/paper_v4_long_sequence/stage7_statistical_analysis.py`.
- Stage 8 baselines: `python experiments/paper_v4_long_sequence/stage8_long_sequence_evaluation.py`.

These references are derived from tracked entry-point files; exact flags, command lines, and historical invocation records are `NOT VERIFIED FROM AVAILABLE PROJECT ARTIFACTS`. Reproduction should use separate output roots and preserve frozen artifacts.
