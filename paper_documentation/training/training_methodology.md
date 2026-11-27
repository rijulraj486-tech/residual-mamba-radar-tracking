# Training methodology and checkpointing

Stage 5A is a single-seed (seed 101) feasibility screen: four T/variant runs (T=200/400 × M0/M1), validation only. It is not confirmation evidence. Stage 5B is the confirmation design: for each T=200 and 400, M0 and M1 are run at seeds 11,22,33,44,55. Paired runs share initial trainable parameter hashes, initialization/training/dropout seeds, and epoch-wise loader order hashes; M1 adds frozen training-only token statistics. Stage 5B code verifies configs inherit settings from frozen Stage 5A configs.

Confirmed setup from configs: Adam, learning rate 0.001, weight decay 0.0001, batch 16, max 180 epochs, validation early stopping patience 25, min_delta 1e-6, MSE over batch × six states × time, global gradient clip norm 1.0, dropout .08, no sequence truncation/chunking, checkpoint chosen by minimum validation six-state MSE. Run-specific seed streams and actual completion/best epochs are in `seed_and_run_registry.csv` and per-run summaries. No scheduler is declared in the run config/script. Stage 5B uses CUDA; test timing specifics are in Stage 6 report.

The historical T=100 M0/M1 pipeline and original five-seed results are separate and documented in the original experiment manifest/config/run metrics. Do not assume its training hyperparameters equal Stage 5B without the historical run config. Where those settings are absent, use `NOT VERIFIED FROM AVAILABLE PROJECT ARTIFACTS`.

## Validation/checkpoint selection

Validation trajectories select minimum validation six-state MSE. Early stopping triggers after configured patience without minimum-delta improvement. Test split is deliberately not opened during Stage 5B; Stage 5A applies preregistered guardrails (M1 position RMSE no more than 1% worse and velocity RMSE no more than 3% worse than same-length M0) for eligibility, while all checkpoint selection remains by validation six-state MSE. Stage 5B confirms across five seeds. Stage 6 performs final held-out scoring after checkpoint selection is frozen. Full per-run logs are retained in epoch_log.jsonl.
