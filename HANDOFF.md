# S6E9 EV Purchase Prediction — GPU Handoff Notes

Kaggle competition `playground-series-s6e9` (binary classification, ROC-AUC). Deadline **2026-09-30**. Goal: push from current best **0.94638 public LB (rank ~523/2927)** toward top 10 (needs ~0.94664) using real GPU compute, since the dev Mac choked on CatBoost (2+ hours, never finished a depth-8 run).

## Do this first on the new machine

1. Install Python 3.11+ and CUDA-enabled drivers (RTX 5060).
2. `python -m venv .venv` then activate it, `pip install pandas numpy scikit-learn lightgbm xgboost catboost scipy pyarrow kaggle`.
   - For GPU CatBoost: `CatBoostClassifier(task_type='GPU', devices='0', ...)`.
   - For GPU LightGBM/XGBoost: install GPU-enabled builds if you want, but CPU is already fast enough for those on any machine — **CatBoost-on-GPU is the actual priority**, it's the one that was infeasible on CPU here.
3. Set up Kaggle credentials: `mkdir .kaggle && echo "<token>" > ~/.kaggle/access_token` (get a fresh token — the old one was exposed in a chat screenshot and should be rotated).
4. Re-download data (not committed to git): `kaggle competitions download -c playground-series-s6e9 -p data/ --unzip` (into `data/train.csv`, `data/test.csv`, `data/sample_submission.csv`).
5. Re-download the supporting datasets used for feature engineering:
   - `kaggle datasets download -d itzzomkar/ev-adoption-behavior-and-range-anxiety -p data/original --unzip`
   - `kaggle datasets download -d marcmaldonado/s6e9-generator-fingerprints -p data/fingerprints --unzip`
6. Run `python src/stage1_train.py` first to confirm the pipeline reproduces ~0.9455 OOF before doing anything new.

## Verified facts (checked against real code/data this session, not just trusted from secondhand research)

- Generator formula is genuine: `S = 1.2*(income/1e5) + 0.6*concern + 2*subsidy - 1*(anxiety==Medium) - 3*(anxiety==High)`, threshold ~5.5. Standalone AUC 0.9377 — confirmed by computing it directly.
- Practical ceiling for a single rigorous pipeline is ~0.9459–0.9465 (independently verified via two real GitHub postmortems: `happyc0der/kaggle-s6e9-ev-purchases`, `avoid137/kaggle-s6e9-postmortem`).
- **Important:** a top-scoring public notebook (`amanatar/s6e9-samart-sota-meta-blend...`) turned out to be doing **leaderboard probing** — its log explicitly shows "Applying Exact Public Split Calibration Swaps" and hardcoded per-row score adjustments tuned to the public test split. This does NOT generalize to the private leaderboard. It was removed from the blend. Be suspicious of any public notebook whose claimed score is way above the plausible cluster (e.g. anything claiming 0.98+ when the real leaderboard max is ~0.949) — that's leaky/buggy local CV or worse, not a real result.
- Tie-breaking/lexsort tricks (hyped in some public notebooks) do NOT apply — verified directly: test.csv has zero duplicate feature rows, and a rank-blended ensemble output has zero tied predictions.
- Explicit interaction features (income×subsidy, concern×subsidy, etc.) and high `max_bin` LightGBM gave **zero measurable improvement** (0.94556 vs 0.94553 baseline) — the fine-resolution target-encoded generator score already captures that structure. Don't re-try this.
- Don't build an elaborate custom neural architecture (FT-Transformer + TabM + mixture-of-experts etc.) — evaluated and rejected. Verified postmortem found plain MLPs underperform GBDT by -0.0045; NN diversity's real contribution here is ~1e-4 to 1e-3 (a decorrelated ensemble member, not a breakthrough). Already captured this cheaply via a verified public RealMLP notebook's OOF predictions rather than training one from scratch. If you want more NN diversity, grab another verified public notebook's OOF rather than building a custom architecture.

## Pipeline / what's built

- `src/features.py` — feature engineering: generator score + interactions (tested, didn't help) + OOF-safe nested target encoding of income/commute at multiple bin widths + categorical target encoding + target-encoded generator score at multiple roundings (this last one dominates feature importance — it's reconstructing the generator's noise/threshold function).
- `src/stage1_train.py` — main LightGBM, ~0.9455 OOF.
- `src/stage2_xgb_cat.py` — XGBoost + CatBoost (CatBoost part unusable on CPU, rewrite to use `task_type='GPU'` here).
- `src/stage2b_catboost.py` — faster CatBoost attempt (Plain boosting, depth 6) — still only got ~90s/fold on Mac CPU when run alone; use GPU here instead, can afford deeper trees / more iterations.
- `src/stage2c_shallow_lgb.py` — shallow/heavily-subsampled LightGBM diversity variant, OOF 0.94570 (best single model so far).
- `src/stage3_blend.py`, `src/stage4_final_blend.py` — rank-blending ensemble logic (linear weights optimized on OOF via Nelder-Mead; a nonlinear meta-model was deliberately avoided per the postmortem's warning about OOF/test aggregation mismatch).
- `data/public_kernels/` (not committed, re-download via `kaggle kernels output <ref> -p <path>`) — verified OOF predictions from legitimate public notebooks:
  - `najiama/xgboost-triple-te-dynamic-pruning-lb-0-94639` (OOF AUC 0.94614)
  - `aryankaisth/signal-that-matters-eda-histgbm-lb-0-94637` (OOF AUC 0.94603)
  - `najiama/pure-lgbm-model-cv-0-94607-lb-0-94638` (OOF AUC 0.94607)
  - `evgendvorkin/s6e9-single-xgb-cv-0-94583` (OOF AUC 0.94583)
  - `yekenot/ps-s6-e9-realmlp-pytorch` (OOF AUC 0.94601, NN diversity)
  - Always validate any newly-downloaded OOF file against real train labels before trusting it (see `stage4_final_blend.py`'s `add_csv`/`add_npy` pattern) — some public notebooks are gamed or buggy.

## Best submission so far

0.94638 public LB, from `src/stage3_blend.py`'s simplest 4-model blend (lgb + xgb + pub_najiama_xgb_te + pub_aryankaisth_hist). More models/re-optimized weights (6-model, 8-model variants) landed in a 0.94635–0.94638 band — within noise, no clear improvement. **A working GPU CatBoost is the most promising untried lever** since it was never actually validated in the ensemble (the Mac attempt was killed before completion).

## Next steps in priority order

1. Get CatBoost training properly on GPU (depth 8-10, more iterations, should take minutes not hours) — add its OOF to the blend.
2. Find 2-3 more verified public OOF-validated notebooks for ensemble diversity (only ~9 of ~30 available were checked).
3. Consider a KNN diversity model (not yet built) — postmortem evidence says even a weak KNN adds real ensemble value via decorrelation.
4. Re-run `stage4_final_blend.py` with everything, submit, check public LB score and rank via `kaggle competitions leaderboard -c playground-series-s6e9 --download`.
