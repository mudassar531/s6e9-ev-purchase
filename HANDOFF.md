# S6E9 EV Purchase Prediction — Handoff (GPU session)

Kaggle competition `playground-series-s6e9` (binary classification, ROC-AUC). Deadline **2026-09-30**. Goal: push toward top 10 (currently needs ~0.94664 public LB; current best is 0.94638). This machine is being used because the dev Mac hit repeated, unexplained severe slowdowns under sustained CPU load (probable thermal throttling) — CatBoost never finished even a single depth-8 run there after 2+ hours.

## Setup

1. Python 3.11+, CUDA drivers for the RTX 5060.
2. `python -m venv .venv`, activate, `pip install pandas numpy scikit-learn lightgbm xgboost catboost scipy pyarrow kaggle`.
3. Kaggle credentials: get a **fresh API token** from kaggle.com account settings (the old one was exposed in a chat screenshot earlier and should not be reused) → `mkdir -p ~/.kaggle && echo "<token>" > ~/.kaggle/access_token`.
4. `kaggle competitions download -c playground-series-s6e9 -p data/ --unzip`
5. `kaggle datasets download -d itzzomkar/ev-adoption-behavior-and-range-anxiety -p data/original --unzip`
6. `kaggle datasets download -d marcmaldonado/s6e9-generator-fingerprints -p data/fingerprints --unzip`
7. Re-download the public notebook OOF predictions listed below via `kaggle kernels output <ref> -p data/public_kernels/<slug>`.
8. Sanity check: `python src/stage5_clean_reference.py` should reproduce clean OOF ≈ 0.94543 (lgb) / 0.94560 (shallow) / 0.94541 (xgb). This also creates `models/folds.npy`, needed by later scripts.

## Verified facts (checked against real code/data, not trusted from secondhand research)

- Generator formula is genuine: `S = 1.2*(income/1e5) + 0.6*concern + 2*subsidy - 1*(anxiety==Medium) - 3*(anxiety==High)`, threshold ~5.5. Standalone AUC 0.9377 — confirmed by direct computation.
- Practical ceiling for a rigorous single pipeline is ~0.9459–0.9465 (independently verified via GitHub postmortems `happyc0der/kaggle-s6e9-ev-purchases`, `avoid137/kaggle-s6e9-postmortem`).
- A top-scoring public notebook (`amanatar/s6e9-samart-sota-meta-blend...`) does **leaderboard probing** — its log shows "Applying Exact Public Split Calibration Swaps" and hardcoded per-row score offsets tuned to the public test split. Removed from the ensemble permanently; never re-add it. Distrust any notebook claiming scores far above the plausible cluster (e.g. "0.98+" when the real leaderboard max is ~0.949).
- Tie-breaking/lexsort tricks (hyped in some public notebooks) do NOT apply here — verified directly: test.csv has zero duplicate feature rows, rank-blended output has zero tied predictions.
- Explicit interaction features and high `max_bin` LightGBM gave **zero measurable improvement** — the target-encoded generator score already captures that structure. Don't re-try.
- A **target-encoding leakage bug** was found and fixed (see below) — real but small (~0.0001 AUC inflation).
- **Meta-CV blend validation result (important):** greedy-forward blend selection, validated with proper held-out meta-folds (not just in-sample OOF), gives held-out AUC that matches in-sample almost exactly (0.94628 both ways) — the blend weights are trustworthy, not overfit. But it **always assigns zero weight to all three local models** (lgb_clean, shallow_clean, xgb_clean) in favor of the 5 verified public models. This means further local LightGBM/XGBoost variants are low-value — they're too redundant with the existing public pool. **CatBoost and KNN are the priority precisely because they're structurally different enough to potentially avoid this fate.**
- Don't build elaborate custom neural architectures (FT-Transformer/TabM/MoE) without evidence — see decision gates below.

## The leakage fix (important context for trusting numbers)

The original `features.py`'s `kfold_target_encode()` built target encodings using its own 5-fold split across the *whole* training set before the outer model CV — even with matching seeds, outer-training rows' TE features ended up computed from folds that included the outer-validation fold's labels. Fixed in `src/nested_features.py`: for each outer fold, outer-train features use an *inner* 5-fold cross-fit (touches only outer-train labels); outer-valid and test features are fit on the *full* outer-train only. Verified leak-free with automated invariance tests in `src/test_nested_features.py` (perturbing outer-valid labels leaves target-derived features bit-identical; perturbing outer-train labels changes them as expected).

**This did not change the actual 0.94638 Kaggle score** (that came from real held-out Kaggle labels, unaffected by internal CV bugs) — it only means our internal OOF numbers are now trustworthy to ~1e-4, which matters at this leaderboard density.

## Pipeline files

- `src/nested_features.py` — the correct, leak-free feature builder. Use this, not the old `src/features.py` (kept only for historical reference / the original stage0-4 scripts).
- `src/test_nested_features.py` — run this after any change to the feature pipeline; must pass before trusting new OOF numbers.
- `src/stage5_clean_reference.py` — clean OOF for lgb/shallow-lgb/xgb using nested features. This is the current trustworthy baseline.
- `src/stage6_meta_cv_blend.py` — meta-CV blend weight validation (greedy forward selection + equal weight baselines; a Dirichlet-search cross-check exists behind `WITH_DIRICHLET=1` env var but is slow and was skipped after equal/greedy already gave a clear, stable picture — don't bother with it unless you have spare cycles).
- `src/stage7_bagged_oof_check.py` — P2 sanity check (see below), **not yet run** — this is queued for this machine.
- `src/stage2_xgb_cat.py`, `src/stage2b_catboost.py` — old CPU CatBoost attempts, both effectively failed (2+ hours, no usable result on the dev Mac). Use these as a *reference for the feature/param setup*, not as-is — rewrite for GPU (see P3 below).
- `src/stage3_blend.py`, `src/stage4_final_blend.py` — older blend scripts (pre-leak-fix, pre-meta-CV). Historical value only now that `stage6_meta_cv_blend.py` exists; `reports/historic_stage3_weights.json` documents exactly what produced the 0.94638 submission for the record.
- `data/public_kernels/` (not committed to git — re-download) — verified OOF predictions, all checked against real train labels before trusting:
  - `najiama/xgboost-triple-te-dynamic-pruning-lb-0-94639` (OOF 0.94614)
  - `aryankaisth/signal-that-matters-eda-histgbm-lb-0-94637` (OOF 0.94603)
  - `najiama/pure-lgbm-model-cv-0-94607-lb-0-94638` (OOF 0.94607)
  - `evgendvorkin/s6e9-single-xgb-cv-0-94583` (OOF 0.94583)
  - `yekenot/ps-s6-e9-realmlp-pytorch` (OOF 0.94601, NN diversity — see the RealMLP decision gate below)
  - Always validate any newly-downloaded OOF file against real train labels before trusting it.

## Current best

0.94638 public LB (historic, preserved at `submissions/historic_stage3_blend_0.94638.csv`). The properly meta-CV-validated greedy blend (5 public models, local models excluded) gives a held-out estimate of 0.94628 — slightly lower but far more trustworthy than the ad-hoc fit that produced 0.94638. Full-data greedy weights (probability scale): `najiama_xgb_te 0.333, yekenot_realmlp 0.333, aryankaisth_hist 0.167, evgendvorkin_xgb 0.167`.

## THE PLAN — trimmed, agreed, do not expand scope

An earlier round of "unlimited compute" research proposals (full MoE + TabM + FT-Transformer + PairRank + an 18-step research program with 3×5×2 bagged OOF, 80-config CatBoost grids, 5-optimizer blend bake-offs, hard-pair analysis) was evaluated and explicitly rejected as disproportionate to the ~3-4 productive days actually remaining. The operating principle now is:

**maximize expected private-LB AUC improvement per unit of human iteration time.**

Do the following in order. Respect the stopping rules — if 2-3 reasonable variants of something show no measurable ensemble improvement, stop and move on; don't brute-force because compute exists.

### P0 — DONE. Clean validation foundation frozen (leak fix + tests + clean OOF, see above).

### P1 — DONE. Meta-CV blend validation (see results above). Held-out ≈ in-sample, local models excluded.

### P2 — NOT YET RUN. Small bagged-OOF sanity check (run this machine, ~5-10 min expected)
Run `python src/stage7_bagged_oof_check.py`. It builds a second, independent 5-fold partition (seed 123) for the strongest local model (shallow LGB) and checks: does repeat-to-repeat OOF/test correlation stay high (>0.98ish) with similar AUC? If yes: stop, the single-partition OOF is trustworthy, move on. If no: investigate further before trusting anything built on top of it.

### P3 — GPU CatBoost (the main reason for this machine)
Focused search, NOT an 80-config sweep — approximately 6-8 configurations:
- depth: 6, 8, 10
- learning_rate: ~0.03, 0.05, 0.07
- l2_leaf_reg: 3, 10
- Use intelligent combinations, not full Cartesian product.
- Compare raw/categorical input representation vs the engineered (target-encoded) representation — CatBoost may prefer raw categoricals.
- `task_type='GPU', devices='0'`, early stopping, up to ~4000-5000 iterations.
- **Must use the nested_features.py pipeline** (or raw features) for proper per-fold OOF — do not reintroduce the old leaky global-TE approach.
- For each config report: OOF AUC, fold AUCs, training time, Spearman correlation vs the current champion blend, and — most importantly — **the blend delta when added to the champion ensemble**, not just standalone AUC. A 0.9450 CatBoost that adds +0.00015 to the ensemble beats a 0.9457 CatBoost that's redundant.
- Once one configuration clearly dominates, stop searching — don't keep going through remaining configs for completeness.

### P4 — KNN diversity (one small branch)
Standardized representation, try k = 64, 128, 256 (or a similarly small trio), distance-weighted. Don't optimize KNN standalone AUC — measure Spearman vs champion and incremental blend AUC only. If all variants add ≤ noise-level improvement, stop KNN research entirely.

### P5 — Rebuild the ensemble and produce final candidates
Combine everything with valid OOF (clean local models if they earn a place, public models, CatBoost, KNN) through `stage6_meta_cv_blend.py`-style meta-CV validation. Explicitly exclude the amanatar leaderboard-probed model and anything else lacking real OOF backing. Produce 2-3 genuinely different, defensible submission candidates (e.g. `submission_robust_rank.csv`, `submission_robust_prob.csv`, `submission_diversity_heavy.csv`) rather than many near-identical ones.

### Neural / MoE decision gates (do not build speculatively)
- First check whether the existing verified RealMLP OOF prediction adds genuine held-out meta-CV value to the *new* clean ensemble. If it contributes essentially nothing, skip local TabM/FT-Transformer/MoE entirely — insufficient evidence it's worth the remaining time.
- If RealMLP does show real value, run exactly ONE local neural experiment (TabM), one serious config, not a research project. Only continue to more seeds/configs if that first one also adds independent signal.
- MoE: only consider it if, after CatBoost/KNN/neural work, there are ≥4 model families with meaningfully different OOF rankings AND clear evidence that different experts win in different feature-space regions (check conditional performance across generator-score bins, income quantiles, anxiety/subsidy groups). If every model performs similarly everywhere, do not build MoE — a router can't manufacture specialization that doesn't exist.
- PairRank/hard-pair analysis: skip unless the robust blend has clearly plateaued, real time remains, hard errors show identifiable structure, and implementation is cheap. Otherwise don't touch it before the deadline.

### Leaderboard discipline
Public LB is a sanity signal, not an optimizer — don't repeatedly retune weights chasing small LB deltas (differences under ~0.0003 are likely noise given the row count). Keep a submission log (file, meta-CV AUC, public LB, models used, weights, hypothesis) for each real submission. We care about private-LB robustness, not public-LB chasing.

### Time allocation (rough guide)
20% meta-CV/bagged-OOF sanity (mostly done), 35% GPU CatBoost, 15% KNN, 20% final ensemble + submissions, 10% optional neural/MoE only if the gates above justify it.

## Immediate next action for this session

Start with **P2** (`python src/stage7_bagged_oof_check.py`), then move to **P3** (GPU CatBoost — the actual reason this moved to a GPU machine). Report back (or relay through the user) the CatBoost OOF results and its blend-delta before doing anything further.
