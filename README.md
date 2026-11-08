# SANKOFA (Python): vetted-answer recommendation with per-language safety guarantees

Code for the paper *SANKOFA: Vetted-Answer Recommendation with Per-Language Safety Guarantees for Health
Questions in Low-Resource African Languages*. CPU only; no GPU, no pretrained models, no internet.
<img width="413" height="315" alt="Screenshot 2026-11-08 at 11 10 35" src="https://github.com/user-attachments/assets/f5432357-0114-478c-8367-765aa4b0ee0d" />

## Setup

```bash
pip install -r requirements.txt
mkdir data   # put Train.csv, Val.csv, Test.csv from the Kaggle dataset here
```

Dataset: *Multilingual Health Q&A Dataset (African Languages)* on Kaggle
(https://www.kaggle.com/datasets/mubarakasijillah/multilingual-health-q-and-a-dataset-african-languages).
Paths can be changed with `SANKOFA_DATA=/path/to/csvs` and `SANKOFA_RESULTS=/path/to/out`.

## Reproduce the paper (run in this order)

| Command | Produces | Paper |
| --- | --- | --- |
| `python experiments/exp_main2.py` | main results, ablations, latency, `sankofa_store.pkl` (needed below) | Tables 1, 2 |
| `python experiments/exp_recsys.py` | Recall@k, nDCG@10, cold-start, popularity bias, conformal lists | Table 3, Figure 2, Table 5 (lists) |
| `python experiments/exp_robust.py` | Akan / Amharic / Luganda keyboard and SMS perturbations | Table 4 |
| `python experiments/exp_abstain.py` (and `GRID=coarse python experiments/exp_abstain.py`) | Learn-then-Test, pooled vs per-language | Figure 3, Table 5 |
| `python experiments/judge_dump.py`, `exp_xl.py`, `exp_audit.py` | anchor bridge mining, cross-lingual view, translation-drift flags | Sections 3.3, 4.4 |
| `python predict_test.py data test_predictions.csv` | test-set answers + confidence + answer/escalate decision | released file |
| `python export_android_assets.py <dir>` | per-language model files for the Android app | app |

Full run time on 2 CPU cores: about 45 minutes. `exp_main2.py` alone takes about 17 minutes.

## Package layout

- `sankofa/normalize.py`: Ge'ez homophone folding + syllable decomposition, Akan open-vowel folding, scaffold mining and
  register factorisation
- `sankofa/retrieve.py`: five retrieval views (question→question char/word/BM25, question→answer char/word) + register view
- `sankofa/fusion.py`: self-tuned per-language fusion weights (train-only development slice)
- `sankofa/abstain.py`: confidence model and Learn-then-Test thresholds (Bonferroni over a fixed grid)
- `sankofa/bridge.py`, `sankofa/xl.py`: anchor-bootstrapped English ↔ local-language bridge
- `sankofa/metrics.py`: exact vetted-answer accuracy, chrF++, script-agnostic ROUGE-L

## Caveats

The mined-pair precision and the translation-drift examples were judged by a single reader and must be re-checked by
native-speaking clinicians. The "lenient" correctness criterion (chrF++ ≥ 50) is a proxy that clinicians must validate.
