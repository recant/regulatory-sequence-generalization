# Regulatory-sequence generalization stress test

A compact evaluation harness showing why sequence models should be checked under **group-held-out** splits, not only random row-level splits.

## Published motivation

`published_split_sensitivity.csv` records enhancer-classification results reported by ETNet under an ordinary split and an enhancer-level split designed to prevent enhancer overlap between train and test.

Reported AUROC changes include:

- GM12878: 0.875 → 0.826
- K562: 0.959 → 0.942
- MCF-7: 0.982 → 0.967

Source: https://academic.oup.com/bib/article/26/6/bbaf634/8361797

The point is methodological: regulatory-sequence benchmarks can move when related sequence groups cannot appear on both sides of the split.

## Runnable stress test

`stress_test.py` generates a transparent synthetic dataset of related sequence families and compares:

- random row-level cross-validation
- sequence-family-held-out cross-validation

The row-level split can exploit family identity. The family-held-out split asks whether the model generalizes to genuinely unseen families.

```bash
pip install -r requirements.txt
python stress_test.py
```

## How to adapt it

Replace the synthetic dataframe with a table containing sequence, target, and grouping metadata. The same pattern can evaluate sequence-cluster-held-out, chromosome-held-out, cell-type-held-out, motif-family-held-out, or low-similarity test sets.

## Limitation

This is a methodology demo. It is **not** an evaluation of Axis or any proprietary model, and it is not evidence that any particular published benchmark is inflated.
