# Ordinal Information Processing Demo

This repository keeps only the reproduction path needed for the synthetic ordinal task:

1. Train the synthetic 4-sequence RNN.
2. Compute PCA explained variance of RNN activity.
3. Plot the learned 3D tree structure in PC space.
4. Verify transfer learning from the learned tree template.

## Scripts

```powershell
python .\train_synthetic.py
python .\analyze_pca_variance.py
python .\plot_tree_structure_3d.py
python .\transfer_learning.py
```

Outputs are written under `outputs/`:

- `outputs/models/synthetic_latest.pt`
- `outputs/figures/pca_explained_variance.png`
- `outputs/figures/tree_structure_3d.png`
- `outputs/figures/transfer_learning_accuracy_*.png`
- `outputs/csv/*.csv`

To change runtime parameters, edit the configuration constants at the top of each script.
For a quick smoke run, reduce values such as `EPOCHS`, `SAMPLES_PER_CLASS`, and
`TEST_SAMPLES_PER_CLASS`, then run the same commands:

```powershell
python .\train_synthetic.py
python .\analyze_pca_variance.py
python .\plot_tree_structure_3d.py
python .\transfer_learning.py
```
