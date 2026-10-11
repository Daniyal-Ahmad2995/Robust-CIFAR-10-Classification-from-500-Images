# Robust CIFAR-10 Classification from 500 Images (Section B)

This repository contains my solution to Section B of the assignment: training a small Inception-style CNN from scratch on only 500 CIFAR-10 images, and making it robust to a shifted test set (rotation, brightness and contrast changes, blur, noise, occlusion and one hidden corruption).

## Results

All numbers are the mean ± standard deviation over 3 seeds (0, 1, 2).

| Configuration | Val Acc | Clean Test | Shifted Test |
|---|---|---|---|
| No augmentation | 50.67 ± 3.09 | 41.94 ± 1.45 | 18.33 ± 0.75 |
| Geometric only (crop, flip, rotation) | 40.67 ± 0.94 | 37.32 ± 2.46 | 19.09 ± 0.33 |
| Photometric only (color jitter, blur, noise) | 50.33 ± 1.25 | 37.39 ± 1.29 | 18.20 ± 1.10 |
| Geometric + photometric | 43.33 ± 2.36 | 32.90 ± 2.18 | 19.46 ± 1.01 |
| + Cutout / Random Erasing | 39.00 ± 1.63 | 33.53 ± 0.02 | 20.11 ± 0.22 |
| + Mixup/CutMix + label smoothing | 36.67 ± 1.25 | 28.19 ± 1.90 | 18.73 ± 1.71 |
| **Final model** | 35.33 ± 2.05 | 30.15 ± 0.58 | **19.71 ± 0.24** |

- **Final shifted-test score:** 19.71 ± 0.24 % (baseline: 18.33 ± 0.75 %)
- **Single change that helped most:** adding random grayscale (+0.98 points on the shifted test)
- **Test-time augmentation (bonus):** 19.60 ± 0.25 %, so no improvement, at about 10× the inference cost

A one-page summary is in [`outputs/summary.docx`](outputs/summary.docx), and the full discussion of every part (A to F) is inside the notebook.

## Repository contents

```
.
├── SectionB_Robust_CIFAR10.ipynb   # fully executed notebook with all outputs and answers
├── requirements.txt                # Python packages
├── README.md
└── outputs/
    ├── summary.docx                # one-page Word summary
    ├── ablation_table.csv          # Part D table
    ├── clean_vs_shifted.png        # clean vs shifted test examples
    ├── augmentation_examples.png   # training images and their augmented versions
    ├── partA_learning_curves.png   # baseline loss and accuracy curves
    ├── partD_ablation.png          # ablation bar chart
    ├── partE_per_class.png         # per-class accuracy
    ├── partE_confusion_matrix.png  # normalised confusion matrix (shifted test)
    ├── partE_misclassified.png     # most confident mistakes
    └── partE_hidden_corruption.png # hidden corruption (channel swap) examples
```

## Model

`MiniInceptionNet` has 489,834 trainable parameters (limit: 1,000,000). It uses a small convolutional stem, four Inception modules with parallel 1×1, 3×3, 5×5 and pooling branches (with 1×1 reductions), global average pooling and a small classifier head with dropout. No pretrained weights are used.

## How to reproduce the results

### Option 1: Google Colab (recommended)

1. Open `SectionB_Robust_CIFAR10.ipynb` in Google Colab (File → Upload notebook, or open it directly from GitHub).
2. Go to **Runtime → Change runtime type** and select a **T4 GPU**.
3. Click **Runtime → Run all**.
4. The full run (hyper-parameter search, 7 configurations × 3 seeds × up to 60 epochs, and the analysis) takes roughly 30 to 50 minutes on a T4. CIFAR-10 is downloaded automatically.
5. When it finishes, all plots, `ablation_table.csv`, `final_summary.json` and `summary.docx` are in the `outputs/` folder. Download them from the Colab file panel on the left before the session ends.

### Option 2: Local machine

```bash
git clone <this-repository-url>
cd <repository-folder>
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install notebook
jupyter notebook SectionB_Robust_CIFAR10.ipynb
```

Then run all cells. A CUDA GPU is strongly recommended; on a CPU the run takes several hours.

### Notes on reproducibility

- **Fixed seeds:** the 500 training images and the 400 / 100 split use `DATA_SEED = 42`, the shifted test set uses `CORRUPT_SEED = 2024`, and the models are trained with seeds 0, 1 and 2. cuDNN is set to deterministic mode, so results should match closely, although small differences can appear across GPU types and library versions.
- **Shifted test set:** if the instructor's `shifted_test.npz` file is placed next to the notebook, it is used directly. Otherwise the notebook generates the shifted set once from the clean test set with the fixed seed.
- **Resuming:** finished runs are saved to `outputs/results.json` and `checkpoints/`. If the session stops, running all cells again skips the work that is already done.
- **Quick check:** setting the environment variable `SMOKE_TEST=1` runs the whole pipeline on fake data for 2 epochs, only to check that the code works.

## Rules followed

| Rule | How it is satisfied |
|---|---|
| No pretrained weights | Model built from scratch with random initialisation |
| At most 1M parameters | 489,834 parameters, checked with an `assert` |
| Inception-style module | Four Inception modules with 1×1, 3×3, 5×5 and pooling branches |
| No tuning on test data | Learning rate, weight decay, checkpoint epoch and analysis seed are chosen on the 100 validation images only |
| On-the-fly augmentation | Applied inside the `Dataset` and the training loop, never saved to disk |
| At most 60 epochs, 3 seeds | `EPOCHS = 60`, seeds 0, 1 and 2, reported as mean ± std |
