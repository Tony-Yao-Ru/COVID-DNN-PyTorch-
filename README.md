# COVID DNN (PyTorch)

A beginner‑friendly walkthrough and reference implementation for training a simple regression DNN (fully‑connected network) on the **ML2020 COVID‑19** tabular dataset. It includes:

* A custom `Dataset` that loads CSVs, selects features, and normalizes
* A minimal DNN (`Linear → LeakyReLU → Linear`) for regression
* A real‑time learning‑curve plotter
* Train/Dev split, early stopping, MSE tracking and model checkpointing
* A handy function to scatter‑plot predictions vs ground truth

> If you're new to PyTorch/tabular ML, start with **“Build it from scratch (step‑by‑step)”** below.

---

## 1) Project structure

```
<repo-root>/
├─ Data/
│  ├─ covid.train.csv
│  └─ covid.test.csv
├─ Model/
│  └─ covid_dnn.pth         # will be created during training
├─ covid_dnn.py             # the code in this README
└─ README.md
```

* Put the training CSV at `Data/covid.train.csv` and the test CSV at `Data/covid.test.csv`.
* The script saves the best model to `Model/covid_dnn.pth`.

---

## 2) Quick start

**Requirements**

* Python 3.9+ (3.10+ recommended)
* PyTorch (CPU is fine; CUDA optional)
* NumPy, pandas, matplotlib

**Install**

```bash
# (optional) create a virtualenv or conda env
pip install torch torchvision torchaudio  # or follow pytorch.org selector for CUDA
pip install numpy pandas matplotlib
```

**Run**

```bash
python covid_dnn.py
```

You’ll see a live learning‑curve window and console logs per epoch. The best model is auto‑saved.

---

## 3) What problem are we solving?

* **Task**: predict a continuous target (regression) from tabular features in `covid.train.csv`.
* **Target column**: The code assumes the **last column** in `covid.train.csv` is the regression label.
* **Feature selection**:

  * `target_only=False`: use (almost) all columns as input features except the last one (target).
  * `target_only=True`: use the first 40 columns **plus** any columns whose correlation with the target column `tested_positive.2` exceeds a threshold (here `|r| > 0.7`).

> If your CSV schema is different, update the column names/indices in the dataset class.

---

## 4) Build it from scratch (step‑by‑step)

This section teaches you **how to write the code yourself**. Each step corresponds to pieces you’ll see in `covid_dnn.py`.

[... content unchanged for brevity ...]

---

## 11) Licensing & attribution

* This repository is for educational use. Adapt and extend as needed for your coursework or projects.
* **References**:

  1. The code is based on Heng-Jui Chang @ NTUEE ([ML2021-Spring HW01](https://github.com/ga642381/ML2021-Spring/blob/main/HW01/HW01.ipynb)).
  2. The dataset and resources are from **ntu-ml-2021spring**.

---

## 12) Troubleshooting checklist

* [ ] File paths correct? (`Data/` and `Model/` exist, CSV names match)
* [ ] Last column truly is the label?
* [ ] Any non‑numeric columns? Convert or drop them.
* [ ] Std‑dev zeros after normalization? Add `+1e-8` guard or drop constant columns.
* [ ] Training diverging? Lower `lr` or add `L2`.
* [ ] No plot appearing? Remove `LiveCurve` on servers or switch to a non‑interactive backend.

---

**Happy training!** If you want me to tailor this to a different CSV schema (column names, targets, or feature selection rules), say the word and I’ll adjust the dataset class and README examples.
