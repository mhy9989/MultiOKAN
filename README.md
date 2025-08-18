# Multiple Operator Kolmogorov-Arnold Network (MultiOKAN)

**Paper:** Encoding multiphase dynamics to predict spatiotemporal evolution via latent-space operators

Codes for training of Multiple Operator Kolmogorov-Arnold Network (**MultiOKAN**).

The present framework delivers end-to-end, data-driven prediction of multiphase flow evolution from initial volume-fraction fields—serving as a scalable surrogate to PDE solvers—by coupling a Kolmogorov-Arnold autoencoder with a latent-space operator, achieving low reconstruction error at comparable compute cost and remaining robust to noise and limited data.

<p align="center" width="100%">
<img src=".\fig\1.png" width="100%" />
</p>


## Code Structures
<details open>
<summary>Expand</summary>

- `core/` core training plugins and metrics.
- `methods/` contains training methods for various prediction methods.
- `Model/` contains the control files for various prediction methods and is used to store the model and training results.
- `models/` contains the main network architectures.
- `modules/` contains network modules and layers.
- `runfiles/` contains various startup scripts.
- `tool/` contains the font files `tool/font/`.
- `utils/` contains a variety of utilities for model training, model modules, plots, parsers, etc.
- `DataDefine.py` is used to get the flow field dataset and make a dataloader.
- `modelbuild.py` is used to build and initialize the model.
- `modeltrain.py` is used to train, validate, test and inference about model.
- `main_AE.py` is the main program for running autoencoder model training and validation.
- `main_DON.py` is the main program for running the latent operator model.
- `main.py` is the main program used to complete the entire training process.
- `inference.py` is used for model inference.
- `test.py` is used for model test.
- `predata.py` is used to write data information to the settings file.
- `requirements.txt` lists the project’s Python dependencies.

</details>

## Installation

```
# (1) Create env
conda create -n multiokan python=3.8 -y
conda activate multiokan

# (2) Install dependencies
pip install -r requirements.txt

# (3) (Optional) Editable install
pip install -e .
```

## Training & Inference
**Before starting any run, edit the control file at** `Model/project/checkpoints/settings.json`. This file drives the whole pipeline (data paths, model hyper-parameters, training schedule, logging/checkpoints). The training scripts read it at startup.

**For training**

```
# Autoencoder 
python/deepspeed main_AE.py

# Latent Operator
python/deepspeed main_DON.py

# AE + DON
python/deepspeed main.py
```

**For inference**
**Comment out or remove** the training calls (`model.train()` / `modelDON.train()`) in `main_AE.py` and `main_DON.py`, then run the scripts **with the same commands as training**.

## Data Format

- **Flow fields (`.npy`)**
   A 5-D NumPy array with shape
   `(**num**, **data_time**, **features/phases**, **height**, **width**)`.
  - `data_time`: time-ordered sequence from `t0` to `t_end` (index `0` is the initial field).
  - **Path**: set in `settings.json` → `data_config.org_path`.
- **Mesh (`.npy`)**
   A container holding two 2-D arrays (3-D NumPy):
   `[ y_mesh[H, W], x_mesh[H, W] ]` (same spatial resolution as the flow fields).
  - **Path**: set in `settings.json` → `data_config.mesh_path`.

> Ensure the mesh and flow fields share the same coordinate system, units, and `(H, W)` resolution.
