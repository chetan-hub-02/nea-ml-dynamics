# nea-ml-dynamics

Machine learning framework for near-Earth asteroid (NEA) dynamical classification using short-duration backward integrations.

---

## Overview

This repository contains the code used to train and evaluate sequence-based neural networks for classifying near-Earth asteroids as dynamically **ejected** or **non-ejected**, based on short segments of their orbital evolution.

The models operate on time series of orbital elements derived from backward N-body integrations.

---

## Dataset

The dataset used in this work is publicly available on Zenodo:

DOI: https://doi.org/10.5281/zenodo.19234295

### Description

This dataset contains time-series inputs used to train and evaluate machine-learning models for NEA classification.

- The full dataset is provided.
- Train, validation, and test splits are defined in the code (including the random seed).
- The dataset is organized into two directories:
  - `class_0`: ejected  
  - `class_1`: non-ejected  

### Input format

- Each sample is a 2000-step time series of orbital elements.
- For training, sequences are downsampled (every 10th step) to obtain 200-step inputs.
- Features used: semi-major axis (`a`), eccentricity (`e`), or both.
- The time series corresponds to the first 0.2 Myr of backward integration.

### Labels

- Labels are assigned using the full 1 Myr backward integration.
- Objects are classified based on whether they are ejected or remain dynamically bound.

### Preprocessing

- All sequences are min-max normalized to [0, 1].
- Sequences shorter than 200 steps are zero-padded.

### Integration details

- Orbital evolution was computed using the Mercury N-body integrator.
- The dataset was balanced using perturbed realizations of initial conditions.

---

## Repository Contents

- `nea_classifier.py` — training and hyperparameter search script  
- `nea_classifier.ipynb` — notebook version of the same pipeline  
- `best_weights.h5` — pretrained model weights  
- `requirements.txt` — Python dependencies  

```markdown
## Installation

```bash
pip install -r requirements.txt

## Usage

### Prepare dataset

Download and extract the dataset, then update paths in the script:

```python
class_0_path = "/path/to/nea_dataset_v1/class_0"
class_1_path = "/path/to/nea_dataset_v1/class_1"
