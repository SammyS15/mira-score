# MIRA: A Score for Conditional Distribution Accuracy and Model Comparison

This repository contains the MIRA score implementation and supplementary materials for ICML 2026.

## Overview

MIRA is a metric for evaluating conditional distribution accuracy and comparing models. This package provides tools to compute MIRA scores and includes a demonstration notebook showing baseline comparisons.

## Installation Instructions

Follow these steps to set up the environment and run the examples:

### 1. Create and Activate a Conda Environment

```bash
conda create -n mira_env python=3.8
conda activate mira_env
```

### 2. Install the Package

```bash
pip install mira_score
```

Or, to install from source in editable mode:

```bash
git clone https://github.com/<your-username>/mira-score.git
cd mira-score
pip install -e .
```

This will install the `mira_score` package along with all required dependencies (tqdm, torch, scipy, numpy).

### 4. Verify Installation

Test that the package is correctly installed:

```bash
python3 -c "from mira_score import mira, mira_bootstrap, get_device; print('MIRA successfully installed!')"
```
