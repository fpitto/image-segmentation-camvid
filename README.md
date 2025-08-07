# Semantic Segmentation Suite

## Overview

This repository provides a comprehensive suite of experiments and demos for semantic segmentation on urban street-view imagery (CamVid dataset), including classical encoder–decoder models, custom modules, inference benchmarks, visualization, and modern transformer-based architectures.

## Repository Structure

```
├── data/                             # CamVid dataset files
│   └── CamVid/                       # train, val, test images & annotations
├── CamVid.ipynb                      # Train & evaluate U-Net, FPN, PSPNet, DenseASPP pipelines
├── MaskFormer.ipynb                  # Baseline MaskFormer setup and training
├── MaskFormerDemo.ipynb              # Fine-tuning & evaluation of MaskFormer on extended dataset
├── InferenceTimeComparison.ipynb     # Inference time benchmarking across models/backbones
├── Plot.ipynb                        # Scripts for plotting metrics and comparison charts
├── My_DenseASPP.py                   # Custom DenseASPP module built on PSPNet–DenseNet121 backbone
├── requirements.txt                  # List of Python dependencies
└── README.md                         # This file
```

## File Descriptions

* **CamVid.ipynb**
  Implements object‐oriented data loading (`Dataset`, `Dataloder`), augmentations (Albumentations), and preprocessing. Benchmarks four segmentation architectures:

  * U-Net (DenseNet121, ResNet101)
  * FPN   (DenseNet121, ResNet101)
  * PSPNet (DenseNet121, ResNet101)
  * DenseASPP (custom module on PSPNet–DenseNet121)

  Reports loss curves, IoU, F1 metrics, and qualitative mask visualizations.

* **MaskFormer.ipynb**
  Provides a baseline MaskFormer implementation using a transformer-based architecture for panoptic/semantic segmentation. Covers model initialization, dataset hooks, and basic training loop.

* **MaskFormerDemo.ipynb**
  Fine-tunes MaskFormer on an extended CamVid subset with more complex scenes. Includes additional data pre‐ and post‐processing, advanced training schedules, and evaluation metrics.

* **InferenceTimeComparison.ipynb**
  Measures and compares inference latency and throughput of each segmentation pipeline (classical models and MaskFormer) across different backbone configurations. Useful for deployment considerations.

* **Plot.ipynb**
  Loads logged metrics and timing results to generate comparative plots (loss vs. epoch, IoU/F1 bar charts, inference time line plots). Leverages matplotlib for clear visual reports.

* **My\_DenseASPP.py**
  Defines a custom DenseASPP block and a `FullModel` that integrates it with a PSPNet–DenseNet121 encoder from `segmentation_models`. Offers `short` and `encoder_freeze` options for flexible experimentation.

* **requirements.txt**
  Pinpoints required packages, e.g.:

  ```text
  tensorflow>=2.10
  segmentation-models>=1.0.1
  albumentations>=1.0.0
  opencv-python
  matplotlib
  transformers         # for MaskFormer
  detectron2           # optional, if using Detectron2 implementation
  ```

## Prerequisites

* Python 3.8 or higher
* GPU-enabled environment for training deep networks
* `pip install -r requirements.txt`

## Usage

1. **Prepare data**

   * Clone or download the CamVid dataset into `data/CamVid/` with the following subfolders: `train`, `trainannot`, `val`, `valannot`, `test`, `testannot`.

2. **Run notebooks**

   * Open each `.ipynb` in Jupyter/Colab.
   * Adjust paths and hyperparameters in the first code cells if needed.
   * Execute cells sequentially to preprocess data, train models, benchmark inference, and visualize results.

3. **Run custom module**

   * Import `My_DenseASPP.py` into your Python script or notebook to instantiate the DenseASPP variant:

     ```python
     from My_DenseASPP import FullModel
     model = FullModel(num_classes=12, short=False, encoder_freeze=False)
     ```

## Evaluation & Results

* **Quantitative**: Each experiment logs loss, IoU, F1, and inference times for direct comparison.
* **Qualitative**: Visual side‐by‐side comparisons of predicted vs. ground‐truth masks.

## Extending the Project

* Swap in new backbones (EfficientNet, MobileNet, etc.) or encoders.
* Experiment with different loss functions or schedulers.
