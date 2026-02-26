# CropG2P
CropG2P: Leveraging Complete Genome-Wide Variations and Innerrelationships Improves Crop Genomic Prediction

## Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions for use](#Instructions-for-use)
- [Demo Instructions](#demo-Instructions)

## Overview

This repository contains the Python code for the CropG2P project. CropG2P is a multi-task machine learning-based GS model that improves predictive accuracy by leveraging genome-wide variations and correlations in agronomic traits. 


## System Requirements
### Hardware Requirements
- Standard desktop/laptop with a minimum of 16 GB RAM
- NVIDIA GPU with ≥ 8 GB VRAM is recommended for accelerated training

### Software Requirements
This package has been tested on the following systems:
- Operating System: Windows 10/11

### Python packages dependencies
This package- Python Version: 3.10
```bash
torch==2.5.1+cu118
numpy==1.26.4
pandas==2.3.3
scikit-learn==1.6.1
scipy==1.15.1
seaborn==0.13.2
matplotlib==3.6.3
```

## Installation Guide

### Install from Github
Install method by using git clone
```bash
git clone https://github.com/yyx-source/CropG2P
```
### Setting up the development environment
```bash
pip install -r requirements.txt
```

## Instructions for use
This is a guide for using CropG2P to perform genomic prediction and identify important SNPs.
### Data Preprocessing
(1) Data Format Preparation
   Ensure your input data conforms to the following formats:
   - **Genotypes**: A NumPy array (`.npy`) of shape `[n_samples, n_SNPs]` (float32). Missing values (NaN/Inf) will be automatically replaced with `0.0` by the code.
   - **Phenotypes**: A PyTorch tensor (`.pt`) of shape `[n_samples, n_traits]` (float32).
   - **Sample Names**: A text file (`.txt`) with one sample name per line, corresponding to the rows in the genotype/phenotype matrices.
   - **SNP Positions**: A VCF file (`.vcf`) containing the chromosomal positions of SNPs, used for mapping saliency scores to physical locations.

(2) File Placement
   Organize your data in the `input_re/` directory following the structure:
```bash
input_re/├── <your_crop_name>_genotypes_SY.npy├── <your_crop_name>_phenotypes_SY.pt├── <your_crop_name>_samples_SY.txt└── <your_crop_name>_snps.vcf


```

### Input data
Genotype data; Phenotype data

### Model Training


### Output


## Demo Instructions
### Run testing code
To run the demo, execute:
```bash
python main.py --demo
```


















