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
Our pipeline includes a standardized data preprocessing workflow using **PLINK** for quality control and custom Python scripts for format conversion and alignment.

(1) Data Preparation
   Ensure your input data conforms to the following formats:
   - **Genotypes**: A NumPy array (`.npy`) of shape `[n_samples, n_SNPs]` .
   - **Phenotypes**: A PyTorch tensor (`.pt`) of shape `[n_samples, n_traits]`.
   - **SNP Positions**: A VCF file (`.vcf`) containing the chromosomal positions of SNPs, used for mapping saliency scores to physical locations.

(2) Dependencies Preparation
Download the PLINK software (v1.9 or v2.0) suitable for your operating system.
The preprocessing script `data_preprocessing/*.py` is configured for Windows using 'plink.exe'.

(3) Automated Preprocessing Pipeline
Run the following scripts in **numerical order** (0 to 4) to process raw genotype/phenotype data into the required input format:
```bash
data_preprocessing/0_quality_control.py: PLINK Quality Control
data_preprocessing/1_convert_gene.py: Genotype Format Conversion
data_preprocessing/2_convert_pheno.py:Phenotype Format Conversion
data_preprocessing/3_filter_nan.py: Missing Value Imputation
data_preprocessing/4_alignment_*.py: Aligns genotypes and phenotypes by sample ID
```

(4) File Placement
Organize your data in the `input_re/` directory following the structure:
```bash
input_re/
├── <crop_name>_genotypes.npy
├── <crop_name>_phenotypes.pt
├── <crop_name>_samples.txt
└── <crop_name>_snps.vcf
```
### Input data
The model accepts the following input data:

Genotype data：`.npy`Format, Genome-wide SNP matrix.

Phenotype data: `.pt` Format, Multi-trait phenotypic measurements for the same samples.

VCF File: For SNP localization; required only if running saliency analysis.

### Model Training
Train the CropG2P.
You need to provide input files, and then
```bash
python main.py
```
After training, the following files will be saved to ./save_re/:
best_model_rice_SY_gated_inception.pt: Checkpoint of the best-performing model.

training_metrics.csv: Evaluation metrics (R², MAE, RMSE) on the test set.

### Output
All model outputs are organized in the ./output_re/ directory with clear subdirectories for each crop/trait:

(1)Comprehensive evaluation metrics for each phenotypic trait (R², MAE, RMSE, PCC).

(2)Scatter plots of predicted vs. actual phenotypic values for each trait 

(3)Saliency score arrays for each trait

## Demo Instructions
Due to data size limitations, the test dataset required to reproduce the rice_SY results reported in the manuscript cannot be hosted on GitHub. To run the demo and reproduce these results:

Access the demo testing dataset via Figshare: https://doi.org/10.6084/m9.figshare.31424108.  

Extract the dataset and place all downloaded files in the `./input_re/rice/` directory of the project

Execute the demo script to reproduce the rice_SY results as presented in the manuscript:

```bash
python run_demo.py
```





























