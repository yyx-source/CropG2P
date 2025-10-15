import torch
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde, skew, kurtosis
from sklearn.model_selection import StratifiedKFold

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, phenotypes):
        self.mean = phenotypes.mean(dim=0, keepdim=True)
        self.std = phenotypes.std(dim=0, keepdim=True)
        self.std[self.std == 0] = 1.0
        return (phenotypes - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, preds):
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer not fitted yet!")
        return preds * self.std + self.mean


def load_aligned_data(crop_name="rice"):
    genotypes = torch.tensor(np.load(f"./input_re/{crop_name}_genotypes_SY.npy"), dtype=torch.float)
    phenotypes = torch.load(f"./input_re/{crop_name}_phenotypes_SY.pt", weights_only=False)
    with open(f"./input_re/{crop_name}_samples_SY.txt", 'r') as f:
        sample_names = f.read().splitlines()
    if torch.isnan(genotypes).any() or torch.isinf(genotypes).any():
        print(f"Warning: {crop_name} genotypes contain NaN or Inf values")
        genotypes = torch.nan_to_num(genotypes, nan=0.0)
    if torch.isnan(phenotypes).any() or torch.isinf(phenotypes).any():
        print(f"Warning: {crop_name} phenotypes contain NaN or Inf values")
        phenotypes = torch.nan_to_num(phenotypes, nan=0.0)
    print(
        f"Loaded {crop_name} - Genotypes shape: {genotypes.shape}, Phenotypes shape: {phenotypes.shape}, Samples: {len(sample_names)}")
    return genotypes, phenotypes, sample_names


def stratified_split(phenotypes, train_ratio=0.8, n_bins=10, use_first=True):
    if isinstance(phenotypes, torch.Tensor):
        phenotypes_np = phenotypes.cpu().numpy()
        phenotypes_df = pd.DataFrame(
            phenotypes_np,
            columns=[f'pheno_{i}' for i in range(phenotypes_np.shape[1])]
        )
    elif isinstance(phenotypes, pd.DataFrame):
        phenotypes_df = phenotypes.copy()

    if use_first:
        best_phenotype = phenotypes_df.columns[0]
    else:
        best_phenotype = None
        min_distance = float('inf')
        for col in phenotypes_df.columns:
            values = phenotypes_df[col].dropna().values
            if len(values) < 10:
                continue
            s = skew(values)
            k = kurtosis(values, fisher=False)
            distance = np.sqrt(s ** 2 + (k - 3) ** 2)
            if distance < min_distance:
                min_distance = distance
                best_phenotype = col

    stratification_values = phenotypes_df[best_phenotype].values
    bins = np.histogram_bin_edges(stratification_values, bins=n_bins)
    labels = np.digitize(stratification_values, bins[:-1])

    n_splits = int(1 / (1 - train_ratio))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(np.zeros(len(phenotypes_df)), labels))

    return train_idx, test_idx

def evaluate_predictions(preds, targets, phenotype_names, csv_path="./output_re/rice/SY/rice_gated_inception.csv"):
    metrics = {'r2': [], 'mae': [], 'rmse': [], 'pcc': []}
    for i in range(len(preds)):
        pred = preds[i].cpu().numpy().flatten()
        target = targets[:, i].cpu().numpy().flatten()
        r2 = r2_score(target, pred)
        mae = mean_absolute_error(target, pred)
        rmse = root_mean_squared_error(target, pred)
        corr, _ = pearsonr(target, pred)

        metrics['r2'].append(r2)
        metrics['mae'].append(mae)
        metrics['rmse'].append(rmse)
        metrics['pcc'].append(corr)

        print(f"rice {phenotype_names[i]}:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  PCC: {corr:.4f}")

    avg_r2 = np.mean(metrics['r2'])
    avg_mae = np.mean(metrics['mae'])
    avg_rmse = np.mean(metrics['rmse'])
    avg_corr = np.mean(metrics['pcc'])

    print(f"\nrice Average Metrics:")
    print(f"  Average R² Score: {avg_r2:.4f}")
    print(f"  Average MAE: {avg_mae:.4f}")
    print(f"  Average RMSE: {avg_rmse:.4f}")
    print(f"  Average PCC: {avg_corr:.4f}")

    df = pd.DataFrame(metrics)
    df.insert(0, 'Phenotype', phenotype_names)
    df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")
    return metrics, avg_r2