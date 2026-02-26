import csv
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from model.model import CropCNNMultiTaskModel
from model.preprocessing import load_aligned_data, Standardizer, stratified_split, evaluate_predictions
from model.visualization import plot_loss, plot_predictions, plot_snp_heatmap, plot_manhattan_snp_importance
from model.explanation import compute_saliency_map, load_snp_positions, map_snp_indices_to_positions
from args import get_parser

# Sanya
phenotype_names = ["YPP", "PN", "GNPP", "SSR", "GWe", "HD", "PH", "PL", "GL", "GWi"]

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    num_tasks = len(phenotype_names)
    rice_genotypes, rice_phenotypes, rice_samples = load_aligned_data(args.dataset)

    train_idx, test_idx = stratified_split(rice_phenotypes, args.train_ratio)

    train_genotypes = rice_genotypes[train_idx]
    test_genotypes = rice_genotypes[test_idx]
    train_phenotypes = rice_phenotypes[train_idx]
    test_phenotypes = rice_phenotypes[test_idx]

    train_standardizer = Standardizer()
    train_phenotypes_norm = train_standardizer.fit_transform(train_phenotypes)
    test_phenotypes_norm = train_standardizer.transform(test_phenotypes)

    train_dataset_norm = TensorDataset(train_genotypes, train_phenotypes_norm)
    test_dataset_norm = TensorDataset(test_genotypes, test_phenotypes_norm)

    train_loader = DataLoader(train_dataset_norm, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_norm, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CropCNNMultiTaskModel(num_tasks=num_tasks).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr,
                                     step_size_up=args.step_size_up, mode='triangular')

    # Model training
    model.train()
    train_loss_history = []
    test_loss_history = []
    best_loss = float('inf')
    counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for batch_g, batch_p in train_loader:
            batch_g, batch_p = batch_g.to(device), batch_p.to(device)
            optimizer.zero_grad()
            preds_orig = model(batch_g)
            losses_orig = [criterion(pred, batch_p[:, i].unsqueeze(1)) for i, pred in enumerate(preds_orig)]
            batch_loss = sum(losses_orig) / len(losses_orig)

            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                print("Batch loss is NaN or Inf, skipping backward")
                continue

            batch_loss.backward()
            torch.nn.util.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += batch_loss.item()
            num_train_batches += 1

        avg_train_loss = total_train_loss / num_train_batches
        train_loss_history.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        num_test_batches = 0
        with torch.no_grad():
            for batch_g, batch_p in test_loader:
                batch_g, batch_p = batch_g.to(device), batch_p.to(device)
                preds = model(batch_g)
                losses = [criterion(pred, batch_p[:, i].unsqueeze(1)) for i, pred in enumerate(preds)]
                batch_loss = sum(losses) / len(losses)
                total_test_loss += batch_loss.item()
                num_test_batches += 1

        avg_test_loss = total_test_loss / num_test_batches
        test_loss_history.append(avg_test_loss)

        scheduler.step()

        print(
            f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, args.best_model_path)
            print(f"Saved best model with test loss: {best_loss:.4f}")
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break