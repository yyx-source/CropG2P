import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from model.model import CropCNNMultiTaskModel
from model.preprocessing import load_aligned_data, Standardizer, stratified_split, evaluate_predictions
from model.visualization import plot_loss, plot_predictions, plot_snp_heatmap, plot_manhattan_snp_importance
from model.explanation import compute_saliency_map, load_snp_positions, map_snp_indices_to_positions
from args import get_parser

# # Hangzhou
# phenotype_names = ["YPP", "PN", "GNPP", "SSR", "GWe", "HD", "PH", "FLL", "FLW", "PL", "GL", "GWi"]

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

    # model.train()
    # train_loss_history = []
    # test_loss_history = []
    # best_loss = float('inf')
    # counter = 0

    # for epoch in range(args.num_epochs):
    #     model.train()
    #     total_train_loss = 0
    #     num_train_batches = 0

    #     for batch_g, batch_p in train_loader:
    #         batch_g, batch_p = batch_g.to(device), batch_p.to(device)
    #         optimizer.zero_grad()
    #         preds_orig = model(batch_g)
    #         losses_orig = [criterion(pred, batch_p[:, i].unsqueeze(1)) for i, pred in enumerate(preds_orig)]
    #         batch_loss = sum(losses_orig) / len(losses_orig)

    #         if torch.isnan(batch_loss) or torch.isinf(batch_loss):
    #             print("Batch loss is NaN or Inf, skipping backward")
    #             continue

    #         batch_loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #         optimizer.step()

    #         total_train_loss += batch_loss.item()
    #         num_train_batches += 1

    #     avg_train_loss = total_train_loss / num_train_batches
    #     train_loss_history.append(avg_train_loss)

    #     model.eval()
    #     total_test_loss = 0
    #     num_test_batches = 0
    #     with torch.no_grad():
    #         for batch_g, batch_p in test_loader:
    #             batch_g, batch_p = batch_g.to(device), batch_p.to(device)
    #             preds = model(batch_g)
    #             losses = [criterion(pred, batch_p[:, i].unsqueeze(1)) for i, pred in enumerate(preds)]
    #             batch_loss = sum(losses) / len(losses)
    #             total_test_loss += batch_loss.item()
    #             num_test_batches += 1

    #     avg_test_loss = total_test_loss / num_test_batches
    #     test_loss_history.append(avg_test_loss)

    #     scheduler.step()

    #     print(
    #         f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    #     if avg_test_loss < best_loss:
    #         best_loss = avg_test_loss
    #         counter = 0
    #         torch.save({
    #             'epoch': epoch + 1,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': best_loss,
    #         }, args.best_model_path)
    #         print(f"Saved best model with test loss: {best_loss:.4f}")
    #     else:
    #         counter += 1
    #         if counter >= args.patience:
    #             print(f"Early stopping at epoch {epoch + 1}")
    #             break

    # plot_loss(train_loss_history, test_loss_history)

    model = CropCNNMultiTaskModel(num_tasks=num_tasks).to(device)
    checkpoint = torch.load(args.best_model_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds_all = [[] for _ in range(num_tasks)]
    test_targets_all = []
    with torch.no_grad():
        for batch_g, batch_p in test_loader:
            batch_g, batch_p = batch_g.to(device), batch_p.to(device)
            preds = model(batch_g)
            for i, pred in enumerate(preds):
                test_preds_all[i].append(pred.cpu())
            test_targets_all.append(batch_p.cpu())

    test_preds = [torch.cat(pred_list, dim=0).squeeze(-1) for pred_list in test_preds_all]
    test_targets = torch.cat(test_targets_all, dim=0)
    test_denorm_preds = train_standardizer.inverse_transform(torch.stack(test_preds, dim=1))
    test_denorm_targets = train_standardizer.inverse_transform(test_targets)

    test_denorm_preds_list = [test_denorm_preds[:, i] for i in range(num_tasks)]
    evaluate_predictions(test_denorm_preds_list, test_denorm_targets, phenotype_names)
    top_10_indices_per_phenotype = plot_predictions(test_denorm_preds_list, test_denorm_targets, phenotype_names)

    # Save top 10% overlap data to CSV
    combinations = []
    counts = []
    num_phenotypes = len(phenotype_names)
    for i in range(2 ** num_phenotypes):
        binary = format(i, f'0{num_phenotypes}b')
        current_combination = []
        for j, pheno_name in enumerate(phenotype_names):
            if binary[j] == '1':
                current_combination.append(pheno_name)
        combination_key = '-'.join(current_combination)
        combinations.append(combination_key)
        intersection_sets = [top_10_indices_per_phenotype[j] for j, bit in enumerate(binary) if bit == '1']
        if intersection_sets:
            condition = set(intersection_sets[0])
            for s in intersection_sets[1:]:
                condition = condition & set(s)
        else:
            condition = set()
        count = len(condition)
        counts.append(count)

    matrix_data = {'Combination': combinations,}
    for pheno_name in phenotype_names:
        matrix_data[pheno_name] = []
    for i, combination in enumerate(combinations):
        for pheno_name in phenotype_names:
            if pheno_name in combination:
                matrix_data[pheno_name].append(counts[i])
            else:
                matrix_data[pheno_name].append(0)

    with open(args.venn_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = phenotype_names
        writer.writerow(header)
        max_len = max(len(top_10_indices_per_phenotype[i]) for i in range(num_phenotypes))
        for row_idx in range(max_len):
            row_data = []
            for col_idx in range(num_phenotypes):
                if row_idx < len(top_10_indices_per_phenotype[col_idx]):
                    sample_name = f'sample_{top_10_indices_per_phenotype[col_idx][row_idx]}'
                    row_data.append(sample_name)
                else:
                    row_data.append('')
            writer.writerow(row_data)

    for i in range(num_phenotypes):
        for j in range(i + 1, num_phenotypes):
            overlap_count = len(set(top_10_indices_per_phenotype[i]) & set(top_10_indices_per_phenotype[j]))
            print(f"Overlap count of top 10% samples (pred vs true) between {phenotype_names[i]} and {phenotype_names[j]}: {overlap_count}")

    # Interpretability: Compute and plot saliency maps for each phenotype
    print("\nComputing SNP importance using Saliency Maps...")

    # Load SNP positions once
    snp_df = load_snp_positions(args.vcf_file_path)

    # Collect saliency scores for all SNPs across phenotypes
    saliency_dict = {}
    all_snp_data = []

    for i in range(num_tasks):
        # Compute saliency map
        saliency = compute_saliency_map(model, rice_genotypes, i, device)

        # If saliency is multi-sample, average across samples
        if saliency.ndim > 1:
            saliency = saliency.mean(axis=0)

        # Store saliency for this phenotype
        saliency_dict[phenotype_names[i]] = saliency

        # Create DataFrame with all SNPs for this phenotype
        pheno_df = snp_df.copy()
        pheno_df['Phenotype'] = phenotype_names[i]
        pheno_df['Saliency_Score'] = saliency
        all_snp_data.append(pheno_df)

    # Combine all SNP data into a single DataFrame
    combined_snp_df = pd.concat(all_snp_data, ignore_index=True)

    # Plot heatmap and Manhattan plot for unique chromosome positions
    plot_snp_heatmap(saliency_dict, snp_df)
    plot_manhattan_snp_importance(saliency_dict, snp_df)

    # Save all SNP data to CSV
    final_df = combined_snp_df[['SNP_Index', 'Chromosome', 'Position', 'Phenotype', 'Saliency_Score']]
    final_df.to_csv(args.output_csv_path, index=False)
    print(f"All SNP positions with phenotypes and saliency scores saved to {args.output_csv_path}")

    # Select top 200 SNPs for each phenotype
    top_200_per_phenotype = combined_snp_df.groupby('Phenotype').apply(
        lambda x: x.nlargest(200, 'Saliency_Score')).reset_index(drop=True)

    # Save top 200 SNP data to CSV
    final_df = top_200_per_phenotype[['SNP_Index', 'Chromosome', 'Position', 'Phenotype', 'Saliency_Score']]
    final_df.to_csv(args.output_csv_path_200, index=False)
    print(f"Top 200 SNP positions with phenotypes and saliency scores saved to {args.output_csv_path_200}")

    torch.cuda.empty_cache()