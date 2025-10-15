import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

phenotype_names = ["YPP", "PN", "GNPP", "SSR", "GWe", "HD", "PH", "PL", "GL", "GWi"]

def plot_loss(train_loss_history, test_loss_history, save_path="./output_re/rice/SY/loss_plot_rice_gated_inception.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-', label='Training Loss')
    plt.plot(range(1, len(test_loss_history) + 1), test_loss_history, 'r-', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Rice Training and Test Loss Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_predictions(preds, targets, phenotype_names, save_path_prefix="./output_re/rice/SY/rice_"):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    top_10_indices_per_phenotype = []
    for i, pheno_name in enumerate(phenotype_names):
        true_values = targets[:, i].cpu().numpy()
        pred_values = np.array([p.item() for p in preds[i]])

        r2_score_val = r2_score(true_values, pred_values)

        slope, _ = np.polyfit(true_values, pred_values, 1)
        r_val = np.sqrt(r2_score_val) * np.sign(slope)

        bins = 20
        true_hist, _ = np.histogram(true_values, bins=bins, density=True)
        pred_hist, _ = np.histogram(pred_values, bins=bins, density=True)
        jsd = jensenshannon(true_hist, pred_hist)
        slope, intercept = np.polyfit(true_values, pred_values, 1)
        fit_equation = f"y = {slope:.3f}x + {intercept:.3f}"

        plt.figure(figsize=(10, 8))
        kernel = gaussian_kde([true_values, pred_values])
        z = kernel([true_values, pred_values])
        norm = plt.Normalize(z.min(), z.max())
        colors = plt.cm.coolwarm(norm(z))
        plt.scatter(true_values, pred_values, c=colors, alpha=1, s=30)

        x_fit = np.linspace(true_values.min(), true_values.max(), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'k--')

        x_min = true_values.min()
        x_max = true_values.max()
        y_min = pred_values.min()
        y_max = pred_values.max()
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        plt.xlim(x_min - x_margin, x_max + x_margin)
        plt.ylim(y_min - y_margin, y_max + y_margin)

        plt.text(0.05, 0.95, fit_equation, transform=plt.gca().transAxes, fontsize=16, fontname='Times New Roman')
        plt.text(0.05, 0.90, fr"R = {r_val:.3f}", transform=plt.gca().transAxes, fontsize=16,
                 fontname='Times New Roman')
        plt.text(0.05, 0.85, fr"JSD = {jsd:.3f}", transform=plt.gca().transAxes, fontsize=16,
                 fontname='Times New Roman')

        plt.xlabel('Observed value', fontsize=18)
        plt.ylabel('Predicted value', fontsize=18)
        plt.title(f'Rice - {pheno_name}', fontsize=20)
        plt.grid(True, linestyle='--', alpha=0.5)

        save_path = save_path_prefix + f"{pheno_name}_prediction_plot.pdf"
        plt.savefig(save_path)
        plt.close()

        sorted_true_indices = np.argsort(true_values)[::-1]
        sorted_pred_indices = np.argsort(pred_values)[::-1]
        top_10_true = set(sorted_true_indices[:int(len(true_values) * 0.1)])
        top_10_pred = set(sorted_pred_indices[:int(len(pred_values) * 0.1)])
        top_10_both = top_10_true & top_10_pred
        top_10_indices_per_phenotype.append(sorted(top_10_both))
        print(f"Number of top 10% samples (pred vs true) for {pheno_name}: {len(top_10_both)}")

    return top_10_indices_per_phenotype

def plot_saliency_map(saliency, phenotype_name, top_k=10, save_path="./output_re/rice/SY/rice_saliency_map_{}.pdf"):
    # Average saliency across samples if multiple samples are provided
    if saliency.ndim > 1:
        saliency = saliency.mean(axis=0)

    # Identify top K influential SNPs
    top_indices = np.argsort(saliency)[-top_k:][::-1]
    top_values = saliency[top_indices]

    plt.figure(figsize=(12, 6))
    plt.plot(saliency, label='Saliency Score', alpha=0.5)
    plt.scatter(top_indices, top_values, c='r', label=f'Top {top_k} SNPs', zorder=5)
    plt.xlabel('SNP Index')
    plt.ylabel('Saliency Score')
    plt.title(f'Saliency Map for {phenotype_name} - Top {top_k} Influential SNPs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.format(phenotype_name))
    plt.close()

    # Save top SNPs to CSV
    top_snps_df = pd.DataFrame({
        'SNP_Index': top_indices,
        'Saliency_Score': top_values
    })
    top_snps_df.to_csv(save_path.format(phenotype_name).replace('.pdf', '.csv'), index=False)
    print(f"Top {top_k} SNPs for {phenotype_name} saved to {save_path.format(phenotype_name).replace('.pdf', '.csv')}")
    return top_indices

def plot_snp_heatmap(saliency_dict, snp_df, save_path="./output_re/rice/SY/rice_snp_heatmap.pdf"):
    combined_df = snp_df[['Chromosome', 'Position']].drop_duplicates()
    for pheno in phenotype_names:
        saliency = saliency_dict[pheno]

        temp_df = snp_df.copy()
        temp_df['Saliency_Score'] = saliency
        aggregated = temp_df.groupby(['Chromosome', 'Position'])['Saliency_Score'].max().reset_index()
        combined_df = combined_df.merge(aggregated[['Chromosome', 'Position', 'Saliency_Score']],
                                        on=['Chromosome', 'Position'], how='left').rename(
            columns={'Saliency_Score': pheno})

    pivot_df = combined_df.set_index(['Chromosome', 'Position'])[phenotype_names].T

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, cmap='viridis', cbar_kws={'label': 'Saliency Score'}, yticklabels=phenotype_names)
    plt.title('Saliency Scores by Chromosome Position Across Phenotypes', fontsize=14)
    plt.xlabel('Chromosome:Position', fontsize=12)
    plt.ylabel('Phenotype', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.close()

def plot_manhattan_snp_importance(saliency_dict, snp_df, save_path_prefix="./output_re/rice/SY/rice_manhattan_{}.pdf"):
    unique_chroms = sorted(snp_df['Chromosome'].unique(), key=lambda x: int(x) if x.isdigit() else float('inf'))
    chrom_map = {chrom: i + 1 for i, chrom in enumerate(unique_chroms)}  # Map to 1-12
    snp_df['Chrom_Num'] = snp_df['Chromosome'].map(chrom_map)

    chrom_lengths = snp_df.groupby('Chrom_Num').agg({'Position': ['min', 'max']})
    chrom_lengths['Length'] = chrom_lengths['Position']['max'] - chrom_lengths['Position']['min']
    chrom_lengths_dict = chrom_lengths['Length'].to_dict()

    cum_pos_start = {}
    current_pos = 0
    for chrom in range(1, 13):
        cum_pos_start[chrom] = current_pos
        if chrom in chrom_lengths_dict:
            current_pos += chrom_lengths_dict[chrom]

    def adjust_position(row):
        chrom = row['Chrom_Num']
        if chrom in cum_pos_start:

            min_pos = snp_df[snp_df['Chrom_Num'] == chrom]['Position'].min()
            return cum_pos_start[chrom] + (row['Position'] - min_pos)
        return 0.0


    snp_df['Cum_Pos'] = snp_df.apply(adjust_position, axis=1)


    colors = plt.cm.tab20(np.linspace(0, 1, 12))


    for pheno in phenotype_names:
        saliency = saliency_dict[pheno]
        temp_df = snp_df.copy()
        temp_df['Saliency_Score'] = saliency

        aggregated = temp_df.groupby(['Chromosome', 'Position'])['Saliency_Score'].max().reset_index()
        plot_df = snp_df[['Chromosome', 'Position', 'Chrom_Num', 'Cum_Pos']].merge(
            aggregated, on=['Chromosome', 'Position'], how='left')

        plt.figure(figsize=(16, 8))

        for chrom in range(1, 13):
            chrom_data = plot_df[plot_df['Chrom_Num'] == chrom]
            if not chrom_data.empty:
                plt.scatter(chrom_data['Cum_Pos'], chrom_data['Saliency_Score'],
                            c=[colors[chrom - 1]], alpha=0.5, s=10, label=f'Chr {chrom}')

        for chrom in range(1, 12):
            if chrom in cum_pos_start and (chrom + 1) in cum_pos_start:
                boundary = cum_pos_start[chrom + 1]
                plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

        plt.xlabel('Chromosome', fontsize=12)
        plt.ylabel('Saliency Score', fontsize=12)
        plt.title(f'Manhattan Plot for {pheno}', fontsize=14)

        tick_positions = []
        tick_labels = []
        for chrom in range(1, 13):
            if chrom in chrom_lengths_dict:
                chrom_midpoint = cum_pos_start[chrom] + (chrom_lengths_dict[chrom] / 2)
                tick_positions.append(chrom_midpoint)
                tick_labels.append(str(chrom))
        plt.xticks(tick_positions, tick_labels)

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

        save_path = save_path_prefix.format(pheno)
        plt.savefig(save_path, format='pdf', dpi=300)
        plt.close()