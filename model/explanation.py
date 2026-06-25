import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def compute_saliency_map(model, genotypes, phenotype_idx, device, batch_size=32):
    model.eval()
    saliency_list = []
    dataset = TensorDataset(genotypes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in loader:
        batch_g = batch[0].to(device)
        batch_g.requires_grad = True
        preds = model(batch_g)
        target_pred = preds[phenotype_idx]
        target_pred.sum().backward()
        saliency = torch.abs(batch_g.grad)
        saliency_list.append(saliency.cpu())
        batch_g.grad.zero_()

    saliency = torch.cat(saliency_list, dim=0)
    return saliency.numpy()

def load_snp_positions(vcf_file_path):
    """
    Load SNP positions (CHROM, POS) from a VCF file.
    Returns a DataFrame with columns: 'Chromosome', 'Position', 'SNP_Index'.
    """
    snp_positions = []
    with open(vcf_file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            chrom = fields[0]
            pos = int(fields[1])
            snp_positions.append((chrom, pos))

    snp_df = pd.DataFrame(snp_positions, columns=['Chromosome', 'Position'])
    snp_df['SNP_Index'] = np.arange(len(snp_df))
    return snp_df

def map_snp_indices_to_positions(top_indices, vcf_file_path, phenotype_name, output_csv_path):
    """
    Map top SNP indices to their actual positions in the VCF file and save to CSV.
    """
    snp_df = load_snp_positions(vcf_file_path)

    top_snp_df = snp_df[snp_df['SNP_Index'].isin(top_indices)].copy()

    top_snp_df['Phenotype'] = phenotype_name

    top_snp_df = top_snp_df.sort_values('SNP_Index')

    top_snp_df.to_csv(output_csv_path, index=False)
    print(f"Top SNP positions for {phenotype_name} saved to {output_csv_path}")

    return top_snp_df