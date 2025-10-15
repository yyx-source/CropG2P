import numpy as np
import torch
import pandas as pd
import re

def parse_vcf_text(vcf_file, num_samples, num_snps):
    mapping = {('A', 'A'): 0, ('A', 'C'): 2, ('A', 'G'): 3, ('A', 'T'): 1, ('C', 'C'): 7,
               ('C', 'G'): 8, ('C', 'T'): 5, ('G', 'G'): 9, ('G', 'T'): 6, ('T', 'T'): 4,}

    genotype_matrix = np.zeros((num_samples, num_snps), dtype=np.int8)
    sample_names = None
    snp_idx = 0
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                sample_names = line.strip().split('\t')[9:]
                # For rice, cotton:
                sample_names = [s.split('_')[0] for s in sample_names]
                # For millet:
                # sample_names = ['_'.join(s.split('_')[:2]) for s in sample_names]
                print(f"Found {len(sample_names)} samples in VCF header")
                assert len(sample_names) == num_samples, f"Expected {num_samples}, got {len(sample_names)}"
            else:
                if snp_idx >= num_snps:
                    break
                fields = line.strip().split('\t')
                REF = fields[3].upper()
                ALT = fields[4].upper()
                genotypes = fields[9:]
                for j, gt in enumerate(genotypes):
                    gt_field = gt.split(':')[0]
                    if gt_field in [".", "./.", ".|."]:
                        genotype_matrix[j, snp_idx] = -1
                    else:
                        gt_parts = re.split(r'[/|]', gt_field)
                        if len(gt_parts) == 2:
                            try:
                                a = int(gt_parts[0])
                                b = int(gt_parts[1])
                                allele1 = REF if a == 0 else ALT
                                allele2 = REF if b == 0 else ALT
                                sorted_alleles = tuple(sorted([allele1, allele2]))
                                if sorted_alleles in mapping:
                                    genotype_matrix[j, snp_idx] = mapping[sorted_alleles]
                                else:
                                    genotype_matrix[j, snp_idx] = -1
                            except ValueError:
                                genotype_matrix[j, snp_idx] = -1
                        else:
                            genotype_matrix[j, snp_idx] = -1
                snp_idx += 1
        print(f"Processed {snp_idx} SNPs")
    return torch.tensor(genotype_matrix, dtype=torch.long), sample_names

def parse_phenotypes(csv_file, num_phenotypes):
    df = pd.read_csv(csv_file, index_col=0)
    phenotype_names = df.columns
    phenotypes = df.values
    print(f"Phenotype CSV shape: {df.shape}")
    print(f"Phenotype names: {list(phenotype_names)[:5]}")
    print(f"Raw phenotype values shape: {phenotypes.shape}")
    phenotype_tensor = torch.tensor(phenotypes, dtype=torch.float32)
    print(f"Final phenotype tensor shape: {phenotype_tensor.shape}")
    assert phenotype_tensor.shape[1] == num_phenotypes, f"Expected {num_phenotypes} traits, got {phenotype_tensor.shape[1]}"
    return phenotype_tensor, df.index.tolist()

def align_samples(genotypes, phenotypes, geno_samples, pheno_samples):
    geno_dict = {s: i for i, s in enumerate(geno_samples)}
    print(f"Genotype samples (first 5): {geno_samples[:5]}")
    print(f"Phenotype samples (first 5): {pheno_samples[:5]}")
    pheno_idx = [geno_dict[s] for s in pheno_samples if s in geno_dict]
    print(f"Aligned samples: {len(pheno_idx)} out of {len(pheno_samples)} phenotype samples")
    if len(pheno_idx) == 0:
        print("No matching samples found. Check sample ID formats.")
        raise ValueError("No samples aligned between genotypes and phenotypes")
    aligned_genotypes = genotypes[pheno_idx]
    aligned_phenotypes = phenotypes[pheno_idx]
    print(f"Aligned genotypes shape: {aligned_genotypes.shape}")
    print(f"Aligned phenotypes shape: {aligned_phenotypes.shape}")
    return aligned_genotypes, aligned_phenotypes

def save_aligned_data(genotypes, phenotypes, sample_names, crop_name):
    np.save(f"./input_re/{crop_name}_genotypes_SY.npy", genotypes.numpy())
    torch.save(phenotypes, f"./input_re/{crop_name}_phenotypes_SY.pt")
    with open(f"./input_re/{crop_name}_samples_SY.txt", 'w') as f:
        f.write('\n'.join(sample_names))
    print(f"Saved {crop_name}_genotypes.npy with shape {genotypes.shape}")

if __name__ == "__main__":
    rice_genotypes, rice_samples = parse_vcf_text("./data/rice/rice_dn/rice_dn_filtered.vcf", 1475, 484880)
    print("Rice genotypes shape:", rice_genotypes.shape)  # [1475,484880]
    rice_phenotypes, rice_pheno_samples = parse_phenotypes("./data/rice/rice_dn/rice_dn_phenotype_filtered_clean_SY.csv", 10)
    rice_genotypes, rice_phenotypes = align_samples(rice_genotypes, rice_phenotypes, rice_samples, rice_pheno_samples)
    save_aligned_data(rice_genotypes, rice_phenotypes, rice_samples, "rice")

    # cotton_genotypes, cotton_samples = parse_vcf_text("./data/cotton/cotton_qc/cotton_qc_imputed.vcf", 1197, 1117772)
    # print("cotton genotypes shape:", cotton_genotypes.shape)     # [1197,1117772]
    # cotton_phenotypes, cotton_pheno_samples = parse_phenotypes("./data/cotton/cotton_qc/cotton_qc_phenotype_filtered1.csv", 4)
    # cotton_genotypes, cotton_phenotypes = align_samples(cotton_genotypes, cotton_phenotypes, cotton_samples, cotton_pheno_samples)
    # save_aligned_data(cotton_genotypes, cotton_phenotypes, cotton_samples, "cotton")

    # millet_genotypes, millet_samples = parse_vcf_text("./data/millet/millet_qc/millet_qc_imputed.vcf", 826, 76168)
    # print("Millet genotypes shape:", millet_genotypes.shape)  # [826,76168]
    # millet_phenotypes, millet_pheno_samples = parse_phenotypes("./data/millet/millet_qc/millet_qc_phenotype_filtered1.csv", 6)
    # millet_genotypes, millet_phenotypes = align_samples(millet_genotypes, millet_phenotypes, millet_samples, millet_pheno_samples)
    # save_aligned_data(millet_genotypes, millet_phenotypes, millet_samples, "millet")