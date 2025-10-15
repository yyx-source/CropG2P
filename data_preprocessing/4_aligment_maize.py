import numpy as np
import torch
import pandas as pd
import os

def parse_vcf_text(vcf_file, num_samples, num_snps):
    nucleotide_to_code = {'AA': 0, 'AT': 1, 'TA': 1, 'AC': 2, 'CA': 2, 'AG': 3, 'GA': 3,
        'TT': 4, 'TC': 5, 'CT': 5, 'TG': 6, 'GT': 6, 'CC': 7, 'CG': 8, 'GC': 8, 'GG': 9}

    genotype_data_list = []
    sample_names = None
    snp_idx = 0
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                sample_names = line.strip().split('\t')[9:]
                new_sample_names = []
                for s in sample_names:
                    if '_' in s:
                        s = '_'.join(s.split('_')[:2])
                    else:
                        s = s.split('_')[0]
                    new_sample_names.append(s)
                sample_names = new_sample_names
                print(f"Found {len(sample_names)} samples in VCF header")
                assert len(sample_names) == num_samples, f"Expected {num_samples}, got {len(sample_names)}"
            else:
                if snp_idx >= num_snps:
                    break
                fields = line.strip().split('\t')
                ref = fields[3]
                alt = fields[4]
                genotypes = fields[9:]
                cur_genotype_row = []
                for gt in genotypes:
                    gt = gt.split(':')[0]
                    if gt == "0/0":
                        pair = ref + ref
                    elif gt in ["0/1", "1/0"]:
                        pair = ref + alt
                        reverse_pair = alt + ref
                        if reverse_pair in nucleotide_to_code and pair not in nucleotide_to_code:
                            pair = reverse_pair
                    elif gt == "1/1":
                        pair = alt + alt
                    else:
                        pair = ref + ref
                    code = nucleotide_to_code.get(pair, 0)
                    cur_genotype_row.append(code)
                genotype_data_list.append(cur_genotype_row)
                snp_idx += 1

    genotype_matrix = torch.tensor(genotype_data_list, dtype=torch.uint8).transpose(0, 1)
    print(f"Processed {snp_idx} SNPs")
    return genotype_matrix, sample_names


def parse_phenotypes(csv_file, num_phenotypes):
    df = pd.read_csv(csv_file, index_col=0)
    phenotype_names = df.columns
    phenotypes = df.values
    print(f"Phenotype CSV shape: {df.shape}")
    print(f"Phenotype names: {list(phenotype_names)[:5]}")
    print(f"Raw phenotype values shape: {phenotypes.shape}")
    phenotype_tensor = torch.tensor(phenotypes, dtype=torch.float32)
    print(f"Final phenotype tensor shape: {phenotype_tensor.shape}")
    assert phenotype_tensor.shape[
               1] == num_phenotypes, f"Expected {num_phenotypes} traits, got {phenotype_tensor.shape[1]}"
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
    np.save(f"./input_re/{crop_name}_genotypes.npy", genotypes.numpy())
    torch.save(phenotypes, f"./input_re/{crop_name}_phenotypes.pt")
    with open(f"./input_re/{crop_name}_samples.txt", 'w') as f:
        f.write('\n'.join(sample_names))
    print(f"Saved {crop_name}_genotypes.npy with shape {genotypes.shape}")


if __name__ == "__main__":
    maize_genotypes, maize_samples = parse_vcf_text("./data/maize/maize_dn/maize_dn.vcf", 1453, 4549828)
    print("maize genotypes shape:", maize_genotypes.shape)  # [1453,4549828]
    maize_phenotypes, maize_pheno_samples = parse_phenotypes("./data/maize/maize_dn/maize_dn_phenotype_clean.csv", 3)
    maize_genotypes, maize_phenotypes = align_samples(maize_genotypes, maize_phenotypes, maize_samples,
                                                      maize_pheno_samples)
    save_aligned_data(maize_genotypes, maize_phenotypes, maize_samples, "maize")