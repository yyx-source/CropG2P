import pandas as pd

base_path = "../data/rice/rice_dn/rice_dn"

# Process phenotype CSV: remove samples with missing values
pheno = pd.read_csv(f"{base_path}_phenotype_filtered.csv", index_col=0)
print("Original phenotype sample count:", pheno.shape[0])
pheno_clean = pheno.dropna()
print("Cleaned phenotype sample count:", pheno_clean.shape[0])

# Save cleaned phenotype data
pheno_clean.to_csv(f"{base_path}_phenotype_filtered_clean.csv")
print("Cleaned phenotype data saved to:", f"{base_path}_phenotype_clean.csv")

# Get list of retained sample IDs and convert to VCF format
samples_to_keep = pheno_clean.index.tolist()
samples_to_keep_vcf_format = [f"{sample}" for sample in samples_to_keep]
samples_keep_set = set(samples_to_keep_vcf_format)
print("Retained sample ID count:", len(samples_to_keep))
print("Converted sample ID examples:", samples_to_keep_vcf_format[:5])

# Process genotype VCF: retain corresponding samples
with open(f"{base_path}.vcf", "r") as vcf_in, open(f"{base_path}_filtered.vcf", "w") as vcf_out:
    keep_indices = None

    for line in vcf_in:
        if line.startswith("##"):

            vcf_out.write(line)
        elif line.startswith("#CHROM"):

            header = line.strip().split("\t")
            samples_in_vcf = header[9:]
            print("Sample count in VCF:", len(samples_in_vcf))
            print("Sample ID examples in VCF:", samples_in_vcf[:5])

            keep_indices = [i for i, sample in enumerate(samples_in_vcf) if sample in samples_keep_set]
            print("Retained sample index count:", len(keep_indices))

            if len(keep_indices) == 0:
                print("Warning: No matching sample IDs found. Check if phenotype and VCF sample IDs match!")
                print("Converted sample ID examples from phenotype:", list(samples_keep_set)[:5])
                print("Check if ID conversion logic is correct!")
                break

            new_header = header[:9] + [samples_in_vcf[i] for i in keep_indices]
            vcf_out.write("\t".join(new_header) + "\n")
        else:

            cols = line.strip().split("\t")
            fixed_cols = cols[:9]

            keep_cols = [cols[9 + i] for i in keep_indices] if keep_indices else []

            new_line = fixed_cols + keep_cols
            vcf_out.write("\t".join(new_line) + "\n")

print("VCF file filtering completed. Output file:", f"{base_path}_filtered.vcf")