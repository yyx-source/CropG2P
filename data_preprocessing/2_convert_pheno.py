import os
import numpy as np
import pandas as pd
from bitarray import bitarray
import csv

def read_plink_files(base_path, pheno_path):
    """ Read PLINK format files and phenotype file """
    fam_path = f"{base_path}.fam"
    if not os.path.exists(fam_path):
        raise FileNotFoundError(f"FAM file not found: {fam_path}")
    fam = pd.read_csv(fam_path, sep='\s+', header=None,
                      names=['FID', 'IID', 'FatherID', 'MotherID', 'Sex', 'Phenotype'])

    bim_path = f"{base_path}.bim"
    if not os.path.exists(bim_path):
        raise FileNotFoundError(f"BIM file not found: {bim_path}")
    bim = pd.read_csv(bim_path, sep='\s+', header=None,
                      names=['CHR', 'SNP_ID', 'POS_CM', 'POS_BP', 'ALLELE1', 'ALLELE2'])

    if not os.path.exists(pheno_path):
        raise FileNotFoundError(f"Phenotype file not found: {pheno_path}")
    pheno = pd.read_csv(pheno_path, sep='\s+', header=None)

    return fam, bim, pheno


def parse_bed_in_chunks(base_path, fam, bim, chunk_size=10000):
    """ Parse BED file in chunks and stream to CSV """
    bed_path = f"{base_path}.bed"
    n_samples = len(fam)
    n_snps = len(bim)
    bytes_per_snp = (n_samples + 3) // 4
    sample_ids = fam['IID'].tolist()

    geno_csv_path = f"{base_path}_genotype.csv"
    header_written = False

    with open(bed_path, 'rb') as bed_file, open(geno_csv_path, 'w', newline='') as csv_file:
        writer = None

        bed_file.seek(3)

        for start_snp in range(0, n_snps, chunk_size):
            end_snp = min(start_snp + chunk_size, n_snps)
            current_chunk_size = end_snp - start_snp

            chunk_data = np.empty((n_samples, current_chunk_size), dtype=np.int8)

            for i in range(current_chunk_size):
                snp_idx = start_snp + i
                bed_file.seek(3 + snp_idx * bytes_per_snp)
                byte_array = np.frombuffer(bed_file.read(bytes_per_snp), dtype=np.uint8)
                bits = bitarray()
                bits.frombytes(byte_array.tobytes())

                for sample_idx in range(n_samples):
                    bit_pos = 2 * sample_idx
                    code = bits[bit_pos:bit_pos + 2]
                    if code == bitarray('10'):
                        chunk_data[sample_idx, i] = 1
                    elif code == bitarray('11'):
                        chunk_data[sample_idx, i] = 2
                    elif code == bitarray('00'):
                        chunk_data[sample_idx, i] = 0
                    else:
                        chunk_data[sample_idx, i] = -9  # Missing value

            chunk_df = pd.DataFrame(
                chunk_data,
                columns=bim['SNP_ID'][start_snp:end_snp],
                index=sample_ids
            )

            if not header_written:
                chunk_df.to_csv(csv_file, mode='w')
                header_written = True
            else:

                temp_path = f"{base_path}_temp.csv"
                with open(temp_path, 'w', newline='') as temp_file:
                    for sample_idx in range(n_samples):
                        # Extract all SNP data for current sample
                        sample_data = chunk_df.iloc[sample_idx, :].tolist()
                        # Write to temp file
                        temp_file.write(','.join(map(str, sample_data)) + '\n')


                with open(temp_path, 'r') as temp_file:
                    for row_idx, new_cols in enumerate(temp_file):
                        if row_idx >= n_samples:
                            raise ValueError(f"Temp file row index {row_idx} exceeds sample count")
                        # Find corresponding row in main file and append new columns
                        main_row = [sample_ids[row_idx]] + new_cols.strip().split(',')
                        if row_idx == 0:
                            # Update header
                            writer = csv.writer(csv_file)
                            writer.writerow(main_row)
                        else:
                            writer.writerow(main_row)

                os.remove(temp_path)

            print(f"Processed SNPs {start_snp}-{end_snp}")

def save_to_csv(base_path, fam, bim, pheno, chunk_size=10000):
    """ Save to CSV in chunks """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    print(f"Starting genotype data conversion (total {len(bim)} SNPs)...")
    parse_bed_in_chunks(base_path, fam, bim, chunk_size)

    pheno_csv_path = f"{base_path}_phenotype.csv"
    pheno.to_csv(pheno_csv_path, index=False, header=False)
    print(f"Conversion complete!\nGenotype file: {base_path}_genotype.csv\nPhenotype file: {pheno_csv_path}")

def sync_files(base_path):
    """ Synchronize phenotype file with genotype file by matching sample IDs """
    genotype_df = pd.read_csv(f"{base_path}_genotype.csv")
    phenotype_df = pd.read_csv(f"{base_path}_phenotype.csv")

    genotype_samples = genotype_df.iloc[:, 0].values
    phenotype_samples = phenotype_df.iloc[:, 0].values

    common_samples = set(genotype_samples).intersection(set(phenotype_samples))

    phenotype_filtered = phenotype_df[phenotype_df.iloc[:, 0].isin(common_samples)]

    phenotype_filtered.to_csv(f"{base_path}_phenotype_filtered.csv", index=False)

    print(f"Filtered phenotype file saved to: {base_path}_phenotype_filtered.csv")


if __name__ == "__main__":
    base_path = "./data/rice/rice"
    pheno_path = "./data/rice/rice.pheno"

    fam, bim, pheno = read_plink_files(base_path, pheno_path)
    save_to_csv(base_path, fam, bim, pheno, chunk_size=10000)
    sync_files(base_path)