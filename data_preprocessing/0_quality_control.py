import subprocess

plink_path = r"F:\Software\plink_win64_20241022\plink.exe"

input_prefix = "./data/rice/rice"  # input（.bed, .bim, .fam）
output_prefix = "./data/rice/rice_dn/rice_dn"  # output

# PLINK quality control command
plink_cmd = [
    plink_path,
    "--bfile", input_prefix,
    '--mind', '0.1',                 # Filter individual missing rate>10%
    '--geno', '0',                 # Filter SNP deletion rate>0%
    '--maf', '0.05',                 # Filter for allele frequencies<5%
    '--hwe', '1e-6',                 # Filter Hardy Weinberg equilibrium p-value<1e-6
    '--allow-extra-chr',             # Allow unconventional chromosomes
    '--make-bed',                    # Output file
    '--out', output_prefix
]

result = subprocess.run(plink_cmd, check=True)
print("Quality control successfully completed!")
print(result.stdout)

