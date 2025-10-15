import subprocess
import os

def remove_duplicate_variants(vcf_path):
    """
    Remove duplicate markers in a VCF file based on chromosome and position, retaining the first occurrence.
    :param vcf_path: Path to the VCF file
    """
    seen = set()
    header_lines = []
    variant_lines = []

    with open(vcf_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                header_lines.append(line)
            else:
                parts = line.split('\t')
                if len(parts) < 2:
                    print(f"Warning: Skip invalid line: {line.strip()}")
                    continue
                chrom = parts[0].strip()
                pos = parts[1].strip()
                key = (chrom, pos)
                if key in seen:
                    print(f"Found duplicate marker: {chrom}:{pos}, skipped.")
                    continue
                seen.add(key)
                variant_lines.append(line)

    with open(vcf_path, 'w') as f:
        f.writelines(header_lines)
        f.writelines(variant_lines)

def convert_plink_to_vcf(bed_file_prefix, output_vcf):
    """
    Convert .bed, .bim, .fam files to .vcf file using Plink:
    param bed_file_prefix: Common prefix for .bed, .bim, .fam files
    param output_vcf: Path for the output .vcf file
    """
    command = [
        plink_path,
        '--bfile', bed_file_prefix,
        '--recode', 'vcf',
        '--set-missing-var-ids', '@:#',
        '--allow-extra-chr',
        '--out', output_vcf.replace('.vcf', '')
    ]
    print("Running Plink command for file conversion...")
    # Execute command
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout)
    print(f"File converted to {output_vcf} successfully.")

    # Remove duplicate markers
    print("Removing duplicate markers...")
    remove_duplicate_variants(f"{output_vcf}.vcf")
    print("Duplicate markers removed.")


def convert_vcf_to_plink(vcf_file, output_prefix):
    """
    Convert VCF file to .bed, .bim, .fam files using Plink.
    :param vcf_file: Path to input .vcf or .vcf.gz file
    :param output_prefix: Prefix for output .bed/.bim/.fam files
    """
    command = [
        plink_path,
        '--vcf', vcf_file,
        '--make-bed',
        '--allow-extra-chr',
        '--out', output_prefix
    ]
    print("Running Plink command for file conversion...")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout)
    print(f"File converted to {output_prefix}.bed/.bim/.fam successfully.")



if __name__ == "__main__":
    plink_path = r"F:\Software\plink_win64_20241022\plink.exe"

    ## Convert plink to vcf
    input_prefix = "../data/maize/maize"
    output_vcf = "../data/maize/maize_dn/maize_dn"

    convert_plink_to_vcf(input_prefix, output_vcf)

    ## Convert vcf to plink
    # input_vcf = "../data/millet/millet_qc/millet_qc_imputed.vcf.vcf.gz"
    # output_plink = "../data/millet/im/millet_imputed"
    # convert_vcf_to_plink(input_vcf, output_plink)