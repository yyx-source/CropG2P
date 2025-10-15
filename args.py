import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # --- Data params ---
    parser.add_argument("--dataset", type=str, default="rice", help="Dataset name")
    parser.add_argument("--subdataset", type=str, default="SY", help="Sub-dataset name")
    parser.add_argument("--normalize", type=bool, default=True, help="Whether to normalize data")

    # --- Training params ---
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--base_lr", type=float, default=1e-5, help="Base learning rate for scheduler")
    parser.add_argument("--max_lr", type=float, default=5e-4, help="Maximum learning rate for scheduler")
    parser.add_argument("--step_size_up", type=int, default=20, help="Step size up for scheduler")

    # --- Output params ---
    parser.add_argument("--best_model_path", type=str, default="./save_re/best_model_rice_SY_gated_inception.pt", help="Path to save best model")
    parser.add_argument("--output_csv_path", type=str, default="./output_re/rice/SY/rice_snp_importance.csv", help="Path to save SNP importance CSV")
    parser.add_argument("--output_csv_path_200", type=str, default="./output_re/rice/SY/rice_snp_importance_200.csv", help="Path to save top 200 SNP importance CSV")
    parser.add_argument("--venn_csv_path", type=str, default="./output_re/rice/SY/rice_venn.csv", help="Path to save venn CSV")
    parser.add_argument("--vcf_file_path", type=str, default="./input_re/rice_genotypes.vcf", help="Path to VCF file")

    return parser