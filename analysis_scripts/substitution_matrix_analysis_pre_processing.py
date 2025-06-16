import blosum as bl
from pathlib import Path
import pandas as pd
import yaml
import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Run the taxonomic analysis pipeline.")
parser.add_argument(
    "config_file",
    type=str,
    help="Path to the configuration YAML file."
)

# Parse arguments
args = parser.parse_args()

# Load the yaml file with the specified file paths
with open(args.config_file, 'r') as f:
    config = yaml.safe_load(f)

plots_dir = Path(config['plots_dir'])
plots_dir.mkdir(parents=True, exists_ok=True)

processed_files_dir = Path(config['processed_files_dir'])
qp_over_90_dir = processed_files_dir / "qp_peptides_over_90_per_phylum"
substitution_analysis_dir = Path(config['substitution_analysis_dir'])

blosum62_matrix = bl.BLOSUM(62)
unknown_codes = ['B', 'Z', 'J', 'X', '*']

chordata_qp_over_90 = pd.read_csv(
    qp_over_90_dir.joinpath("Chordata.txt"), 
    sep='\t',
    header=None,
    names=['QP_peptide']
)

def generate_substitution_variants(peptide):
    positive_variants = set()

    for i, original_aa in enumerate(peptide):
        substitutions = blosum62_matrix[original_aa]
        
        for new_aa, score in substitutions.items():
            if score > 0 and original_aa != new_aa and new_aa not in unknown_codes:
                new_peptide = peptide[:i] + new_aa + peptide[i+1:]
                positive_variants.add(new_peptide)

    variants = list(positive_variants)
    return variants

chordata_qp_over_90['Variants'] = chordata_qp_over_90['QP_peptide'].apply(generate_substitution_variants)

output_file = substitution_analysis_dir.joinpath("all_chordata_variants.txt")
output_df_path = substitution_analysis_dir.joinpath("qps_with_variants.txt")
chordata_qp_over_90.to_csv(output_df_path, sep='\t', index=False)

all_variants_series = chordata_qp_over_90['Variants'].explode()
all_variants_series = all_variants_series.dropna()
all_variants_series.to_csv(output_file, header=False, index=False)