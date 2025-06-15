import blosum as bl
from pathlib import Path
import pandas as pd

base_dir = Path("/storage/group/izg5139/default/lefteris/taxonomic_qps_project/")
qp_over_90_dir = base_dir.joinpath("qp_peptides_over_90_per_phylum")
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

output_file = base_dir.joinpath("substitution_matrix_analysis").joinpath("all_chordata_variants.txt")
output_df_path = base_dir.joinpath("substitution_matrix_analysis").joinpath("qps_with_variants.txt")
chordata_qp_over_90.to_csv(output_df_path, sep='\t', index=False)

all_variants_series = chordata_qp_over_90['Variants'].explode()
all_variants_series = all_variants_series.dropna()
all_variants_series.to_csv(output_file, header=False, index=False)