import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import textwrap
import seaborn as sns
import sys
import ast
import polars as pl
import graphviz
import numpy as np
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

sys.setrecursionlimit(100000)

sns.set_style("ticks",{'font.family':'serif', 'font.serif':'Microsoft Sans Serif'})
plt.style.use('seaborn-v0_8-ticks')
sns.set_context("paper")

cm_data = [          
           [0.029592, 0.05624, 0.085837],      
           [0.031984, 0.059842, 0.090619],      
           [0.034477, 0.063258, 0.095163],      
           [0.037057, 0.066553, 0.09953],      
           [0.039346, 0.06983, 0.104],      
           [0.041492, 0.072993, 0.10848],      
           [0.043238, 0.076022, 0.1131],      
           [0.045051, 0.079044, 0.11765],      
           [0.046508, 0.082102, 0.12227],      
           [0.047784, 0.084964, 0.12697],      
           [0.049035, 0.087853, 0.13174],      
           [0.049925, 0.09066, 0.13649],      
           [0.050671, 0.093388, 0.14135],      
           [0.051245, 0.096126, 0.14619],      
           [0.051698, 0.098921, 0.15108],      
           [0.05216, 0.10171, 0.15603],      
           [0.052632, 0.10451, 0.16098],      
           [0.053115, 0.10741, 0.16595],      
           [0.053612, 0.11032, 0.17099],      
           [0.054125, 0.11324, 0.17602],      
           [0.054655, 0.11618, 0.18112],      
           [0.055201, 0.11913, 0.18627],      
           [0.055777, 0.12209, 0.1914],      
           [0.056384, 0.12516, 0.19661],      
           [0.056955, 0.12823, 0.20178],      
           [0.057498, 0.13131, 0.20704],      
           [0.058136, 0.13444, 0.2123],      
           [0.058867, 0.13757, 0.21759],      
           [0.059601, 0.14073, 0.22291],      
           [0.060281, 0.14391, 0.22826],      
           [0.060943, 0.14711, 0.2336],      
           [0.061699, 0.15035, 0.23901],      
           [0.062585, 0.15358, 0.24441],      
           [0.063381, 0.15688, 0.24986],      
           [0.064199, 0.16015, 0.25535],      
           [0.065067, 0.16352, 0.26083],      
           [0.065965, 0.16686, 0.26635],      
           [0.066906, 0.1702, 0.27186],      
           [0.06788, 0.17357, 0.27744],      
           [0.068913, 0.17697, 0.28302],      
           [0.069887, 0.18038, 0.28861],      
           [0.070924, 0.18381, 0.29422],      
           [0.071974, 0.18728, 0.29987],      
           [0.073118, 0.19072, 0.30554],      
           [0.074222, 0.19423, 0.31121],      
           [0.075341, 0.1977, 0.31691],      
           [0.076503, 0.20119, 0.32264],      
           [0.077709, 0.20473, 0.32836],      
           [0.078959, 0.20826, 0.33412],      
           [0.080255, 0.21181, 0.33989],      
           [0.081609, 0.21535, 0.34567],      
           [0.082887, 0.21892, 0.35147],      
           [0.084299, 0.22249, 0.35731],      
           [0.085598, 0.22609, 0.36313],      
           [0.087109, 0.22965, 0.36899],      
           [0.088529, 0.23325, 0.37486],      
           [0.090015, 0.2369, 0.38075],      
           [0.091564, 0.24048, 0.38663],      
           [0.093033, 0.2441, 0.39255],      
           [0.094704, 0.24776, 0.39848],      
           [0.096248, 0.2514, 0.40442],      
           [0.097979, 0.25504, 0.41039],      
           [0.099642, 0.25867, 0.41635],      
           [0.10138, 0.26232, 0.42232],      
           [0.1032, 0.26598, 0.42833],      
           [0.10497, 0.26965, 0.43433],      
           [0.10692, 0.27331, 0.44035],      
           [0.10879, 0.27697, 0.44639],      
           [0.1108, 0.28062, 0.45244],      
           [0.11283, 0.28431, 0.45849],      
           [0.11484, 0.28799, 0.46456],      
           [0.11699, 0.29169, 0.47064],      
           [0.11919, 0.29536, 0.47672],      
           [0.1214, 0.29907, 0.48282],      
           [0.12371, 0.30275, 0.48893],      
           [0.12613, 0.30647, 0.49506],      
           [0.12854, 0.31018, 0.50119],      
           [0.13107, 0.31389, 0.50733],      
           [0.13364, 0.31762, 0.51348],      
           [0.13625, 0.32135, 0.51964],      
           [0.139, 0.32509, 0.52582],      
           [0.14184, 0.32887, 0.532],      
           [0.14474, 0.33265, 0.5382],      
           [0.14772, 0.33642, 0.5444],      
           [0.15079, 0.34023, 0.55062],      
           [0.15395, 0.34405, 0.55685],      
           [0.15723, 0.34791, 0.56308],      
           [0.16061, 0.35178, 0.56932],      
           [0.16413, 0.3557, 0.57558],      
           [0.16771, 0.35962, 0.58184],      
           [0.17142, 0.36359, 0.58811],      
           [0.17526, 0.36759, 0.59439],      
           [0.17923, 0.37163, 0.60067],      
           [0.1833, 0.37572, 0.60697],      
           [0.18756, 0.37985, 0.61324],      
           [0.19189, 0.38402, 0.61953],      
           [0.19642, 0.38824, 0.6258],      
           [0.20102, 0.39251, 0.63207],      
           [0.20583, 0.39683, 0.63832],      
           [0.21074, 0.40122, 0.64454],      
           [0.21582, 0.40565, 0.65075],      
           [0.22103, 0.41013, 0.65692],      
           [0.22637, 0.41465, 0.66304],      
           [0.23185, 0.41923, 0.6691],      
           [0.23747, 0.42385, 0.67511],      
           [0.24318, 0.42853, 0.68103],      
           [0.24906, 0.43324, 0.68689],      
           [0.25504, 0.438, 0.69265],      
           [0.26109, 0.44277, 0.6983],      
           [0.26724, 0.44756, 0.70383],      
           [0.27351, 0.4524, 0.70924],      
           [0.27982, 0.45722, 0.7145],      
           [0.28617, 0.46205, 0.71962],      
           [0.29259, 0.46688, 0.72457],      
           [0.29905, 0.47169, 0.72935],      
           [0.30552, 0.47648, 0.73395],      
           [0.31196, 0.48123, 0.73836],      
           [0.31844, 0.48596, 0.74258],      
           [0.32488, 0.49065, 0.74659],      
           [0.33129, 0.49527, 0.7504],      
           [0.33767, 0.49983, 0.75401],      
           [0.34399, 0.50435, 0.75741],      
           [0.35027, 0.50878, 0.7606],      
           [0.35647, 0.51314, 0.76358],      
           [0.3626, 0.51744, 0.76637],      
           [0.36866, 0.52164, 0.76895],      
           [0.37464, 0.52578, 0.77135],      
           [0.38054, 0.52984, 0.77355],      
           [0.38634, 0.53381, 0.77557],      
           [0.39208, 0.53772, 0.77743],      
           [0.39773, 0.54152, 0.77911],      
           [0.4033, 0.54528, 0.78064],      
           [0.40879, 0.54895, 0.78202],      
           [0.4142, 0.55255, 0.78326],      
           [0.41955, 0.55607, 0.78438],      
           [0.42482, 0.55955, 0.78536],      
           [0.43003, 0.56296, 0.78624],      
           [0.43518, 0.56631, 0.78702],      
           [0.44025, 0.56961, 0.78769],      
           [0.44529, 0.57287, 0.78828],      
           [0.45027, 0.57608, 0.78879],      
           [0.45521, 0.57923, 0.78922],      
           [0.4601, 0.58235, 0.78958],      
           [0.46496, 0.58545, 0.78988],      
           [0.46979, 0.5885, 0.79013],      
           [0.47458, 0.59153, 0.79033],      
           [0.47934, 0.59453, 0.79049],      
           [0.48408, 0.59752, 0.7906],      
           [0.4888, 0.60047, 0.79068],      
           [0.4935, 0.60342, 0.79073],      
           [0.4982, 0.60634, 0.79076],      
           [0.50285, 0.60926, 0.79075],      
           [0.50752, 0.61215, 0.79073],      
           [0.51216, 0.61504, 0.79069],      
           [0.51681, 0.61793, 0.79063],      
           [0.52144, 0.62081, 0.79057],      
           [0.52608, 0.62368, 0.79049],      
           [0.5307, 0.62654, 0.7904],      
           [0.53532, 0.6294, 0.79031],      
           [0.53994, 0.63227, 0.79021],      
           [0.54457, 0.63513, 0.79011],      
           [0.5492, 0.638, 0.79001],      
           [0.55382, 0.64086, 0.78991],      
           [0.55845, 0.64373, 0.78981],      
           [0.5631, 0.64661, 0.78972],      
           [0.56773, 0.64949, 0.78964],      
           [0.57238, 0.65238, 0.78956],      
           [0.57704, 0.65527, 0.7895],      
           [0.58169, 0.65818, 0.78944],      
           [0.58637, 0.66109, 0.7894],      
           [0.59105, 0.66403, 0.78938],      
           [0.59574, 0.66696, 0.78937],      
           [0.60044, 0.66993, 0.78938],      
           [0.60516, 0.6729, 0.78942],      
           [0.60989, 0.67589, 0.78948],      
           [0.61464, 0.67891, 0.78957],      
           [0.61941, 0.68194, 0.78969],      
           [0.62419, 0.68501, 0.78985],      
           [0.62899, 0.6881, 0.79004],      
           [0.63382, 0.69121, 0.79027],      
           [0.63866, 0.69435, 0.79055],      
           [0.64352, 0.69754, 0.79087],      
           [0.64842, 0.70076, 0.79124],      
           [0.65334, 0.70401, 0.79167],      
           [0.65829, 0.70731, 0.79216],      
           [0.66327, 0.71065, 0.79272],      
           [0.66828, 0.71403, 0.79334],      
           [0.67332, 0.71747, 0.79403],      
           [0.6784, 0.72094, 0.79479],      
           [0.68351, 0.72448, 0.79564],      
           [0.68866, 0.72807, 0.79657],      
           [0.69384, 0.73173, 0.79759],      
           [0.69907, 0.73544, 0.79871],      
           [0.70433, 0.73922, 0.79993],      
           [0.70964, 0.74305, 0.80124],      
           [0.71499, 0.74695, 0.80266],      
           [0.72038, 0.75093, 0.8042],      
           [0.72581, 0.75497, 0.80584],      
           [0.73128, 0.75908, 0.80761],      
           [0.73679, 0.76326, 0.80949],      
           [0.74235, 0.76752, 0.8115],      
           [0.74795, 0.77185, 0.81362],      
           [0.75358, 0.77626, 0.81589],      
           [0.75926, 0.78073, 0.81827],      
           [0.76498, 0.78528, 0.8208],      
           [0.77073, 0.7899, 0.82345],      
           [0.77651, 0.7946, 0.82624],      
           [0.78233, 0.79936, 0.82916],      
           [0.78818, 0.80419, 0.83221],      
           [0.79406, 0.8091, 0.83539],      
           [0.79997, 0.81406, 0.8387],      
           [0.8059, 0.81908, 0.84215],      
           [0.81186, 0.82418, 0.84572],      
           [0.81783, 0.82933, 0.84942],      
           [0.82383, 0.83454, 0.85323],      
           [0.82985, 0.8398, 0.85717],      
           [0.83587, 0.84511, 0.86123],      
           [0.84192, 0.85047, 0.8654],      
           [0.84796, 0.85588, 0.86968],      
           [0.85402, 0.86133, 0.87406],      
           [0.8601, 0.86682, 0.87855],      
           [0.86617, 0.87236, 0.88313],      
           [0.87225, 0.87793, 0.88781],      
           [0.87832, 0.88353, 0.89257],      
           [0.88441, 0.88917, 0.89742],      
           [0.89049, 0.89484, 0.90235],      
           [0.89657, 0.90052, 0.90736],      
           [0.90265, 0.90624, 0.91243],      
           [0.90873, 0.91198, 0.91758],      
           [0.91481, 0.91775, 0.92278],      
           [0.92088, 0.92353, 0.92804],      
           [0.92695, 0.92933, 0.93336],      
           [0.93302, 0.93515, 0.93873],      
           [0.9391, 0.94098, 0.94414],      
           [0.94517, 0.94683, 0.94959],      
           [0.95123, 0.9527, 0.95509],      
           [0.9573, 0.95858, 0.96061],      
           [0.96337, 0.96446, 0.96617],      
           [0.96943, 0.97036, 0.97175],      
           [0.9755, 0.97627, 0.97736],      
           [0.98157, 0.9822, 0.98298],      
           [0.98765, 0.98813, 0.98863],      
           [0.99372, 0.99407, 0.99429],      
           [0.9998, 1, 0.99996]]      
      
oslo_map = mcolors.LinearSegmentedColormap.from_list('oslo', cm_data)  

substitution_analysis_dir = Path(config['substitution_analysis_dir'])

variant_mappings = pd.read_csv(
    config['substitution_analysis_peptide_match_results'], 
    sep ='\t', 
    comment='#',
    header=None,
    names=[
        "QP_variant",
        "Protein_mapping",
        "Protein_length",
        "Match_start",
        "Match_end",
        "Column_to_drop"
    ]
).drop(columns=['Column_to_drop'])

variant_mappings['Protein_ID'] = variant_mappings['Protein_mapping'].str.split('|').str[1]
processed_variant_mappings = variant_mappings.drop(columns=['Protein_mapping', "Protein_length", 'Match_start', 'Match_end'])[['QP_variant', 'Protein_ID']]

def read_quasi_prime_peptide_data(qp_file_path: str, 
               qp_peptide_length: str) -> pl.DataFrame:
    
    qp_peptides_df= pl.read_csv(
        qp_file_path,
        separator='\t',  
        has_header=False 
    )
    
    if qp_peptide_length == "7mer":
        peptide_length = 7
        regex_number = 8
    elif qp_peptide_length == "6mer":
        peptide_length = 6
        regex_number = 7
    elif qp_peptide_length == "5mer":  
        peptide_length = 5
        regex_number = 6
    else:
        raise ValueError(f"Unsupported peptide length: {qp_peptide_length}")
        
    qp_peptides_df = qp_peptides_df.with_columns([
        pl.col("column_1").str.slice(0, peptide_length).alias("QP_peptide"),

        pl.col("column_1").str.extract(fr"^.{{{regex_number}}}(.*?)\s+\d+\.?\d*%$").alias("Taxonomy"),

        pl.col("column_1").str.extract(r"(\d+\.?\d*)%$").cast(pl.Float64).alias("Epsilon_score")
    ])

    qp_peptides_df = qp_peptides_df.drop("column_1")
    
    qp_peptides_df = qp_peptides_df.with_columns(
    pl.col("Taxonomy").str.strip_chars_end(" ").alias("Taxonomy")).with_columns(
    (pl.col("Taxonomy").str.slice(0, 1).str.to_uppercase() +
     pl.col("Taxonomy").str.slice(1)).alias("Taxonomy"))
    
    qp_peptides_df = qp_peptides_df.with_columns(
        pl.lit(qp_peptide_length).alias("QP_peptide_length"))

    qp_peptides_df = qp_peptides_df.with_columns([
        pl.col("QP_peptide").cast(pl.Utf8),
        pl.col("Taxonomy").cast(pl.Utf8),
        pl.col("Epsilon_score").cast(pl.Float32),
        pl.col("QP_peptide_length").cast(pl.Categorical)
    ])
    return qp_peptides_df

phylum_quasi_prime_7mers = read_quasi_prime_peptide_data(
    config['phylum_7mers'], 
    "7mer"
)

protein_to_taxid = pd.read_csv(
    config['substitution_analysis_id_mapping'],
    sep='\t',
    header=None,
    names=[
        "Protein_ID",
        "Database",
        "Taxon_ID"
    ]
).drop(columns=['Database'])

final_variant_mappings = pd.merge(processed_variant_mappings, protein_to_taxid, on='Protein_ID', how='left')

final_variant_mappings = final_variant_mappings.astype({
    'Taxon_ID': 'Int64',
})

final_variant_mappings = final_variant_mappings.astype({
    'Taxon_ID': 'str',
})

final_variant_mappings = final_variant_mappings.fillna("No match")

full_lineage = pd.read_csv(
    config['ranked_lineage_file'], 
    sep ='\t',
    header=None,
    names=[
        "Taxon_ID", 1,
        "Taxon_name", 2,
        "Species", 3,
        "Genus", 4,
        "Family", 5,
        "Order", 6,
        "Class", 7,
        "Phylum", 8,
        "Kingdom", 9,
        "Superkingdom", 10
    ]
).drop(columns=[
    1,2,3, "Species",
    "Genus", 4,
    "Family", 5,
    "Order", 6,
    "Class", 7, 8, 9, 10
])

full_lineage = full_lineage.astype({
    'Taxon_ID': 'str',
})

final_variant_mappings['Taxon_ID'] = final_variant_mappings['Taxon_ID'].str.replace('<NA>', 'No match')

missing_taxon_ids = pd.DataFrame({
    'Taxon_ID': ['No match', '1283342', '1283343', '1554497'], 
    'Taxon_name': ['No match', 'Ranid herpesvirus 1 (strain McKinnell)', 'Anguillid herpesvirus 1 (isolate Anguilla anguilla/Netherlands/500138/1998)', 'Hibiscus green spot virus (isolate Citrus volkameriana /USA/WAI 1-1/2009)'], 
    'Phylum': ['No match', 'Peploviricota', 'Peploviricota', 'Kitrinoviricota'],
    'Kingdom': ['No match', 'Heunggongvirae', 'Heunggongvirae', 'Orthornavirae'],
    'Superkingdom': ['No match', 'Viruses', 'Viruses', 'Viruses'],
})

full_lineage = pd.concat([full_lineage, missing_taxon_ids], ignore_index=True)

updated_taxon_ids = pd.read_csv(
    config['merged_dmp_file'],
    sep='\t',
    header=None,
    names=[
        "Old_Taxon_ID", 0,
        "New_Taxon_ID", 1
    ]
).drop(columns=[0,1]).astype({
    "Old_Taxon_ID": "str",
    "New_Taxon_ID": "str",
})

variant_mappings_with_taxon_info = pd.merge(final_variant_mappings, full_lineage, on='Taxon_ID', how='left')

nan_superkingdom_df = variant_mappings_with_taxon_info[variant_mappings_with_taxon_info['Superkingdom'].isna()]
missing_taxon_ids = nan_superkingdom_df['Taxon_ID'].unique().tolist()
missing_taxon_id_mapping = updated_taxon_ids[updated_taxon_ids['Old_Taxon_ID'].isin(missing_taxon_ids)]
id_map = pd.Series(missing_taxon_id_mapping.New_Taxon_ID.values, index=missing_taxon_id_mapping.Old_Taxon_ID)
variant_mappings_with_taxon_info['Taxon_ID'] = variant_mappings_with_taxon_info['Taxon_ID'].map(id_map).fillna(variant_mappings_with_taxon_info['Taxon_ID'])
updated_variant_mappings = variant_mappings_with_taxon_info.drop(columns=['Taxon_name', 'Phylum', 'Kingdom', 'Superkingdom'])
final_variant_mappings = pd.merge(updated_variant_mappings, full_lineage, on='Taxon_ID', how='left')

taxonomic_levels = ['Phylum', 'Kingdom', 'Superkingdom']

final_variant_mappings_tax_columns = final_variant_mappings[taxonomic_levels]
final_variant_mappings_anchors = final_variant_mappings_tax_columns.bfill(axis=1)

for col in taxonomic_levels:
    original_na_mask = final_variant_mappings[col].isna()
    
    anchor_found_mask = final_variant_mappings_anchors[col].notna()
    
    effective_mask_to_fill = original_na_mask & anchor_found_mask
    
    if effective_mask_to_fill.any():
        anchor_values_for_filling = final_variant_mappings_anchors.loc[effective_mask_to_fill, col]
        
        fill_strings = anchor_values_for_filling.astype(str) + " I.S."
    
        final_variant_mappings.loc[effective_mask_to_fill, col] = fill_strings

final_variant_mappings = final_variant_mappings.drop_duplicates()

peptide_phyla = final_variant_mappings.groupby('QP_variant')['Phylum'].apply(lambda x: set(x)).reset_index()

def categorize_peptide(phyla_set):
    if not phyla_set:
        return 'Phylum Undetermined'
    
    if phyla_set == {'No match'}:
        return 'No match'
    
    elif phyla_set == {'Chordata'}:
        return 'Found only in Chordata'
    
    else:
        return 'Found also in other Phyla'

peptide_phyla['Category'] = peptide_phyla['Phylum'].apply(categorize_peptide)
category_counts = peptide_phyla['Category'].value_counts()  


plt.figure(figsize=(6, 6))

colors = [
    '#726fad', 
    '#a8aaac', 
    '#178a6a'
]
colors = colors[:len(category_counts)]

wrapped_labels = [textwrap.fill(label, width=15) for label in category_counts.index]

wedges, texts, autotexts = plt.pie(category_counts.values, 
                                   labels=wrapped_labels,
                                   autopct='%1.2f%%',
                                   colors=colors,
                                   startangle=90,
                                   explode=[0.02] * len(category_counts),
                                   shadow=False)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')

plt.text(0, 0, f'{category_counts.sum():,}\nVariants', 
         ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=1))


plt.tight_layout()
plt.savefig(plots_dir / "pie_chart.svg", format='svg', bbox_inches='tight', pad_inches=0)
plt.show()

palette = {
    'Archaea': '#0072b2',
    'Bacteria': '#e69f00',
    'Eukaryota': '#009e73',
    'Viruses': '#cc79a7',
    'No match': '#B7B9BD'
}

phylum_counts = final_variant_mappings['Phylum'].value_counts()

plot_df = phylum_counts.reset_index()
plot_df.columns = ['Phylum', 'Count']
plot_df['Percentage'] = (plot_df['Count'] / plot_df['Count'].sum()) * 100

phylum_to_superkingdom = final_variant_mappings.drop_duplicates(subset=['Phylum']).set_index('Phylum')['Superkingdom']
plot_df['Superkingdom'] = plot_df['Phylum'].map(phylum_to_superkingdom)

label_replacement_map = {
    'Archaea I.S.': 'Archaeal I.S.',
    'Bacteria I.S.': 'Bacterial I.S.',
    'Viruses I.S.': 'Viral I.S.'
}

plot_df['Phylum'] = plot_df['Phylum'].replace(label_replacement_map)
plot_df['Phylum'] = plot_df['Phylum'].str.replace('Candidatus ', 'C. ', regex=False)

plt.figure(figsize=(5, 20))

sns.barplot(
    x='Percentage', 
    y='Phylum', 
    hue='Superkingdom',
    data=plot_df,
    dodge=False,
    palette=palette,
    legend=False
)

plt.xlabel('Percentage (%)', fontsize=10)
plt.ylabel('')
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.xscale('log')
plt.grid(visible=True, which='major', alpha=0.7)

sns.despine()
plt.tight_layout()
plt.savefig(plots_dir / "protein_distribution.svg", format='svg', bbox_inches='tight', pad_inches=0)
plt.show()

def parse_string_list(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []

original_data = pd.read_csv(
    substitution_analysis_dir / "qps_with_variants.txt",
    sep='\t',
    dtype={0: str},
    converters={1: parse_string_list}
)

qp_peptide_to_variant = original_data.explode('Variants')
merged_data = pd.merge(
    qp_peptide_to_variant,
    final_variant_mappings,
    left_on='Variants',
    right_on='QP_variant',
    how='left'
)
superkingdom_counts = merged_data.groupby(['QP_peptide', 'Superkingdom']).size().reset_index(name='count')
total_counts = merged_data.groupby('QP_peptide').size().reset_index(name='total_count')
superkingdom_counts = superkingdom_counts.merge(total_counts, on='QP_peptide')
superkingdom_counts['normalized_proportion'] = (superkingdom_counts['count'] / superkingdom_counts['total_count']) * 100
heatmap_data = superkingdom_counts.pivot(index='QP_peptide', 
                                              columns='Superkingdom', 
                                              values='normalized_proportion').fillna(0)
heatmap_data = heatmap_data[['Archaea', 'Bacteria', 'Eukaryota', 'Viruses', 'No match']]

g = sns.clustermap(
    heatmap_data, 
    annot=False, 
    cmap=oslo_map.reversed(), 
    linewidths=0,
    figsize=(6, 10),
    col_cluster=False,
    yticklabels=False,
    dendrogram_ratio=0.1,
    cbar_pos=(0.8, 0.95, 0.15, 0.02), #(left, bottom, width, height)
    cbar_kws={
        "orientation": "horizontal",
        'label': "Percentage (%)"
    }
)

g.ax_heatmap.set_xlabel('')
g.ax_heatmap.set_ylabel('Taxonomic Quasi-Prime Variant Superkingdom Distribution', fontsize=10)
g.ax_heatmap.tick_params(axis='y', which='both', length=0)
g.ax_heatmap.tick_params(axis='x', labelsize=10)

plt.savefig(plots_dir / "clustermap.svg", format='svg', bbox_inches='tight', pad_inches=0)
plt.show()

fully_eukaryotic_qps = heatmap_data[heatmap_data['Eukaryota']==100.0].index.tolist()
fully_eukaryotic_qps_with_variants = merged_data[merged_data['QP_peptide'].isin(fully_eukaryotic_qps)][['QP_peptide', 'QP_variant']].drop_duplicates()

completely_no_match_qp = heatmap_data[heatmap_data['No match']==100.0].index.tolist()

fully_chordata_variants = peptide_phyla[peptide_phyla['Phylum']=={"Chordata"}]['QP_variant'].tolist()
chordata_only = merged_data[merged_data['QP_variant'].isin(fully_chordata_variants)][['QP_peptide', 'QP_variant']].drop_duplicates()

peptide_variant_counts = chordata_only.groupby('QP_peptide')['QP_variant'].count().reset_index()
peptide_variant_counts.columns = ['QP_peptide', 'Number of Variants']
peptide_variant_counts = peptide_variant_counts.sort_values(by='Number of Variants', ascending=False)

max_num_of_variants = peptide_variant_counts['Number of Variants'].max()
top_qp_with_variants = peptide_variant_counts[peptide_variant_counts['Number of Variants']==max_num_of_variants]['QP_peptide'].tolist()

qp_with_variant_and_max_score = phylum_quasi_prime_7mers.filter(pl.col("QP_peptide").is_in(top_qp_with_variants)).sort("Epsilon_score", descending=True)[0]['QP_peptide'].to_list()[0]
final_qps_for_tree = chordata_only[chordata_only['QP_peptide']==top_qp_with_variant_and_max_score]['QP_variant'].tolist()
final_qps_for_tree.append(qp_with_variant_and_max_score)

def truncate_colormap(cmap, minval, maxval):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, 100)))
    return new_cmap

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.output = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, peptide, score):
        node = self.root
        for char in peptide:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.output = {'peptide': peptide, 'score': score}

    def to_bracket_notation(self):
        kmers = []
        
        def collect_kmers(node, current_kmer=""):
            if node.is_end_of_word:
                kmers.append(current_kmer)
            
            for char, child in node.children.items():
                collect_kmers(child, current_kmer + char)
        
        collect_kmers(self.root)
        
        if not kmers:
            return ""
        
        min_length = min(len(kmer) for kmer in kmers)
        
        shortest_kmers = [kmer for kmer in kmers if len(kmer) == min_length]
        
        position_chars = [set() for _ in range(min_length)]
        
        for word in shortest_kmers:
            for i, char in enumerate(word):
                position_chars[i].add(char)
        
        bracket_pattern = ""
        for pos_set in position_chars:
            if len(pos_set) == 1:
                bracket_pattern += list(pos_set)[0]
            else:
                sorted_chars = sorted(list(pos_set))
                bracket_pattern += "[" + "".join(sorted_chars) + "]"
                
        return bracket_pattern

def visualize_trie(trie):
    dot = graphviz.Digraph()
    dot.attr('graph', rankdir='TB', splines='line')
    dot.attr('node', shape='circle', fixedsize='true', width='0.3', fontsize='5')
    dot.node('root', 'root', width='0.8', fontsize='10', style='filled', fillcolor='#C2C2C2')

    def _collect_scores(node, scores_list):
        if node.is_end_of_word and node.output:
            scores_list.append(node.output['score'])
        for child in node.children.values():
            _collect_scores(child, scores_list)

    all_scores = []
    _collect_scores(trie.root, all_scores)

    log_norm = None
    colormap = None
    if all_scores:
        log_scores = np.log10([s + 1e-9 for s in all_scores])
        log_norm = mcolors.Normalize(vmin=min(log_scores), vmax=max(log_scores))
        original_cmap = plt.colormaps.get_cmap('Blues')
        colormap = truncate_colormap(original_cmap, 0.0, 0.7)

    queue = [(trie.root, 'root')]
    node_id_counter = 0

    while queue:
        parent_node, parent_id_str = queue.pop(0)

        for char, child_node in sorted(parent_node.children.items()):
            node_id_counter += 1
            child_id_str = f'node{node_id_counter}'

            if child_node.is_end_of_word:
                peptide = child_node.output['peptide']
                score = child_node.output['score']
                label = f"{peptide}\nε-score: {score:.2f}%"
                
                
                if colormap and log_norm:
                    log_score = np.log10(score + 1e-19)
                    rgba_color = colormap(log_norm(log_score))
                    node_color = mcolors.to_hex(rgba_color)

                dot.node(child_id_str, label, shape='ellipse', width='1.2', fontsize='10', style='filled', fillcolor=node_color)
            else:
                dot.node(child_id_str, '', style='filled', fillcolor='#E3EEF8')

            dot.edge(parent_id_str, child_id_str, label=char)
            queue.append((child_node, child_id_str))

    return dot

data_for_tree = phylum_quasi_prime_7mers.filter(pl.col("QP_peptide").is_in(final_qps_for_tree)).to_pandas()

my_trie = Trie()

for index, row in data_for_tree.iterrows():
    my_trie.insert(row['QP_peptide'], row['Epsilon_score'])

dot = visualize_trie(my_trie)
dot.render(plots_dir / 'trie_for_top_qp', format='svg', view=True)

bracket_notation = my_trie.to_bracket_notation()
print(f"Bracket Notation: {bracket_notation}")