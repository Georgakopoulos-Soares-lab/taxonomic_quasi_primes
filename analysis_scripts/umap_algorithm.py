import polars as pl
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import *
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
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

# Enable StringCache for effective Categorical data handling
pl.enable_string_cache()

sns.set_style("ticks",{'font.family':'serif', 'font.serif':'Microsoft Sans Serif'})
plt.style.use('seaborn-v0_8-ticks')
sns.set_context("paper")

plots_dir = Path(config['plots_dir'])
plots_dir.mkdir(parents=True, exists_ok=True)

# File containing mappings of taxons to superkingdoms
mappings = pl.read_csv(config['proteome_to_superkingdom_file'], new_columns=['col'])

# Format the file
mappings = mappings.select(
    pl.col("col").str.split_exact(" ", n=1)
    .struct.rename_fields(["full_path", "domain"])
    .alias("split")
).unnest("split").with_columns([
    pl.col("full_path").str.extract(".*/(UP.*?.txt)").alias("filename")
]).drop("full_path")

mappings = mappings.with_columns([
    pl.col("filename").str.extract("(UP\d+_\d+)").alias("proteomeID")
]).drop("filename")

# Read all the phylum quasi-prime data and the data containing their mapped proteins
phylum_quasi_primes = pl.read_csv(config['phylum_7mers'], separator='\t').drop("QP_peptide_length")
species_quasi_primes = pl.read_csv(config['7mers_to_proteome_file'], separator ='\t')

# Cast to data types with lower memory requirement 
phylum_quasi_primes = phylum_quasi_primes.with_columns([
    pl.col("QP_peptide").cast(pl.Utf8),
    pl.col("Taxonomy").cast(pl.Categorical),
    pl.col("Epsilon_score").cast(pl.Float32),   
    pl.col("Domain").cast(pl.Categorical)
])

# Add the Superkingdom label to the data and cast to more efficient data types
species_quasi_primes = (species_quasi_primes
    .join(mappings, on='proteomeID')
    .with_columns([
        pl.col("proteomeID").str.split("_").list.get(0).alias("Proteome_ID"),
        pl.col("proteomeID").str.split("_").list.get(1).alias("Taxon_ID")
    ])
    .drop("proteomeID", "protein_name", "domain", "Proteome_ID")
    .rename({
        'kmer': "QP_peptide"
    }))

species_quasi_primes = species_quasi_primes.with_columns([
        pl.col("QP_peptide").cast(pl.Utf8),
        pl.col("Taxon_ID").cast(pl.Categorical)
    ])

def filter_by_epsilon_percentile(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filters a DataFrame by the median (50th percentile) of 'Epsilon_score' for each 'Taxonomy' group.
    """
    percentiles = df.group_by('Taxonomy').agg([
        pl.col('Epsilon_score').quantile(0.50).alias('percentile_50'),
        pl.col('Epsilon_score').count().alias('total_peptides'),
        pl.col('Epsilon_score').mean().alias('mean_epsilon')
    ])
    
    df_with_thresh = df.join(
        percentiles.select(['Taxonomy', 'percentile_50']), 
        on='Taxonomy'
    )
    
    filtered_df = df_with_thresh.filter(
        pl.col('Epsilon_score') >= pl.col('percentile_50')
    ).drop('percentile_50')
    return filtered_df

filtered_phylum_quasi_primes = filter_by_epsilon_percentile(phylum_quasi_primes)
species_quasi_primes_with_scores = species_quasi_primes.join(filtered_phylum_quasi_primes, on='QP_peptide').drop("Taxonomy")

# Get unique taxon ids along with the corresponding superkingdom
unique_taxons = (species_quasi_primes_with_scores
    .select(['Taxon_ID', 'Domain'])
    .unique()
    .with_row_index('idx')
)

# Get unique taxonomic quasi-prime peptides
unique_peptides = (species_quasi_primes_with_scores
    .select('QP_peptide')
    .unique()
    .with_row_index('idx')
)

# Get dimensions for the matrix
n_taxons = len(unique_taxons)
n_peptides = len(unique_peptides)

print(f"Matrix dimensions will be: {n_taxons} x {n_peptides}")

# Create mapping DataFrames for indices
df_with_indices = (species_quasi_primes_with_scores
    .join(
        unique_taxons.select(['Taxon_ID', 'idx', 'Domain']).rename({'idx': 'row_idx'}), 
        on='Taxon_ID'
    )
    .join(
        unique_peptides.select(['QP_peptide', 'idx']).rename({'idx': 'col_idx'}), 
        on='QP_peptide'
    )
)

# Get the domain labels in the same order as the unique_taxons
domain_labels = unique_taxons['Domain']

# Convert domains to numeric labels for UMAP
domain_to_num = {domain: i for i, domain in enumerate(domain_labels.unique())}
labels = domain_labels.replace_strict(domain_to_num).to_numpy()

# Extract arrays for sparse matrix construction
row_indices = df_with_indices['row_idx'].to_numpy()
col_indices = df_with_indices['col_idx'].to_numpy()
values = df_with_indices['Epsilon_score'].to_numpy()

# Create sparse matrix in CSR format
sparse_matrix = sparse.csr_matrix(
    (values, (row_indices, col_indices)),
    shape=(n_taxons, n_peptides),
    dtype=np.float32
)

# Scale non-zero data
scaler = StandardScaler(with_mean=False)
scaled_matrix = scaler.fit_transform(sparse_matrix)

# UMAP on scaled sparse matrix
umap = UMAP(
    n_neighbors=30,
    min_dist=0.1,
    n_components=2,
    densmap=True,
    metric='cosine',
    n_jobs=-1,
    # Parameters for semi-supervised learning
    target_metric='categorical',  # Since domains are discrete categories
    target_weight=0.25           # Balance between preserving distances and label information
)

embedding = umap.fit_transform(scaled_matrix, y=labels)

# Create DataFrame with UMAP coordinates
umap_df = pl.DataFrame({
    'UMAP1': embedding[:, 0],
    'UMAP2': embedding[:, 1]
})

# Add row index to match with unique_taxons
umap_df = umap_df.with_row_index('idx')

# Join with unique_taxons to get Taxon_IDs and Domain
result_df = (umap_df
    .join(
        unique_taxons.select(['Taxon_ID', 'Domain', 'idx']),
        on='idx'
    )
    .select(['Taxon_ID', 'Domain', 'UMAP1', 'UMAP2'])
)

plt.figure(figsize=(12, 8))

# Define the custom color palette for the unique domains
domain_colors = {
    'Archaea': '#0072b2',
    'Bacteria': '#e69f00',
    'Eukaryota': '#009e73',
    'Viruses': '#cc79a7'
}

# Create the main scatter plot using Seaborn
scatter = sns.scatterplot(
    data=result_df.to_pandas(),
    x='UMAP1',
    y='UMAP2',
    hue='Domain',
    palette=domain_colors,
    alpha=0.8,
    s=30,
    legend=False,
    edgecolor='white'
)

# Remove ticks while keeping the lines
scatter.set_xticks([])
scatter.set_yticks([])
scatter.set_xlabel('UMAP 1', fontsize=16)
scatter.set_ylabel('UMAP 2', fontsize=16)

# Improve the layout with some padding
plt.tight_layout()
sns.despine()

# Show the plot
plt.savefig(plots_dir / "umap.svg")
plt.show()
