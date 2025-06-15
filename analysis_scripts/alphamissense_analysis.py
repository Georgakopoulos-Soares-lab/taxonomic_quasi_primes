import polars as pl
import numpy as np
from pathlib import Path
from typing import *
import seaborn as sns 
import matplotlib.pyplot as plt
import scipy.stats as stats

# Setup base directories
base_dir = Path("/storage/group/izg5139/default/lefteris")
alphamissense_data = base_dir / 'alphamissense_files'

# Read Chordata quasi-prime data 
chordata_quasi_primes = pl.read_csv(base_dir/"qp_peptides_over_90_per_phylum/formated_reference_mappings/Chordata_formated_reference_mappings.txt", separator = '\t')

# Read human AlphaMissense data 
hg38_missense_data = pl.read_csv(alphamissense_data/ 'AlphaMissense_hg38.tsv', 
                                 separator = '\t', 
                                 comment_prefix='#', 
                                 new_columns=['Chrom', 
                                              'Possition', 
                                              'REF_nucleotide', 
                                              'ALT_nucleotide', 
                                              'Genome', 
                                              'Protein_accession',
                                              'Ensembl_transcript',
                                              'Variant', 
                                              'Score', 
                                              'Pathogenicity'])

hg38_missense_data = hg38_missense_data.drop(['Chrom', 'Possition', 'REF_nucleotide', 'ALT_nucleotide', 'Genome', 'Ensembl_transcript'])

# Extract the variant locations
hg38_missense_data = hg38_missense_data.with_columns([
    pl.col("Variant").str.extract(r"^(\w)", 1).alias("Original_AA"),
    pl.col("Variant").str.extract(r"(\d+)", 1).cast(pl.Int64).alias("Position"),
    pl.col("Variant").str.extract(r"(\w)$", 1).alias("New_AA")
])

human_accessions = hg38_missense_data['Protein_accession'].unique()

# Filter to keep human proteins only based on the AlphaMissesne data and join the pathogenicity scores
human_quasi_primes = chordata_quasi_primes.filter(pl.col('Protein_accession').is_in(human_accessions))
quasi_prime_missense_data = hg38_missense_data.join(human_quasi_primes, on='Protein_accession')

# Calculate Kolmogorov-Smirnov for distribution comparison and calculate Cliff's delta to calculate the magnitude of change
mutations_inside_quasi_prime_regions = quasi_prime_missense_data.filter(
    (pl.col("Position") >= pl.col("Match_start")) & 
    (pl.col("Position") <= pl.col("Match_end"))
)

mutations_outside_quasi_prime_regions = quasi_prime_missense_data.filter(
    (pl.col("Position") < pl.col("Match_start")) | 
    (pl.col("Position") > pl.col("Match_end"))
)

inside_scores = mutations_inside_quasi_prime_regions['Score'].to_list()
outside_scores = mutations_outside_quasi_prime_regions['Score'].to_list()

ks_stat, ks_p_value = stats.ks_2samp(inside_scores, outside_scores)
print(f"K-S statistic: {ks_stat:.2f}, p-value: {ks_p_value}")

def cliffs_delta(x, y):
    """
    Compute Cliff's delta, a non-parametric effect size measure, using 
    a method that is much more efficient than O(n*m) pairwise comparisons.

    Cliff’s delta is defined as:
       δ = (#(x > y) - #(x < y)) / (nx * ny)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    nx, ny = len(x_sorted), len(y_sorted)
    
    # less_counts[i]: how many values in y are strictly less than x_sorted[i]
    less_counts = np.searchsorted(y_sorted, x_sorted, side='left')
    # greater_counts[i]: how many values in y are strictly greater than x_sorted[i]
    greater_counts = ny - np.searchsorted(y_sorted, x_sorted, side='right')
    
    x_greater_than_y = np.sum(less_counts)      # sum of counts of y < x[i]
    x_less_than_y = np.sum(greater_counts)      # sum of counts of y > x[i]
    
    delta = (x_greater_than_y - x_less_than_y) / (nx * ny)
    return delta

delta = cliffs_delta(inside_scores, outside_scores)
print(f"Cliff's delta: {delta:.3f}")

# Calculate fraction of pathogenic variants in each region
inside_total = len(mutations_inside_quasi_prime_regions)
outside_total = len(mutations_outside_quasi_prime_regions)

inside_pathogenic = mutations_inside_quasi_prime_regions.filter(
    pl.col("Score") >= 0.9
).height

outside_pathogenic = mutations_outside_quasi_prime_regions.filter(
    pl.col("Score") >= 0.9
).height

# Calculate fractions
inside_fraction = inside_pathogenic / inside_total
outside_fraction = outside_pathogenic / outside_total

# Calculate fold enrichment
fold_enrichment = inside_fraction / outside_fraction

print(f"Pathogenic enrichment: {fold_enrichment:.2f}")

def calculate_font_sizes(fig_width: float, 
                         fig_height: float) -> Dict[str, float]:
    
    """
    This is a helper function that will be used on the subsequent plots to dynamically calculate the font size of various elements of the plots.
    """
    
    min_dimension = min(fig_width, fig_height)
    base_size = min_dimension * 2
    return {
        'axis_label': max(base_size * 1.35, 10),
        'tick_label': max(base_size * 1.05, 8),
        'legend': max(base_size * 1.2, 8),
        'metric_value': max(base_size * 1, 8),
        'table': max(base_size * 0.9, 8)
    }
fontsizes = calculate_font_sizes(8,6)

plt.figure(figsize=(8, 6))

sns.kdeplot(data=mutations_inside_quasi_prime_regions.to_pandas(), 
            x='Score', 
            fill=True, 
            label='in_quasi_prime', 
            zorder=2, 
            color='#5460D1',
            common_norm=True,
            clip=(0,1))
sns.kdeplot(data=mutations_outside_quasi_prime_regions.to_pandas(), 
            x='Score', 
            fill=True, 
            label='out_quasi_prime', 
            zorder=2, 
            color='#D1B654',
            common_norm=True,
            clip=(0,1))

# Add rectangles for benign and pathogenic regions
plt.fill_betweenx(y=[0, 7.5], x1=0, x2=0.1, color='#3E9C57', alpha=0.2)
plt.fill_betweenx(y=[0, 7.5], x1=0.9, x2=1.0, color='#D15457', alpha=0.2)

# Adjust tick parameters for both axes
plt.xticks(fontsize=fontsizes['tick_label'])
plt.yticks(fontsize=fontsizes['tick_label'])
plt.xlabel('Pathogenicity score (greater is worse)', fontsize=fontsizes['axis_label'])
plt.ylabel('Density', fontsize=fontsizes['axis_label'])

# Add labels for regions
plt.text(0.95, 7.6, 'Pathogenic', fontsize=fontsizes['metric_value'], color='#D15457', ha='center')
plt.text(0.05, 7.6, 'Benign', fontsize=fontsizes['metric_value'], color='#3E9C57', ha='center')
plt.text(0.5, 7.3, 'K-S: 0.34, δ=0.448 \n p<0.001', fontsize=fontsizes['metric_value'], color='black', ha='center')

plt.ylim(top=8)
plt.tight_layout()
plt.grid(zorder=0, alpha=0.8, linewidth=0.5)
sns.despine()
plt.savefig("pathogenicity.svg")
plt.show()
