import os
from pathlib import Path
import textwrap
from typing import *
import numpy as np
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator
from matplotlib.colorbar import ColorbarBase
from pycirclize import Circos
from ete3 import NCBITaxa, Tree
import ete3
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
processed_files_dir.mkdir(parents=True, exists_ok=True)

# Enable string cache in Polars for performance improvements
pl.enable_string_cache()

# Set the style for seaborn plots
sns.set_style("ticks",{'font.family':'serif', 'font.serif':'Microsoft Sans Serif'})
plt.style.use('seaborn-v0_8-ticks')
sns.set_context("paper")

def read_quasi_prime_peptide_data(qp_file_path: str, 
               qp_peptide_length: str) -> pl.DataFrame:
    """
    This function is used to read the dataframes containing taxonomic quasi-prime peptides with their ε-score for the Phylum level. 
    It formats the data to a Polars DataFrame and adds a new column specifing the peptide length.
    """
    
    qp_peptides_df= pl.read_csv(
        qp_file_path,
        separator='\t',  
        has_header=False 
    )
    
    # Specify how many peptides will the regex keep downstream
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
        
    # Split into three columns using regex patterns
    qp_peptides_df = qp_peptides_df.with_columns([
        # First 7 characters for code
        pl.col("column_1").str.slice(0, peptide_length).alias("QP_peptide"),

        # Extract phylum (everything between code and percentage)
        pl.col("column_1").str.extract(fr"^.{{{regex_number}}}(.*?)\s+\d+\.?\d*%$").alias("Taxonomy"),

        # Extract percentage (numbers followed by %)
        pl.col("column_1").str.extract(r"(\d+\.?\d*)%$").cast(pl.Float64).alias("Epsilon_score")
    ])

    # Drop the original column
    qp_peptides_df = qp_peptides_df.drop("column_1")
    
    qp_peptides_df = qp_peptides_df.with_columns(
    pl.col("Taxonomy").str.strip_chars_end(" ").alias("Taxonomy")).with_columns(
    (pl.col("Taxonomy").str.slice(0, 1).str.to_uppercase() +
     pl.col("Taxonomy").str.slice(1)).alias("Taxonomy"))
    
    # Adds the length column 
    qp_peptides_df = qp_peptides_df.with_columns(
        pl.lit(qp_peptide_length).alias("QP_peptide_length"))

    # Specifies a memory efficient schema for the dataframe.
    qp_peptides_df = qp_peptides_df.with_columns([
        pl.col("QP_peptide").cast(pl.Utf8),
        pl.col("Taxonomy").cast(pl.Utf8),
        pl.col("Epsilon_score").cast(pl.Float32),
        pl.col("QP_peptide_length").cast(pl.Categorical)
    ])
    return qp_peptides_df

# This is a nested dictionary with the exact mapping of how the data are classifiied in domains, kingdoms and phyla
# The data are ordered as follows:
#     Domain
#         Kingdom
#             Phylum

domain_to_kingdom_to_phylum = {
    'Archaea':{
        'Archaea': [
            'Euryarchaeota', 'Nanoarchaeota', 'Nitrososphaerota', 
            'Thermoproteota', 'Candidatus Woesearchaeota', 'Candidatus Undinarchaeota',
            'Candidatus Thorarchaeota', 'Candidatus Thermoplasmatota', 'Candidatus Parvarchaeota',
            'Candidatus Nanohaloarchaeota', 'Candidatus Micrarchaeota', 'Candidatus Lokiarchaeota',
            'Candidatus Korarchaeota', 'Candidatus Diapherotrites', 'Candidatus Bathyarchaeota',
            'Candidatus Altarchaeota', 'Candidatus Aenigmarchaeota', 'Archaeal Incertae sedis'
        ]
    },
    'Bacteria':{
        'Bacteria':[
            'Abditibacteriota','Acidobacteriota', 'Actinomycetota', 
            'Aquificota', 'Armatimonadota','Atribacterota', 
            'Bacillota', 'Bacteroidota', 'Balneolota', 
            'Bdellovibrionota', 'Caldisericota', 'Calditrichota', 
            'Campylobacterota','Chlorobiota', 'Chloroflexota', 
            'Chlamydiota', 'Chrysiogenota', 'Coprothermobacterota', 
            'Cyanobacteriota', 'Deferribacterota', 'Deinococcota', 
            'Dictyoglomota', 'Elusimicrobiota', 'Fibrobacterota', 
            'Fusobacteriota', 'Gemmatimonadota','Ignavibacteriota', 
            'Kiritimatiellota', 'Lentisphaerota', 'Mycoplasmatota', 
            'Myxococcota', 'Nitrospinota', 'Nitrospirota', 
            'Planctomycetota','Pseudomonadota', 'Rhodothermota', 
            'Spirochaetota', 'Synergistota', 'Thermodesulfobacteriota',
            'Thermotogota', 'Verrucomicrobiota', 'Candidatus Tectomicrobia', 
            'Thermodesulfobiota','Thermomicrobiota', 'Candidatus Saccharibacteria',
            'Candidatus Poribacteria', 'Candidatus Parcubacteria', 'Candidatus Paceibacterota',
            'Candidatus Omnitrophota', 'Candidatus Nomurabacteria', 'Candidatus Moduliflexota',
            'Candidatus Melainabacteria', 'Candidatus Mcinerneyibacteriota', 'Candidatus Marinimicrobia',
            'Candidatus Margulisiibacteriota', 'Candidatus Lithacetigenota', 'Candidatus Latescibacteria',
            'Candidatus Kryptonia', 'Candidatus Kapabacteria', 'Candidatus Hydrogenedentes',
            'Candidatus Gracilibacteria', 'Candidatus Dormibacteraeota', 'Candidatus Cryosericota',
            'Candidatus Cloacimonadota', 'Candidatus Bipolaricaulota', 'Candidatus Absconditabacteria',
            'Candidate division Zixibacteria', 'Candidate division NC10', 'Bacterial Incertae sedis'
        ],
    },
    'Eukaryota':{
        'Fungi':[
            'Ascomycota','Basidiomycota','Blastocladiomycota', 
                 'Chytridiomycota', 'Cryptomycota', 'Mucoromycota', 
                 'Olpidiomycota','Oomycota', 'Zoopagomycota', 
                 'Microsporidia', 'Fungal Incertae sedis'
        ],
        'Metazoa':[
            'Annelida', 'Arthropoda', 'Brachiopoda', 
                   'Bryozoa','Chordata', 'Cnidaria', 
                   'Echinodermata', 'Mollusca','Nematoda', 
                   'Orthonectida', 'Placozoa', 'Platyhelminthes', 
                   'Porifera', 'Rotifera', 'Tardigrada', 'Fornicata'
        ],
        'Protista':[
            'Apicomplexa', 'Bacillariophyta', 'Cercozoa', 
                    'Ciliophora','Discosea', 
                    'Euglenozoa', 'Foraminifera', 'Haptophyta', 
                    'Heterolobosea', 'Parabasalia', 'Perkinsozoa', 
                    'Rhodophyta', 'Endomyxa', 'Evosea', 'Bigyra',
                    'Choanozoa', 'Cryptophyta', 'Heliozoa', 'Loukozoa',
                    'Metamonada', 'Myzozoa', 'Ochrophyta', 'Sulcozoa'
        ],
        'Viridiplantae':[
            'Chlorophyta', 'Streptophyta'
        ]
    },
    'Viruses':{
        'Bamfordvirae':[
            'Nucleocytoviricota','Preplasmiviricota'
        ],
        'Helvetiavirae':[
            'Dividoviricota'
        ],
        'Heunggongvirae':[
            'Peploviricota','Uroviricota'
        ],
        'Loebvirae':[
            'Hofneiviricota'
        ],
        'Orthornavirae':[
            'Duplornaviricota','Kitrinoviricota','Lenarviricota',
                         'Negarnaviricota','Pisuviricota', 'Orthornavirae Incertae sedis'
        ],
        'Pararnavirae':[
            'Artverviricota'
        ],
        'Sangervirae':[
            'Phixviricota'
        ],
        'Shotokuvirae':[
            'Cossaviricota','Cressdnaviricota'
        ],
        'Trapavirae':[
            'Saleviricota'
        ],
        'Zilligvirae':[
            'Taleaviricota'
        ],
        'Viral Incertae sedis':[
            "Viral Incertae sedis"
        ]
    }
}

def sort_dictionary(domain_to_kingdom_to_phylum):
    """
    Sorts a nested dictionary structure containing domains, kingdoms, and phyla.
    """
    sorted_dict = {}
    
    for domain in sorted(domain_to_kingdom_to_phylum.keys()):
        sorted_dict[domain] = {}
        
        # Sort kingdoms within each domain
        for kingdom in sorted(domain_to_kingdom_to_phylum[domain].keys()):
            # Sort phyla within each kingdom
            sorted_phyla = sorted(domain_to_kingdom_to_phylum[domain][kingdom])
            sorted_dict[domain][kingdom] = sorted_phyla
            
    return sorted_dict

domain_to_kingdom_to_phylum = sort_dictionary(domain_to_kingdom_to_phylum)

# Read and format domain data
domains_5mers = read_quasi_prime_peptide_data(config['superkingdom_5mers'], '5mer')
domains_6mers = read_quasi_prime_peptide_data(config['superkingdom_6mers'], '6mer')
domains_7mers = read_quasi_prime_peptide_data(config['superkingdom_7mers'], '7mer')

# Concatenate dataframes of 6mer and 7mer Quasi prime peptides
domain_QPs = pl.concat([domains_6mers, domains_7mers])

# Create separate dataframes for each greater domain
archaea_domain_QPs = domain_QPs.filter(pl.col('Taxonomy') == 'Archaea')
bacteria_domain_QPs = domain_QPs.filter(pl.col('Taxonomy') == 'Bacteria')
eukaryota_domain_QPs = domain_QPs.filter(pl.col('Taxonomy') == 'Eukaryota')
viruses_domain_QPs = domain_QPs.filter(pl.col('Taxonomy') == 'Viruses')

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

# Define the custom color palette for the unique domains to be used in the subsequent plots
domain_palette = {
    'Archaea': '#0072b2',
    'Bacteria': '#e69f00',
    'Eukaryota': '#009e73',
    'Viruses': '#cc79a7'
}

# Set an order for plotting the various domains
domain_order = ['Archaea', 'Bacteria', 'Eukaryota', 'Viruses']

def plot_domain_epsilon_score_density(domain_QPs: pl.DataFrame,
                                      fig_width: float,
                                      fig_height: float,
                                      qp_peptide_length: str, 
                                      separate_legend: bool=True) -> Union[plt.Figure, Tuple[plt.Figure, plt.Figure]]:

    """
    This function creates a Kernel Density Estimate (KDE) plot that shows the distribution of the ε-scores across the various taxonomic domains. 
    Also, plots a dotted line showing the geometric mean ε-score, a solid line showing the max ε-score and a 
    table showing the number of observed taxonomic quasi-prime peptides per domain.
    """

    # Select specific columes and convert to Pandas dataframe so that it will be accepted in Seaborn
    domain_QPs_pandas = domain_QPs.filter(pl.col('QP_peptide_length') == qp_peptide_length).select(['Taxonomy', 'Epsilon_score']).to_pandas()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate font sizes
    font_sizes = calculate_font_sizes(fig_width, fig_height)
    
    # KDE plot function
    sns.kdeplot(data=domain_QPs_pandas, 
                x='Epsilon_score', 
                hue="Taxonomy",
                palette=domain_palette,
                log_scale=True,
                common_norm=False, 
                common_grid=True,
                fill=True,
                alpha=.1,
                linewidth=1,
                ax=ax,
                bw_method='scott',
                legend=False)
    
    # Adjust x-axis right limit to create more space for the max ε-score indicators
    ax.set_xlim(right=400)
    
    # Get current y-axis limits
    min_y, max_y = ax.get_ylim()

    # Adjust y-axis limits to create more space
    ax.set_ylim(min_y, max_y * 1.2)

    # Collect quasi-prime counts, epsilon score geometric means and max values
    quasi_prime_counts = {}
    medians = {} 
    max_scores = {}
   
    # Define y-axis positions for geometric mean and max ε-scores
    median_positions = {
        'Eukaryota': 1.1,
        'Bacteria': 1.05,
        'Archaea': 1.05,
        'Viruses': 1.00
    }
    
    max_score_positions = {
        'Eukaryota': 0.10,
        'Bacteria': 0.20,
        'Archaea': 0.14,
        'Viruses': 0.20
    }

    # Add median and max ε-score indicators
    for taxonomy in domain_order:
        # Extract relevant data
        data = domain_QPs_pandas[domain_QPs_pandas['Taxonomy'] == taxonomy]
        
        # Calculate median and store it
        medians[taxonomy] = data['Epsilon_score'].median()
        
        # Calculate max score and store it
        max_scores[taxonomy] = data['Epsilon_score'].max()
       
        # Count quasi-prime peptides
        quasi_prime_counts[taxonomy] = len(data)
        
        # Assign median values
        median = medians[taxonomy]
        max_score = max_scores[taxonomy]
        
        # Assign positions for the median and max score indicators
        median_pos = median_positions[taxonomy] * max_y
        max_score_pos = max_score_positions[taxonomy] * max_y

        # Draw the median line (top)
        ax.plot([median, median], [min_y, median_pos], color=domain_palette[taxonomy], linestyle='--', linewidth=1.5)

        # Draw the max score line (bottom)
        ax.plot([max_score, max_score], [min_y, max_score_pos], color=domain_palette[taxonomy], linestyle='-', linewidth=1.5)

        # Add median value (top)
        ax.text(median, median_pos, f'$ε_M = {median:.2f}$', 
                color=domain_palette[taxonomy],
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=font_sizes['metric_value'],
                rotation=0,
                weight='bold')

        # Add max score value (bottom)
        ax.text(max_score, max_score_pos, f'$ε_{{max}} = {max_score:.2f}$', 
                color=domain_palette[taxonomy],
                horizontalalignment='center',
                verticalalignment='bottom',
                fontsize=font_sizes['metric_value'],
                rotation=0,
                weight='bold')
        
    # Set labels and title
    ax.set_title(' ')
    ax.set_xlabel('ε-score (%)', fontsize=font_sizes['axis_label'])
    ax.set_ylabel('Density', fontsize=font_sizes['axis_label'])

    # Adjust tick label font size
    ax.tick_params(axis='both', which='major', labelsize=font_sizes['tick_label'])

    # Adjust layout
    plt.tight_layout()

    # Create a table with number of observed quasi-prime peptides
    table_data = [[taxonomy, f"{count:,}"] for taxonomy, count in quasi_prime_counts.items()]
    table = ax.table(cellText=table_data,
                     colLabels=['Domain', 'Quasi-Prime \npeptide count'],
                     cellLoc='center',
                     loc='upper right',
                     bbox=[0.58, 0.5, 0.41, 0.38], 
                     zorder=10)  # Set high zorder to ensure it's on top

    table.auto_set_font_size(False)
    table.set_fontsize(font_sizes['table'])
    table.scale(1, 1.2)

    # Adjust individual column widths
    table.auto_set_column_width([0, 1])

    # Make the table edges visible and cells semi-transparent
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor('black')
        cell.set_facecolor((1, 1, 1, 0.8))  # Semi-transparent white background

    # Save the plot without legend
    sns.despine()
    plt.grid(visible=True, which='major', alpha=0.7)
    plt.savefig(plots_dir / f'domain_qp_peptide_kde_plot_for_{qp_peptide_length}s.svg', format="svg", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Create a separate legend
    if separate_legend:
        # Create a new figure for the legend
        legend_fig, legend_ax = plt.subplots(figsize=(8, 6 * 0.2))

        # Create custom legend elements
        legend_elements = [Patch(facecolor=domain_palette[taxonomy], edgecolor=domain_palette[taxonomy], label=taxonomy) 
                           for taxonomy in domain_order]

        # Add the legend to the new figure
        legend_ax.legend(handles=legend_elements, loc='center', ncol=4, fontsize=font_sizes['legend'])

        # Remove axes from the legend figure
        legend_ax.axis('off')

        # Save the legend as a separate file
        legend_fig.savefig(plots_dir / f'domain_qp_peptide_kde_plot_legend.svg', format="svg", dpi=300, bbox_inches='tight', pad_inches=0)
        
        return fig, legend_fig
    else:
        return fig


kde_5mers = plot_domain_epsilon_score_density(domains_5mers,
                                              fig_width = 8,
                                              fig_height = 6,
                                              qp_peptide_length='5mer', 
                                              separate_legend = False)

# Create KDE plots for 6mer and 7mer taxonomic Quasi-Prime peptides at domain level
kde_6mers, legend_fig = plot_domain_epsilon_score_density(domain_QPs, 
                                                          fig_width = 8,
                                                          fig_height = 6,
                                                          qp_peptide_length='6mer')

kde_7mers = plot_domain_epsilon_score_density(domain_QPs,
                                              fig_width = 8,
                                              fig_height = 6,
                                              qp_peptide_length='7mer', 
                                              separate_legend = False)

# Read and format kingdom data
kingdoms_6mers = read_quasi_prime_peptide_data(config['kingdom_6mers'], '6mer')
kingdoms_7mers = read_quasi_prime_peptide_data(config['kingdom_7mers'], '7mer')

# Concatenate dataframes of 6mer and 7mer Quasi prime peptides
kingdom_QPs = pl.concat([kingdoms_6mers, kingdoms_7mers])

# Create a look-up dictionary that can categorize kingdoms based on the greater domain level
kingdom_to_domain = {kingdom: domain for domain, kingdoms in domain_to_kingdom_to_phylum.items() for kingdom in kingdoms}

# Single-pass categorization
kingdom_QPs = kingdom_QPs.with_columns(
    pl.col('Taxonomy').replace(kingdom_to_domain).alias('Domain')
)

# Organize taxonomy data in alphabetical order based on the domain they belong
kingdom_QPs = kingdom_QPs.sort('Domain')

# Specifies a memory efficient schema for the dataframe.
kingdom_QPs = kingdom_QPs.with_columns([
    pl.col("Taxonomy").cast(pl.Categorical),
    pl.col("Domain").cast(pl.Categorical)
])

# Create separate dataframes with kingdoms that belong to each greater domain
archaea_kingdom_QPs = kingdom_QPs.filter(pl.col('Domain') == 'Archaea')
bacteria_kingdom_QPs = kingdom_QPs.filter(pl.col('Domain') == 'Bacteria')
eukaryota_kingdom_QPs = kingdom_QPs.filter(pl.col('Domain') == 'Eukaryota')
viruses_kingdom_QPs = kingdom_QPs.filter(pl.col('Domain') == 'Viruses')

# Filter for eukaryotic 7ners 
eukaryota_kingdom_7mers = eukaryota_kingdom_QPs.filter(pl.col("QP_peptide_length") == '7mer')

# Get max epsilon per taxonomy
max_eps = eukaryota_kingdom_7mers.group_by('Taxonomy').agg([
    pl.col('Epsilon_score').max().alias('max_epsilon')
])

# Join back and filter for matching max values
statistics_eukaryota = eukaryota_kingdom_7mers.join(max_eps, on='Taxonomy').filter(
    pl.col('Epsilon_score') == pl.col('max_epsilon')
).sort(['Taxonomy', 'QP_peptide'])

# Show the sorted statistics
statistics_eukaryota.sort("Taxonomy", descending=True)

def order_taxonomies(taxonomic_kingdom_quasi_prime_peptides: pl.DataFrame, domain_order: List[str]) -> List[str]:
    
    """
    Returns a list with the order of taxonomies to plot based on the median ε-score of each one.
    """
    
    # Calculate median Epsilon_score for each Domain-Taxonomy combination
    median_epsilon_scores = (
        taxonomic_kingdom_quasi_prime_peptides.group_by(['Domain', 'Taxonomy'])
        .agg(pl.col('Epsilon_score').median().alias('Median_epsilon_score'))
    )

    final_taxonomy_order = []

    # For each domain, get the taxonomies sorted by median ε-score
    for domain in domain_order:
        domain_taxonomies = (
            median_epsilon_scores
            .filter(pl.col('Domain') == domain)
            .sort('Median_epsilon_score')
            .get_column('Taxonomy')
            .to_list()
        )
        final_taxonomy_order.extend(domain_taxonomies)

    return final_taxonomy_order

def format_count(peptide_count: float) -> float:

    """
    This is a helper function that will be later used to format the total number of taxonomic quasi prime peptides, 
    ploted in the count plot to improve visibility
    """
    
    if peptide_count >= 1e9:
        return f'{peptide_count/1e9:.1f}B'
    elif peptide_count >= 1e6:
        return f'{peptide_count/1e6:.1f}M'
    elif peptide_count >= 1e3:
        return f'{peptide_count/1e3:.1f}K'
    else:
        return f'{peptide_count:.0f}'
    
def plot_kingdom_epsilon_scores(kingdom_QPs: pl.DataFrame,
                                fig_width: float,
                                fig_height: float,
                                qp_peptide_length: str) -> plt.Figure:

    """
    This function creates a combined letter-value and count plot that shows the distribution of the ε-scores across the various taxonomic kingdoms 
    and the number of taxonomic quasi-prime peptides each kingdom possesses.
    """
    
    # Filter kingdom quasi-prime peptides based on length and select appropriate colums for plotting 
    kingdom_QPs = kingdom_QPs.filter(pl.col('QP_peptide_length') == qp_peptide_length).select(['Taxonomy', 'Epsilon_score', 'Domain'])
    
    # Order taxonomies (kingdoms) based on median ε-score and per domain
    ordered_kingdoms = order_taxonomies(kingdom_QPs, domain_order)
    
    # Create the figure with two subplots side-by-side, one is a letter-value plot to show ε-score distribution (left) and the other one is a countplot (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': [2, 1]})

    # Calculate font sizes based on figure size
    font_sizes = calculate_font_sizes(fig_width, fig_height)

    # Create letter-value plot
    sns.boxenplot(
        data=kingdom_QPs.to_pandas(),
        y='Taxonomy',
        x='Epsilon_score',
        hue='Domain',
        order=ordered_kingdoms,
        hue_order=domain_order,
        palette=domain_palette,
        legend=False,
        log_scale=(True, False),
        k_depth='trustworthy',
        ax=ax1,
        dodge=False,
        line_kws=dict(linewidth=2, linestyle='-', color="black"),
        orient='h'
    )

    # Calculate max ε-score for each kingdom and store to a dictionary
    max_scores = kingdom_QPs.group_by('Taxonomy').agg(
        pl.col('Epsilon_score').max().alias('Max_Score'),
        pl.col('Domain').first().alias('Domain')
    )
    max_scores_dict = dict(zip(max_scores['Taxonomy'], zip(max_scores['Max_Score'], max_scores['Domain'])))

    # Add max ε-score indicators to letter-value plot
    for i, taxonomy_name in enumerate(ordered_kingdoms):
        max_score, domain = max_scores_dict[taxonomy_name]
        color = domain_palette[domain]
        
        # Add text annotation
        ax1.text(max_score * 1.5, i, f'$ε_{{max}}$\n${max_score:.2f}$', 
                verticalalignment='center', color=color, fontsize=10)
        
        # Add marker
        ax1.plot(max_score, i, marker='D', color=color, markersize=6, 
                markeredgecolor='black', markeredgewidth=0.5)

    # Set labels for letter-value plot
    ax1.set_xlabel('ε-score (%)', fontsize=font_sizes['axis_label'])
    ax1.set_ylabel('')
    ax1.set_xlim(right=800) # Adjust x limit to accomodate the max ε-score indicators
    ax1.tick_params(axis='both', which='major', labelsize=font_sizes['tick_label'])

    # Create count plot
    total_counts = kingdom_QPs.group_by('Taxonomy').agg(
        pl.len().alias('Total_Count'),
        pl.col('Domain').first().alias('Domain')
    )

    sns.barplot(
        data=total_counts.to_pandas(),
        y='Taxonomy',
        x='Total_Count',
        hue='Domain',
        order=ordered_kingdoms,
        hue_order=domain_order,
        palette=domain_palette,
        ax=ax2,
        orient='h',
        dodge=False,
        legend=False
    )

    # Create a dictionary for quick lookup of sort order
    order_dict = {taxonomy: i for i, taxonomy in enumerate(ordered_kingdoms)}

    # Sort the DataFrame
    total_counts = total_counts.sort(
        pl.col("Taxonomy").map_elements((lambda x: order_dict.get(x, len(ordered_kingdoms))), return_dtype=pl.Int64)
    )

    # Add formatted count labels to bars
    for i, count in enumerate(total_counts['Total_Count']):
        ax2.text(count, i, f' {format_count(count)}', 
                 ha='left', va='center', fontsize=font_sizes['metric_value'])

    # Set labels for bar plot
    ax2.set_xlabel('Peptide number', fontsize=font_sizes['axis_label'])
    ax2.set_ylabel('')
    ax2.set_xscale('log')
    ax2.tick_params(axis='both', which='major', labelsize=font_sizes['tick_label'])
    ax2.set_yticklabels([])

    # Increase x-axis limit by 10% to accomodate the count labels
    x_max = ax2.get_xlim()[1]
    ax2.set_xlim(right=x_max*100)

    # Ensure the y-axis limits are the same for both plots
    y_min, y_max = ax1.get_ylim()
    ax2.set_ylim(y_min, y_max)

    # Remove y-axis ticks from the right plot
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False)

    # Adjust layout
    plt.tight_layout()

    # Reduce the space between subplots
    plt.subplots_adjust(wspace=0.05)
    ax1.grid(visible=True, which='major', alpha=0.7)
    ax2.grid(visible=True, which='major', alpha=0.7)
    sns.despine()
    plt.savefig(plots_dir / f'kingdom_qp_peptide_combined_plot_for_{qp_peptide_length}s.svg',format="svg", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    return fig

combined_plot_6mers = plot_kingdom_epsilon_scores(kingdom_QPs, 
                                                  fig_width = 8,
                                                  fig_height = 6,
                                                  qp_peptide_length = '6mer')

combined_plot_7mers = plot_kingdom_epsilon_scores(kingdom_QPs, 
                                                  fig_width = 8,
                                                  fig_height = 6,
                                                  qp_peptide_length = '7mer')

# Read and format phylum data
phyla_6mers = read_quasi_prime_peptide_data(config['phylum_6mers'], '6mer')
phyla_7mers = read_quasi_prime_peptide_data(config['phylum_7mers'], '7mer')

# Concatenate dataframes of 6mer and 7mer Quasi prime peptides
phylum_QPs = pl.concat([phyla_6mers, phyla_7mers])

# Intialize an empty dictionary to store the phylum to domain mappings 
phylum_to_domain = {}

# Create a look-up dictionary that can categorize phyla based on the greater domain level
for domain, kingdoms in domain_to_kingdom_to_phylum.items():
    for kingdom, phyla in kingdoms.items():
        for phylum in phyla:
            phylum_to_domain[phylum] = domain

# Single-pass categorization
phylum_QPs = phylum_QPs.with_columns(
    pl.col('Taxonomy').replace(phylum_to_domain).alias('Domain')
)

# Organize taxonomy data in alphabetical order based on the domain they belong
phylum_QPs = phylum_QPs.sort('Domain')

# Specifies a memory efficient schema for the dataframe.
phylum_QPs = phylum_QPs.with_columns([
    pl.col("Taxonomy").cast(pl.Categorical),
    pl.col("Domain").cast(pl.Categorical)
])

# Create separate dataframes with phyla that belong to each greater domain
archaea_phylum_QPs = phylum_QPs.filter(pl.col('Domain') == 'Archaea')
bacteria_phylum_QPs = phylum_QPs.filter(pl.col('Domain') == 'Bacteria')
eukaryota_phylum_QPs = phylum_QPs.filter(pl.col('Domain') == 'Eukaryota')
viruses_phylum_QPs = phylum_QPs.filter(pl.col('Domain') == 'Viruses')

# Extract Phylum QPs at different amino acid length for each domain level
archaea_phylum_QPs_6mers = archaea_phylum_QPs.filter(pl.col("QP_peptide_length") == "6mer")
archaea_phylum_QPs_7mers = archaea_phylum_QPs.filter(pl.col("QP_peptide_length") == "7mer")

bacterial_phylum_QPs_6mers = bacteria_phylum_QPs.filter(pl.col("QP_peptide_length") == "6mer")
bacterial_phylum_QPs_7mers = bacteria_phylum_QPs.filter(pl.col("QP_peptide_length") == "7mer")

eukaryotic_phylum_QPs_6mers = eukaryota_phylum_QPs.filter(pl.col("QP_peptide_length") == "6mer")
eukaryotic_phylum_QPs_7mers = eukaryota_phylum_QPs.filter(pl.col("QP_peptide_length") == "7mer")

viral_phylum_QPs_6mers = viruses_phylum_QPs.filter(pl.col("QP_peptide_length") == "6mer")
viral_phylum_QPs_7mers = viruses_phylum_QPs.filter(pl.col("QP_peptide_length") == "7mer")

# Initialize NCBI Taxa database
ncbi = NCBITaxa()

# Create phylum to domain mapping from the nested dictionary
phylum_to_domain = {}
for domain, kingdoms in domain_to_kingdom_to_phylum.items():
    for kingdom, phyla in kingdoms.items():
        for phylum in phyla:
            # Convert to underscore format
            phylum_underscore = phylum.replace(' ', '_')
            phylum_to_domain[phylum_underscore] = domain

# Create a name mapping dictionary, since these names have been updated in the NCBI Taxonomy browser
name_mappings = {
    'Archaeal_Incertae_sedis': "Archaea_incertae_sedis",
    'Candidatus_Aenigmarchaeota': 'Candidatus_Aenigmatarchaeota',
    'Candidatus_Diapherotrites': 'Candidatus_Iainarchaeota',
    'Candidatus_Nanohaloarchaeota': "Candidatus_Nanohalarchaeota",
    'Candidatus_Tectomicrobia': 'Candidatus_Tectimicrobiota',
    'Candidatus_Marinimicrobia': 'Candidatus_Neomarinimicrobiota',
    'Candidatus_Latescibacteria': 'Candidatus_Latescibacterota',
    'Candidatus_Kryptonia': 'Candidatus_Kryptoniota',
    'Candidatus_Kapabacteria': 'Candidatus_Kapaibacteriota',
    'Candidatus_Hydrogenedentes': 'Candidatus_Hydrogenedentota',
    'Candidatus_Dormibacteraeota': 'Candidatus_Dormiibacterota',
    'Candidate_division_Zixibacteria': 'Candidatus_Zixiibacteriota',
    'Candidate_division_NC10': 'Candidatus_Methylomirabilota',
    'Bacterial_Incertae_sedis': 'Bacteria_incertae_sedis',
    'Fungal_Incertae_sedis': 'Fungi_incertae_sedis',
    'Choanozoa': 'Choanoflagellata',
    'Cryptophyta': 'Cryptophyceae',
    'Orthornavirae_Incertae_sedis': 'unclassified_Orthornavirae',
    'Viral_Incertae_sedis': 'unclassified_viruses'
}

# Additional phyla to be added manually
additional_phyla = ['Heliozoa', 'Loukozoa', 'Myzozoa', 'Sulcozoa']

# Extract all phyla from the nested dictionary
phyla_names = []
for domain in domain_to_kingdom_to_phylum.values():
    for kingdom in domain.values():
        phyla_names.extend(kingdom)

# Replace spaces with underscores in phyla names
phyla_names = [name.replace(' ', '_') for name in phyla_names]

# Create mappings
taxid_to_original_name = {}
taxid_list = []

# Get taxonomy IDs and store original name mapping
for name in phyla_names:
        
    # Apply name mapping if it exists
    search_name = name_mappings.get(name, name)
    # Convert back to space-separated for NCBI search
    search_name_spaced = search_name.replace('_', ' ')
    
    try:
        taxid = ncbi.get_name_translator([search_name_spaced])[search_name_spaced][0]
        taxid_list.append(taxid)
        taxid_to_original_name[taxid] = name  
    except KeyError:
        print(f"Phylum '{name}' (search name: '{search_name_spaced}') not found in NCBI Taxonomy Database.")

# Build the phylogenetic tree from Taxonomy IDs
tree = ncbi.get_topology(taxid_list)

# Get names for all nodes, including internal ones
all_taxids = []
for node in tree.traverse():
    if node.name.isdigit():
        all_taxids.append(int(node.name))

# Get scientific names for all taxids
names_dict = ncbi.get_taxid_translator(all_taxids)

# Convert all taxids to names and replace spaces with underscores
for node in tree.traverse():
    if node.name.isdigit():
        taxid = int(node.name)
        if taxid in taxid_to_original_name:
            node.name = taxid_to_original_name[taxid]
        else:
            node.name = names_dict.get(taxid, str(taxid)).replace(' ', '_')
    # Add domain information as a feature
    if not node.name.isdigit():
        domain = phylum_to_domain.get(node.name, 'Unknown')
        node.add_features(domain=domain)

# Find the Alveolata node
alveolata_node = None
for node in tree.traverse():
    if node.name == "Alveolata":
        alveolata_node = node
        break

if alveolata_node:
    # Add the additional phyla to the existing Alveolata clade
    for phylum in additional_phyla:
        new_node = alveolata_node.add_child(name=phylum.replace(' ', '_'))
        domain = phylum_to_domain.get(phylum.replace(' ', '_'), 'Eukaryota')
        new_node.add_features(domain=domain)
else:
    print("Could not find Alveolata clade in the tree")

# Display the tree in ASCII format with all names
# print("\nTree structure:")
# print(tree.get_ascii(show_internal=True))

# Save the tree in Newick format with domain features
output_tree_file = processed_files_dir / "complete_phylogenetic_tree.nwk"
tree.write(outfile=output_tree_file, format=1, features=['domain'])

def calculate_summary_statistics(phylum_QPs: pl.DataFrame) -> pd.DataFrame:

    """
    This is a helper function, which creates a Pandas DataFrame with taxonomic Quasi-prime peptide 6mer count, 
    7mer count and median values for the ε-scores that correspond to the 6mer and 7mer peptides respectively
    """
    
    summary_statistics = (
        phylum_QPs.group_by("Taxonomy")
        .agg([
            pl.col("QP_peptide_length").filter(pl.col("QP_peptide_length") == "6mer").len().alias("6mer_count"),
            pl.col("Epsilon_score").filter(pl.col("QP_peptide_length") == "6mer").median().alias("6mer_median_epsilon"),
            pl.col("QP_peptide_length").filter(pl.col("QP_peptide_length") == "7mer").len().alias("7mer_count"),
            pl.col("Epsilon_score").filter(pl.col("QP_peptide_length") == "7mer").median().alias("7mer_median_epsilon")
        ])
        .sort("Taxonomy"))

    summary_statistics = summary_statistics.fill_null(strategy="zero")
    summary_statistics_pandas = summary_statistics.to_pandas().set_index('Taxonomy')
    # Replace spaces with underscores in index labels
    summary_statistics_pandas.index = summary_statistics_pandas.index.str.replace(' ', '_')
    
    return summary_statistics_pandas

# Generate summary statistics
domain_summary_statistics = calculate_summary_statistics(domain_QPs)
kingdom_summary_statistics = calculate_summary_statistics(kingdom_QPs)
phylum_summary_statistics = calculate_summary_statistics(phylum_QPs)

domain_summary_statistics.to_csv(processed_files_dir / 'domains_summary.csv')
kingdom_summary_statistics.to_csv(processed_files_dir / 'kingdoms_summary.csv')
phylum_summary_statistics.to_csv(processed_files_dir / 'phyla_summary.csv')

def format_label(t):
    # Replace underscores with spaces
    t = t.replace("_", " ")
    # Replace Candidatus/Candidate with C.
    t = t.replace("Candidatus ", "C. ")
    t = t.replace("Candidate ", "C. ")
    # Replace Incertae sedis with I.S.
    t = t.replace("Incertae sedis", "I.S.")
    return t

# First, modify the phylum names in the summary statistics DataFrame
phylum_summary_statistics = phylum_summary_statistics.copy()
phylum_summary_statistics.index = phylum_summary_statistics.index.map(
    lambda x: x.replace("Candidatus ", "C. ")
             .replace("Candidate ", "C. ")
             .replace("Incertae sedis", "I.S.")
)

# Create the phylogenetic tree plot
circos, tv = Circos.initialize_from_tree(
    output_tree_file,
    outer=True,
    start=10,
    end=350,
    r_lim=(22, 60),
    leaf_label_rmargin=40,
    ignore_branch_length=True,
    align_leaf_label=True,
    line_kws=dict(lw=2),
    leaf_label_size=20,
    label_formatter=lambda t: format_label(t)
)

# Highlight phyla branches based on the greater domain they belong to
for domain, color in domain_palette.items():
    tv.highlight([f"{domain}"], color=color)

# Get the order of phyla leaves and order summary statistics based on that to plot
leaf_order = tv.leaf_labels
phyla_positions = np.arange(0, tv.leaf_num) + 0.5

# Now use the fixed names to index the DataFrame
phylum_summary_statistics = phylum_summary_statistics.loc[leaf_order]

sector = tv.track.parent_sector

# Add 6mer heatmap track
heatmap_6mer = sector.add_track((60, 65))
heatmap_6mer_data = phylum_summary_statistics['6mer_median_epsilon'].to_numpy()
heatmap_6mer_data_log = np.log10(heatmap_6mer_data + 1)
heatmap_6mer.heatmap(heatmap_6mer_data_log, cmap="BuPu", show_value=False, rect_kws=dict(ec="black", lw=0.5))

# Calculate min and max values for 6mer heatmap (excluding zero values)
heatmap_6mer_data_min = np.min(heatmap_6mer_data[heatmap_6mer_data > 0]) 
heatmap_6mer_data_max = np.max(heatmap_6mer_data) 

# Add 6mer count plot track
countplot_6mer = sector.add_track((65, 80), r_pad_ratio=0.1)
countplot_6mer.axis()
countplot_6mer.yticks(np.log10([1e0, 1e1, 1e2, 1e3, 1e4]), ['$10^{{0}}$', '$10^{{1}}$', '$10^{{2}}$', '$10^{{3}}$', '$10^{{4}}$'])
countplot_6mer.grid(y_grid_num = 5, x_grid_interval=0.5)
countplot_6mer_data = phylum_summary_statistics['6mer_count'].to_numpy()
countplot_6mer_data_log = np.log10(countplot_6mer_data + 1)
countplot_6mer.bar(phyla_positions, countplot_6mer_data_log, width=0.3, color="#46327e")

# Add 7mer heatmap track
heatmap_7mer = sector.add_track((80, 85))
heatmap_7mer_data = phylum_summary_statistics['7mer_median_epsilon'].to_numpy()
heatmap_7mer_data_log = np.log10(heatmap_7mer_data + 1)
heatmap_7mer.heatmap(heatmap_7mer_data_log, cmap="Blues", show_value=False, rect_kws=dict(ec="black", lw=0.5))

# Calculate min and max values for 7mer heatmap (excluding zero values)
heatmap_7mer_data_min = np.min(heatmap_7mer_data[heatmap_7mer_data > 0]) 
heatmap_7mer_data_max = np.max(heatmap_7mer_data) 

# Add 7mer count plot track
countplot_7mer = sector.add_track((85, 99), r_pad_ratio=0.1)
countplot_7mer.axis()
countplot_7mer.yticks(np.log10([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]), ['$10^{{0}}$', '$10^{{1}}$', '$10^{{2}}$', '$10^{{3}}$', '$10^{{4}}$', '$10^{{5}}$', '$10^{{6}}$', '$10^{{7}}$'])
countplot_7mer.grid(y_grid_num = 8, x_grid_interval=0.5)
countplot_7mer_data = phylum_summary_statistics['7mer_count'].to_numpy()
countplot_7mer_data_log = np.log10(countplot_7mer_data + 1)
countplot_7mer.bar(phyla_positions, countplot_7mer_data_log, width=0.3, color="#365c8d")

# Add labels to the tracks
circos.text(r"$\epsilon_{M}$ 6mer (%)", r=heatmap_6mer.r_center, color="black", size = 18)
circos.text("Counts\n6mer", r=countplot_6mer.r_center, color="black", size = 18)
circos.text(r"$\epsilon_{M}$ 7mer (%)", r=heatmap_7mer.r_center, color="black", size = 18)
circos.text("Counts\n7mer", r=countplot_7mer.r_center, color="black", size = 18)

# Create the main figure
fig = circos.plotfig(dpi = 600, figsize = (25, 25))

# Create a new axes for the colorbar
cbar_6mer = fig.add_axes([0.475, 0.5, 0.1, 0.01])
cbar_7mer = fig.add_axes([0.475, 0.55, 0.1, 0.01])

# Create the colorbar with logarithmic normalization
norm_6mer = LogNorm(vmin=heatmap_6mer_data_min, vmax=heatmap_6mer_data_max)
norm_7mer = LogNorm(vmin=heatmap_7mer_data_min, vmax=heatmap_7mer_data_max)
cbar_6mer = ColorbarBase(cbar_6mer, cmap=plt.get_cmap("BuPu"), norm=norm_6mer, orientation='horizontal')
cbar_7mer = ColorbarBase(cbar_7mer, cmap=plt.get_cmap("Blues"), norm=norm_7mer, orientation='horizontal')

# Set custom ticks and labels for the colorbars
ticks = [2e-2, 1e-1, 1e0, 1e1, 1e2]
labels = ['$10^{{-2}}$','$10^{{-1}}$', '$10^{{0}}$', '$10^{{1}}$', '$10^{{2}}$']
cbar_6mer.set_ticks(ticks)
cbar_6mer.set_ticklabels(labels)
cbar_7mer.set_ticks(ticks)
cbar_7mer.set_ticklabels(labels)

# Set tick label properties for the colorbars
cbar_6mer.ax.tick_params(labelsize=10, colors="black")
cbar_7mer.ax.tick_params(labelsize=10, colors="black")

# Add titles to the colorbars
cbar_6mer.ax.set_title('6mer ε-score (%)', fontsize=18, pad=10)
cbar_7mer.ax.set_title('7mer ε-score (%)', fontsize=18, pad=10)

# Save the figure as a SVG file
plt.savefig(plots_dir / 'phylum_circos_plot.svg', format="svg", dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

# Create a dictionary with superkingdom representative phyla to showcase as much of the superkingdoms diversity
representative_phyla = [
    'Euryarchaeota', 'Thermoproteota', 'Nitrososphaerota',
    'Nanoarchaeota', 'Candidatus Woesearchaeota', 'Candidatus Lokiarchaeota',
    'Candidatus Bathyarchaeota', 'Candidatus Korarchaeota','Acidobacteriota',
    'Myxococcota', 'Mucoromycota', 'Bacillariophyta',
    'Rhodophyta', 'Foraminifera', 'Pseudomonadota',
    'Bacillota', 'Actinomycetota', 'Bacteroidota',
    'Cyanobacteriota', 'Chlamydiota', 'Spirochaetota',
    'Chloroflexota', 'Planctomycetota', 'Deinococcota',
    'Ascomycota', 'Basidiomycota', 'Chordata',
    'Arthropoda', 'Mollusca', 'Nematoda',
    'Streptophyta', 'Chlorophyta', 'Apicomplexa',
    'Ciliophora', 'Nucleocytoviricota', 'Negarnaviricota',
    'Pisuviricota', 'Duplornaviricota', 'Kitrinoviricota',
    'Lenarviricota', 'Artverviricota','Cressdnaviricota',
    'Phixviricota', 'Peploviricota'
]

def plot_phylum_epsilon_score_distribution(phylum_QPs: pl.DataFrame,
                                fig_width: float,
                                fig_height: float,
                                qp_peptide_length: str) -> plt.Figure:

    """
    This function creates a letter-value that shows the distribution of the ε-scores across the various taxonomic phyla
    """    

    # Filter phylum quasi-prime peptides based on length and select appropriate columns for plotting 
    phylum_QPs = phylum_QPs.filter((pl.col('QP_peptide_length') == qp_peptide_length) & (pl.col('Taxonomy').is_in(representative_phyla))).select(['Taxonomy', 'Epsilon_score', 'Domain'])
    
    # Order taxonomies (phyla) based on median ε-score and per domain
    ordered_phyla = order_taxonomies(phylum_QPs, domain_order)
    
    # Create the figure with a single subplot for the boxenplot
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Calculate font sizes based on figure size
    font_sizes = calculate_font_sizes(10, 10)

    # Create letter-value plot (boxenplot)
    sns.boxenplot(
        data=phylum_QPs.to_pandas(),
        y='Taxonomy',
        x='Epsilon_score',
        hue='Domain',
        order=ordered_phyla,
        hue_order=domain_order,
        palette=domain_palette,
        legend=False,
        log_scale=(True, False),
        k_depth='trustworthy',
        ax=ax,
        dodge=False,
        line_kws=dict(linewidth=2, linestyle='-', color="black"),
        orient='h'
    )

    # Calculate max ε-score for each phylum and store to a dictionary
    max_scores = phylum_QPs.group_by('Taxonomy').agg(
        pl.col('Epsilon_score').max().alias('Max_Score'),
        pl.col('Domain').first().alias('Domain')
    )
    max_scores_dict = dict(zip(max_scores['Taxonomy'], zip(max_scores['Max_Score'], max_scores['Domain'])))

    # Add max ε-score indicators to letter-value plot
    for i, taxonomy_name in enumerate(ordered_phyla):
        max_score, domain = max_scores_dict[taxonomy_name]
        color = domain_palette[domain]
        
        # Add text annotation
        ax.text(max_score * 1.5, i, f'$ε_{{max}}: {max_score:.2f}$', 
                verticalalignment='center', color=color, fontsize=18)
        
        # Add marker
        ax.plot(max_score, i, marker='D', color=color, markersize=6, 
                markeredgecolor='black', markeredgewidth=0.5)

    # Set labels for letter-value plot 
    ax.set_xlabel('ε-score (%)', fontsize=font_sizes['axis_label'])
    ax.set_ylabel(' ')
    ax.set_ylim(bottom=len(ordered_phyla), top=-1) # Adjust y limit to remove extra space on the top and bottom
    ax.set_xlim(right=2800)  # Adjust x limit to accommodate the max ε-score indicators
    ax.tick_params(axis='x', which='major', labelsize=font_sizes['tick_label'])
    
    # Update y-axis tick labels
    y_ticks = range(len(ax.get_yticklabels()))
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    y_labels = [label.replace("Candidatus ", "C. ").replace("Candidate ", "C. ") for label in y_labels]
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.set_yticklabels(y_labels, fontsize=font_sizes['tick_label'])
    
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.grid(visible=True, which='major', alpha=0.7)
    sns.despine()
    plt.savefig(plots_dir / f'phylum_qp_peptide_boxenplot_for_{qp_peptide_length}s.svg', format="svg", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    return fig

phylum_boxenplot_6mers = plot_phylum_epsilon_score_distribution(phylum_QPs, 
                                              fig_width=9,
                                              fig_height=16,
                                              qp_peptide_length='6mer')

phylum_boxenplot_7mers = plot_phylum_epsilon_score_distribution(phylum_QPs, 
                                              fig_width=9,
                                              fig_height=16,
                                              qp_peptide_length='7mer')

# Create a dictionary to store information about each taxonomy level
taxonomy_dict = {
    'Archaea': {
        'Domain_QPs': archaea_domain_QPs,
        'Kingdom_QPs': archaea_kingdom_QPs,
        'Phylum_QPs': archaea_phylum_QPs
    },
    'Bacteria': {
        'Domain_QPs': bacteria_domain_QPs,
        'Kingdom_QPs': bacteria_kingdom_QPs,
        'Phylum_QPs': bacteria_phylum_QPs
    },
    'Eukaryota': {
        'Domain_QPs': eukaryota_domain_QPs,
        'Kingdom_QPs': eukaryota_kingdom_QPs,  
        'Phylum_QPs': eukaryota_phylum_QPs
    },
    'Viruses': {

        'Domain_QPs': viruses_domain_QPs,
        'Kingdom_QPs': viruses_kingdom_QPs,
        'Phylum_QPs': viruses_phylum_QPs
    }
}

def filter_dataframe(qp_peptides_df: pl.DataFrame) -> pl.DataFrame:

    """Helper function to filter dataframe to keep only quasi-prime peptides with ε-score over 90%"""
    
    return qp_peptides_df.filter(pl.col('Epsilon_score') >= 90)

e_score_over_90_taxonomy_dict = {}

for domain, data in taxonomy_dict.items():
    e_score_over_90_taxonomy_dict[domain] = {}
    for level in ['Domain_QPs', 'Kingdom_QPs', 'Phylum_QPs']:
        original_qp_peptides_df = data[level]
        filtered_qp_peptides_df = filter_dataframe(original_qp_peptides_df)
        new_var_name = f"{level}_filtered"
        e_score_over_90_taxonomy_dict[domain][new_var_name] = filtered_qp_peptides_df
        
phylum_qps_over_90_list = [] 

# Iterate through the dictionary and collect all 'Phylum_QPs_filtered' dataframes
for domain, data in e_score_over_90_taxonomy_dict.items():
    if 'Phylum_QPs_filtered' in data:
        # Filter dataframes to keep only 7mer Quasi-prime peptides
        phylum_df = data['Phylum_QPs_filtered']
        phylum_df_7mer = phylum_df.filter(pl.col('QP_peptide_length') == '7mer')
        phylum_qps_over_90_list.append(phylum_df_7mer)
        
# Concatenate all the collected dataframes
phylum_qps_over_90_df = pl.concat(phylum_qps_over_90_list)

# Group by Taxonomy and collect unique peptides
grouped_phyla = phylum_qps_over_90_df.group_by('Taxonomy').agg(pl.col('QP_peptide').unique())

output_dir_peptides_over_90 = processed_files_dir / "qp_peptides_over_90_per_phylum"
# Create a directory to store the output files
output_dir_peptides_over_90.mkdir(parents=True, exists_ok=True)

# Iterate through each taxonomy and write peptides to files
for row in grouped_phyla.iter_rows():
    taxonomy, peptides = row[0], row[1]
    
    # Create a valid filename
    filename = f"{taxonomy.replace(' ', '_').replace('/', '_')}.txt"
    # Use the path from the config file here
    filepath = output_dir_peptides_over_90 / filename
    
    # Write peptides to file
    with open(filepath, 'w') as f:
        for peptide in peptides:
            f.write(f"{peptide}\n")