import requests
import polars as pl
import pandas as pd
from typing import *
import seaborn as sns
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt 

sns.set_context("paper")

# Base directory for data storage
base_dir = Path("/storage/group/izg5139/default/lefteris/")

# Directory that contains the structural data 
structural_analysis_dir = base_dir / "multi_species_structural_analysis_files/"

# Lists containing taxon ids for model organisms and global health risk organisms present in the dataset
model_organisms_taxon_ids = [
    3702, 6239, 237561, 7955, 44689, 7227, 83333, 3847, 
    9606, 243232, 10090, 39947, 10116, 559292, 284812, 4577,
]

global_health_taxon_ids = [
    447093, 6279, 192222, 86049, 318479, 1352, 1442368, 71421,
    85962, 1125630, 5671, 100816, 272631, 83332, 1299332, 242231, 
    1133849, 6282, 502779, 36329, 208964, 99287, 6183, 300267, 1391915, 
    93061, 171101, 6248, 36087, 185431, 353153, 6293
]

# Define organism categories
organism_categories = {
    "Animals": [
        "Homo sapiens",
        "Caenorhabditis elegans",
        "Drosophila melanogaster",
        "Danio rerio",
        "Mus musculus",
        "Rattus norvegicus"
    ],
    "Plants": [
        "Arabidopsis thaliana",
        "Oryza sativa Japonica Group",
        "Glycine max",
        "Zea mays"
    ],
    "Single-celled": [
        "Saccharomyces cerevisiae S288C",
        "Schizosaccharomyces pombe 972h-"
    ],
    "Protozoan": [
        "Trypanosoma cruzi strain CL Brener",
        "Leishmania infantum",
        "Trypanosoma brucei brucei TREU927",
        "Plasmodium falciparum 3D7"
    ],
    "Helminths": [
        "Brugia malayi",
        "Onchocerca volvulus",
        "Strongyloides stercoralis",
        "Schistosoma mansoni",
        "Dracunculus medinensis",
        "Wuchereria bancrofti"
    ],
    "Fungi": [
        "Cladophialophora carrionii",
        "Sporothrix schenckii ATCC 58251",
        "Madurella mycetomatis",
        "Candida albicans SC5314",
        "Histoplasma capsulatum G186AR",
        "Fonsecaea pedrosoi CBS 271.37",
        "Paracoccidioides lutzii Pb01"
    ],
    "Bacteria": [
        "Campylobacter jejuni ATCC 700819",
        "Helicobacter pylori 26695"
    ]
}

# Read the original file
secondary_structures = pl.read_csv(structural_analysis_dir / "secondary_structures.txt", 
                                   separator ="\t")

# List of columns to process
percentage_cols = [
    "Alpha_percent", "3_10_helix_percent", "Pi_helix_percent",
    "Extended_percent", "Isolated_bridge_percent", "Turn_percent",
    "Coil_percent"
]

# Create expressions to replace "N/A" with "0" for each column
expressions = [
    pl.col(col).map_elements(lambda x: '0' if x == "N/A" else x, return_dtype=pl.Utf8)
    for col in percentage_cols + ["Predominant_structure"]
]
secondary_structures = secondary_structures.with_columns(expressions)

# Replace values in Predominant_structure
secondary_structures = secondary_structures.with_columns(
    pl.col("Predominant_structure").replace('0', "Disordered")
)

# Split PDB_ID and create new columns
secondary_structures = secondary_structures.with_columns([
    pl.col("PDB_ID").str.split('_').list.get(0).alias("Accession"),
    pl.col("PDB_ID").str.split('_').list.get(1).cast(pl.Int64).alias("Start"),
    pl.col("PDB_ID").str.split('_').list.get(2).cast(pl.Int64).alias("End")
])

# Drop PDB_ID column
secondary_structures = secondary_structures.drop("PDB_ID")

secondary_structures = secondary_structures.with_columns([
    pl.col("Alpha_percent").cast(pl.Float64),
    pl.col("3_10_helix_percent").cast(pl.Float64),
    pl.col("Pi_helix_percent").cast(pl.Float64),
    pl.col("Extended_percent").cast(pl.Float64),
    pl.col("Isolated_bridge_percent").cast(pl.Float64),
    pl.col("Turn_percent").cast(pl.Float64),
    pl.col("Coil_percent").cast(pl.Float64),
])

# Add structure percenteges to the disordered column
secondary_structures = secondary_structures.with_columns([
    pl.when(pl.col("Predominant_structure") == "Disordered")
    .then(100.0)
    .otherwise(0.0)
    .alias("Disordered_percent")
])

# Reorder columns with select
secondary_structures = secondary_structures.select(
    ["Accession", "Start", "End"] + percentage_cols + ["Disordered_percent", "Predominant_structure"]
)

# Read the data to map UniProt accession numbers to taxon ids
accession_taxon_mappings = pl.read_csv(base_dir / "peptide_match_files/uniprot_2024_01_reference_proteomes/reference_proteomes_2024_01_taxid_mappings.idmapping", 
                                       separator='\t',
                                       new_columns = ["Accession", "Database", "Taxon_ID"]).drop("Database")

# Add taxon ids to the accessions
sec_structures_with_mappings = secondary_structures.join(
    accession_taxon_mappings,
    on="Accession",
    how="left"
)

# Get unique Taxon_IDs 
unique_taxon_ids = sec_structures_with_mappings["Taxon_ID"].unique().to_list()

# Initialize dictionary for results
taxon_names = {}

# Search for scientific names on the taxon ids from the E-utilities of NCBI
base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
batch_size = 50

# Process IDs in batches
for i in range(0, len(unique_taxon_ids), batch_size):
    batch = unique_taxon_ids[i:i + batch_size]
    id_string = ",".join(map(str, batch))
    
    # Construct the E-utilities URL
    url = f"{base_url}/esummary.fcgi?db=taxonomy&id={id_string}&retmode=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # Extract scientific names from the response
        for taxon_id in batch:
            try:
                taxon_data = data["result"][str(taxon_id)]
                scientific_name = taxon_data["scientificname"]
                taxon_names[taxon_id] = scientific_name
            except KeyError:
                taxon_names[taxon_id] = "Not found"
                
    except requests.exceptions.RequestException as e:
        print(f"Error fetching batch {i//batch_size + 1}: {str(e)}")
        

# Convert dictionary to DataFrame and join
names_df = pl.DataFrame({
    "Taxon_ID": list(taxon_names.keys()),
    "Scientific_Name": list(taxon_names.values())
})

# Join scientific names to sec_structures_with_mappings
sec_structures_with_mappings = sec_structures_with_mappings.join(
    names_df,
    on="Taxon_ID",
    how="left"
)

# Replace the specific strain name
sec_structures_with_mappings = sec_structures_with_mappings.with_columns(
    pl.col("Scientific_Name").map_elements(
        lambda x: "Campylobacter jejuni ATCC 700819" 
        if x == "Campylobacter jejuni subsp. jejuni NCTC 11168 = ATCC 700819" 
        else x,
        return_dtype = pl.Utf8
    )
)

# Split original DataFrame to model organisms and global health risk organisms
model_organisms_df = sec_structures_with_mappings.filter(pl.col("Taxon_ID").is_in(model_organisms_taxon_ids))
global_health_df = sec_structures_with_mappings.filter(pl.col("Taxon_ID").is_in(global_health_taxon_ids))

def secondary_structure_kruskal_test(df):
    """
    Perform Kruskal-Wallis test on secondary structure percentage columns.
    """
    
    # Get all percentage columns
    percent_columns = [
        'Alpha_percent', '3_10_helix_percent', 'Pi_helix_percent', 
        'Extended_percent', 'Isolated_bridge_percent', 'Turn_percent', 
        'Coil_percent', 'Disordered_percent'
    ]
    
    # Filter out columns where all values are 0
    non_zero_cols = [
        col for col in percent_columns 
        if df.select(pl.col(col)).sum().item() > 0
    ]
    
    # Convert each non-zero column to a list for the Kruskal-Wallis test
    data_for_test = [
        df.select(pl.col(col)).to_series().to_list()
        for col in non_zero_cols
    ]
    
    # Perform Kruskal-Wallis H-test
    h_statistic, p_value = stats.kruskal(*data_for_test)
    
    return {
        'h_statistic': h_statistic,
        'df': len(non_zero_cols) - 1,
        'p_value': p_value,
        'n': len(df),
        'groups_tested': non_zero_cols
    }

results = secondary_structure_kruskal_test(global_health_df)

print(f"H = {results['h_statistic']:.2f}, df = {results['df']},"
      f"p < 0.001, n = {results['n']}).")

def create_visualisation_data(structural_dataframe: pl.DataFrame) -> pd.DataFrame:
    """
    Calculates the mean secondary structure percentage for secondary structure type and for each species present in the dataset.
    """
    # Calculate percentage means
    grouped = (structural_dataframe
        .group_by("Scientific_Name")
        .agg([
            pl.mean(col).alias(f"{col}") 
            for col in percentage_cols + ["Disordered_percent"]
        ])

    )

    # Convert Polars dataframe to pandas for later visualisations
    grouped_pd = grouped.to_pandas()

    # Set "Scientific_Name" as the index
    grouped_pd.set_index("Scientific_Name", inplace=True)

    # Select the data for clustering
    data_for_clustering = grouped_pd[percentage_cols + ["Disordered_percent"]]
    data_for_clustering = data_for_clustering.rename(columns={"Alpha_percent": "Alpha_helix_percent"})
    
    # Create a mapping dictionary that maps organism names to their categories
    organism_to_category = {}
    for category, organisms in organism_categories.items():
        for organism in organisms:
            organism_to_category[organism] = category

    # Add new column to the dataframe
    data_for_clustering['Category'] = data_for_clustering.index.map(organism_to_category)

    return data_for_clustering

model_organisms_vis = create_visualisation_data(model_organisms_df)
global_health_vis = create_visualisation_data(global_health_df)

def calculate_font_sizes(fig_width: float, 
                         fig_height: float) -> Dict[str, float]:
    """
    Helper function to dynamically calculate font sizes for plot elements based on figure dimensions.
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

def create_category_legend(category_colors: dict, output_file_path: str) -> None:
    """
    Creates a separate legend figure for the categories.
    """
    
    fig, ax = plt.subplots(figsize=(3, 4))
    
    # Create patches for each category
    patches = []
    for label, color in category_colors.items():
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, label=label)
        patches.append(patch)
    
    # Add the patches to the legend
    ax.legend(patches, 
             category_colors.keys(),
             title='Categories',
             loc='center',
             fontsize=12,
             title_fontsize=14,
             frameon=True)
    
    # Hide the axes
    ax.set_axis_off()
    
    # Save the legend with a tight layout
    plt.savefig(output_file_path, 
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=300,
                transparent=True)
    plt.close()

def create_clustermap(visualisation_data: pd.DataFrame, colorbar: str, output_file_path: str) -> None:
    """
    Creates a clustermap visualisation of the secondary structures for each species,
    with rows colored based on organism categories.
    """
    
    fontsizes = calculate_font_sizes(12,9)
    # Clean up column names
    clean_col_names = [col.replace("_percent", "").replace("_", " ")
                      for col in visualisation_data.columns[:-1]]  # Exclude Category column
    viz_data = visualisation_data.drop('Category', axis=1)  # Remove Category from main data
    viz_data.columns = clean_col_names
    
    # Create a color palette for categories
    category_colors = {
        'Helminths': '#9375E0',
        'Fungi': '#759CE0',
        'Bacteria': '#E0CB74',
        'Animals': '#8B8777',     
        'Plants': '#363221',      
        'Single-celled': '#777E8B',
        'Protozoan': '#7D778B'
    }
    
    # Create row colors based on categories
    row_colors = visualisation_data['Category'].map(category_colors)
    
    # Create the clustermap
    g = sns.clustermap(
        data=viz_data,
        cmap=colorbar,
        metric='euclidean',
        method='ward',
        annot=True,
        fmt=".2f",
        annot_kws={'size': 14},
        xticklabels=clean_col_names,
        yticklabels=True,
        linewidths=0.5,
        linecolor="black",
        figsize=(16, 12),
        dendrogram_ratio=0.2,
        tree_kws={"linewidths": 1.0},
        row_colors=row_colors
    )
    
    # Customize the plot
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    
    # Rotate axis labels
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=fontsizes['tick_label']
    )
    
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        rotation=0,
        va="center",
        fontsize=fontsizes['tick_label']
    )
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the clustermap
    plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # Create and save the separate legend
    legend_file_path = output_file_path.replace('.svg', '_legend.svg')
    create_category_legend(category_colors, legend_file_path)

create_clustermap(model_organisms_vis, "viridis", "model_organisms_sec_structures.svg")
create_clustermap(global_health_vis, "cividis", "global_health_sec_structures.svg")
