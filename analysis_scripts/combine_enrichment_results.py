import os 
import glob
import numpy as np
import polars as pl
from typing import *
from pathlib import Path
from scipy.stats import t
from numpy.typing import NDArray
from statsmodels.stats.multitest import multipletests
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


# Set input and output directories
mspeea_results_dir = Path(config['protein_entry_output_dir']) / "multi_species_entry_enrichment_results"
msgoea_results_dir = Path(config['multi_species_goea_files_dir']) / "multi_species_goea_results"

entry_results_dir = {
    "domains": mspeea_results_dir / 'domain_enrichment_results',
    "families": mspeea_results_dir / 'family_enrichment_results'
}

output_dirs = {
    "combined_domains": mspeea_results_dir / 'combined_domain_enrichment_results',
    "combined_families": mspeea_results_dir / 'combined_family_enrichment_results',
    "combined_go_terms": msgoea_results_dir / 'combined_gene_ontology_results'
}

# Create all output directories
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

def check_taxon_count(df: pl.DataFrame) -> bool:
    """
    Check how many unique taxon IDs are in the phylum DataFrame.
    """
    if df.is_empty():
        return False 
        
    unique_taxa_count = df['Taxon_ID'].n_unique()
    
    # Returns a boolean True or False if the taxon count is equal to 1
    return unique_taxa_count == 1

def process_single_taxon_data(enrichment_df: pl.DataFrame) -> pl.DataFrame:
    """
    Process data when there's only one taxon ID.
    Adds percentage calculation but skips random effects model, for single taxon-data.
    """
    # Filter significant entries based on Odds Ratio
    significant_df = enrichment_df.filter(pl.col('Odds_Ratio') >= 1)
    
    # Add Species_Count and Species_Percentage columns (will be 1 and 100 respectively)
    result_df = significant_df.with_columns([
        pl.lit(1).alias('Species_Count'),
        pl.lit(100.0).alias('Species_Percentage')
    ])
    
    # Group by Entry and create lists of values while preserving necessary columns
    grouped_df = result_df.group_by('Entry').agg([
        pl.col('Interpro_ID').first().alias('Interpro_ID'),
        pl.col('Log_Odds_Ratio').alias('Log_Odds_Ratio_List'),
        pl.col('SE_Log_Odds_Ratio').alias('SE_Log_Odds_Ratio_List'),
        pl.col('P_Value').first().alias('P_Value'),
        pl.col('Odds_Ratio').first().alias('Odds_Ratio')
    ])
    
    # Select and order columns to match the multi-species output format
    final_df = grouped_df.select([
        'Entry',
        'Interpro_ID',
        pl.lit(1).alias('Species_Count'),
        pl.lit(100.0).alias('Species_Percentage'),
        pl.col('Log_Odds_Ratio_List').list.first().alias('Combined_Log_Odds_Ratio'),
        pl.col('SE_Log_Odds_Ratio_List').list.first().alias('Combined_SE'),
        pl.col('P_Value'),
        pl.col('P_Value').alias('Adjusted_Meta_P_Value'),
        'Log_Odds_Ratio_List',
        'SE_Log_Odds_Ratio_List'
        
    ])
    
    return final_df.sort('Combined_Log_Odds_Ratio', descending=True)

def preprocess_and_calculate_statistics(enrichment_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, List[str]]:
    """
    Preprocess data, filter significant entries, and calculate species statistics for all entries.
    Also identifies multi-species entries.
    """
    # Replace zero p-values with a very small non-zero value
    enrichment_df = enrichment_df.with_columns(
        pl.when(pl.col("P_Value") == 0)
        .then(np.nextafter(0, 1))
        .otherwise(pl.col("P_Value"))
        .alias("P_Value")
    )
    
    # Filter for significant entries based on Odds Ratio
    significant_entries_df = enrichment_df.filter(pl.col('Odds_Ratio') >= 1)
    
    # Calculate total species count
    total_species_count = significant_entries_df['Taxon_ID'].n_unique()
    
    # Calculate species counts and percentages for all entries
    species_statistics = (
        significant_entries_df.group_by('Entry')
        .agg([
            pl.col('Interpro_ID').first().alias('Interpro_ID'),
            pl.col('Taxon_ID').n_unique().alias('Species_Count'),
            pl.col('Log_Odds_Ratio').alias('Log_Odds_Ratio_List'),
             pl.col('SE_Log_Odds_Ratio').alias('SE_Log_Odds_Ratio_List')
        ])
        .with_columns(
            (pl.col('Species_Count') / total_species_count * 100).alias('Species_Percentage')
        )
        .sort('Species_Percentage', descending=True)
    )
        
    return significant_entries_df.join(
        species_statistics.select([
            'Entry',
            'Interpro_ID',
            'Species_Count', 
            'Species_Percentage', 
            'Log_Odds_Ratio_List',
            'SE_Log_Odds_Ratio_List' 
        ]), 
        on='Entry'
    )

def random_effects_meta_analysis(interpro_id,
                                 log_odds_ratios: Union[List[float], NDArray[np.float64]], 
                                 se_log_odds_ratios: Union[List[float], NDArray[np.float64]], 
                                 original_p_value: float,
                                 species_count: int,
                                 species_percentage: int) -> Dict[str, Union[float, int]]:
    """
    Perform random effects meta-analysis on log odds ratios.
    """
    num_entries = len(log_odds_ratios)
    if num_entries <= 1:
        return {
            'Interpro_ID': interpro_id,
            'Combined_Log_Odds_Ratio': log_odds_ratios[0],
            'Combined_SE': se_log_odds_ratios[0],
#             'Tau_Squared': float('nan'),
#             'Q_Statistic': float('nan'),
#             'I_Squared': float('nan'),
            'P_Value': float(original_p_value),
            'Species_Count': species_count,
            'Species_Percentage': species_percentage
        }
    
    weights = 1 / np.square(se_log_odds_ratios)
    weighted_mean = np.average(log_odds_ratios, weights=weights)
    Q_statistic = np.sum(weights * np.square(log_odds_ratios - weighted_mean))
    degrees_freedom = len(log_odds_ratios) - 1
    C_value = np.sum(weights) - (np.sum(np.square(weights)) / np.sum(weights))
    tau_squared = max(0, (Q_statistic - degrees_freedom) / C_value)
    random_weights = 1 / (np.square(se_log_odds_ratios) + tau_squared)
    combined_mean = np.average(log_odds_ratios, weights=random_weights)
    combined_se = np.sqrt(1 / np.sum(random_weights))
    z_score = combined_mean / combined_se
    
    # Use t-distribution to calculate p-value, to better account for the small sizes of samples
    p_value = 2 * (1 - t.cdf(abs(z_score), df=num_entries - 1))
    return {
        'Interpro_ID':interpro_id,
        'Combined_Log_Odds_Ratio': float(combined_mean),
        'Combined_SE': float(combined_se),
        # Uncomment these lines to get heterogeneity data
        #'Tau_Squared': float(tau_squared),
        #'Q_Statistic': float(Q_statistic),
        #'I_Squared': float(max(0, (Q_statistic - degrees_freedom) / Q_statistic * 100) if Q_statistic > 0 else 0),
        'P_Value': float(p_value),
        'Species_Count': species_count,
        'Species_Percentage': species_percentage
    }

def apply_meta_analysis_and_stats(multi_species_data: pl.DataFrame) -> pl.DataFrame:
    """
    Apply random effects meta-analysis and compute log odds ratio statistics.
    """
    return (
        multi_species_data.group_by('Entry')
        .agg(
            [
                pl.col('Interpro_ID').first().alias('Interpro_ID'),
                pl.col('Log_Odds_Ratio').alias('Log_Odds_Ratio_List'),
                pl.col('SE_Log_Odds_Ratio').alias('SE_Log_Odds_Ratio_List'),
                pl.col('P_Value').first().alias('Original_P_Value'),
                pl.col('Species_Count').first().alias('Species_Count'),
                pl.col('Species_Percentage').first().alias('Species_Percentage')
                
            ]
        )
        .with_columns([
            pl.struct([
                'Interpro_ID',
                'Log_Odds_Ratio_List', 
                'SE_Log_Odds_Ratio_List', 
                'Original_P_Value',
                'Species_Count',
                'Species_Percentage'
            ]).map_elements(
                lambda s: random_effects_meta_analysis(
                    s['Interpro_ID'],
                    s['Log_Odds_Ratio_List'], 
                    s['SE_Log_Odds_Ratio_List'],
                    s['Original_P_Value'],
                    s['Species_Count'],
                    s['Species_Percentage']
                ),
                return_dtype=pl.Struct([
                    pl.Field('Interpro_ID', pl.Utf8),
                    pl.Field('Combined_Log_Odds_Ratio', pl.Float64),
                    pl.Field('Combined_SE', pl.Float64),
                    # Uncomment these lines to get heterogeneity data
                    #pl.Field('Tau_Squared', pl.Float64),
                    #pl.Field('Q_Statistic', pl.Float64),
                    #pl.Field('I_Squared', pl.Float64),
                    pl.Field('P_Value', pl.Float64),
                    pl.Field('Species_Count', pl.Int64),
                    pl.Field('Species_Percentage', pl.Float64),
                ])
            ).alias('Meta_Analysis_Results')
        ])
    )

def unpack_meta_analysis_results(combined_effects_df: pl.DataFrame) -> pl.DataFrame:
    """
    Unpack the Meta_Analysis_Results and Log_Odds_Ratio_Stats structs into separate columns.
    """
    return combined_effects_df.with_columns([
        pl.col('Meta_Analysis_Results').struct.field('Interpro_ID'),
        pl.col('Meta_Analysis_Results').struct.field('Combined_Log_Odds_Ratio'),
        pl.col('Meta_Analysis_Results').struct.field('Combined_SE'),
        # Uncomment these lines to get heterogeneity data
        #pl.col('Meta_Analysis_Results').struct.field('Tau_Squared'),
        #pl.col('Meta_Analysis_Results').struct.field('Q_Statistic'),
        #pl.col('Meta_Analysis_Results').struct.field('I_Squared'),
        pl.col('Meta_Analysis_Results').struct.field('P_Value'),
        pl.col('Meta_Analysis_Results').struct.field('Species_Count'),
        pl.col('Meta_Analysis_Results').struct.field('Species_Percentage')        
    ]).drop(['Meta_Analysis_Results'])

def adjust_p_values(enrichment_summary_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adjust p-values using FDR.
    """
    # Replace zero p-values with a very small non-zero value
    enrichment_summary_df = enrichment_summary_df.with_columns(
        pl.when(pl.col("P_Value") == 0)
        .then(np.nextafter(0, 1))
        .otherwise(pl.col("P_Value"))
        .alias("P_Value")
    )
    
    # Adjust meta-analysis p-values using FDR (Benjamini-Hochberg method)
    meta_p_values = enrichment_summary_df['P_Value'].to_numpy()
    _, adjusted_meta_p_values, _, _ = multipletests(meta_p_values, method='fdr_bh', alpha=0.05)
    enrichment_summary_df = enrichment_summary_df.with_columns(
        pl.Series('Adjusted_Meta_P_Value', adjusted_meta_p_values)
    )
    
# Adds biological interpretation to the statistical data
#     # Calculate Odds Ratio, Cohen's d, and Yule's Q
#     enrichment_summary_df = enrichment_summary_df.with_columns([
#         pl.col('Combined_Log_Odds_Ratio').exp().alias('Odds_Ratio')])
    
#     enrichment_summary_df = enrichment_summary_df.with_columns([
#         (pl.col('Combined_Log_Odds_Ratio') * (np.sqrt(3) / np.pi)).alias('Cohens_d'),
#         ((pl.col('Odds_Ratio') - 1) / (pl.col('Odds_Ratio') + 1)).alias('Yules_Q')
#     ])
    
    return enrichment_summary_df.drop("Original_P_Value")

# Uncomment these lines to add biological interpretation to the statistical data

# def interpret_and_filter_effect_size(enrichment_summary_df):
#     """
#     Apply effect size interpretation to enrichment data and filter significant entries.
#     """
#     def interpret_effect_size(cohens_d, yules_q, odds_ratio):
#         if abs(cohens_d) < 0.2 or abs(yules_q) < 0.1 or 0.8 < odds_ratio < 1.25:
#             return "Negligible effect"
#         elif 0.2 <= abs(cohens_d) < 0.5 or 0.1 <= abs(yules_q) < 0.3 or 0.5 < odds_ratio <= 0.8 or 1.25 <= odds_ratio < 2:
#             return "Small effect"
#         elif 0.5 <= abs(cohens_d) < 0.8 or 0.3 <= abs(yules_q) < 0.5 or 0.33 < odds_ratio <= 0.5 or 2 <= odds_ratio < 3:
#             return "Medium effect"
#         else:
#             return "Large effect"
    
#     enrichment_summary_df = enrichment_summary_df.with_columns(
#         pl.struct(['Cohens_d', 'Yules_Q', 'Odds_Ratio']).map_elements(
#             lambda s: interpret_effect_size(s['Cohens_d'], s['Yules_Q'], s['Odds_Ratio']),
#             return_dtype=pl.Utf8
#         ).alias('Effect_Size_Interpretation')
#     )
    
#     # Filter significant entries based on adjusted meta-analysis p-values
#     significant_enrichment_df = enrichment_summary_df.filter(
#         (pl.col('Adjusted_Meta_P_Value') < 0.05)
#     ).sort('Odds_Ratio', descending=True)
    
#     # Convert Log Odds Ratio IQR to Odds Ratio IQR
#     significant_enrichment_df = significant_enrichment_df.with_columns(
#         (significant_enrichment_df['Log_Odds_Ratio_IQR'].exp()).alias('Odds_Ratio_IQR')
#     )
    
#     return significant_enrichment_df

def process_enrichment_results(enrichment_df: pl.DataFrame) -> pl.DataFrame:
    """
    Main processing function that handles both single-taxon and multi-taxon cases.
    """
    
    # Check if DataFrame is empty
    if enrichment_df.is_empty():
        return pl.DataFrame({
            'Entry': ['No significant entries'],
            'Interpro_ID': ['IPR000000'],
            'Species_Count': [0],
            'Species_Percentage': [0.0],
            'Combined_Log_Odds_Ratio': [0.0],
            'Combined_SE': [0.0],
            'P_Value': [1.0],
            'Adjusted_Meta_P_Value': [1.0],
            'Log_Odds_Ratio_List': [[0.0]],  # List with a single value
            'SE_Log_Odds_Ratio_List': [[0.0]]  # List with a single value
            })
        
    # Check taxon count
    has_one_taxon_id = check_taxon_count(enrichment_df)
    
    if has_one_taxon_id:
        significant_enrichment_df = process_single_taxon_data(enrichment_df)
        
        # Process single-taxon data with percentage but without random effects
        return significant_enrichment_df
    
    else :
        significant_entries_df = preprocess_and_calculate_statistics(enrichment_df)

        combined_effects_df = apply_meta_analysis_and_stats(significant_entries_df)

        combined_effects_df = unpack_meta_analysis_results(combined_effects_df)
       
        if combined_effects_df.is_empty():
            return pl.DataFrame({
                'Entry': ['No significant entries'],
                'Interpro_ID': ['IPR000000'],
                'Species_Count': [0],
                'Species_Percentage': [0.0],
                'Combined_Log_Odds_Ratio': [0.0],
                'Combined_SE': [0.0],
                'P_Value': [1.0],
                'Adjusted_Meta_P_Value': [1.0],
                'Log_Odds_Ratio_List': [[0.0]], 
                'SE_Log_Odds_Ratio_List': [[0.0]] 
            })
 
        enrichment_summary_df = adjust_p_values(combined_effects_df)
# Uncomment for interpretation
#         significant_enrichment_df = interpret_and_filter_effect_size(enrichment_summary_df)

        # Select and reorder columns for the final summary
        significant_enrichment_df = (
            enrichment_summary_df
            .sort('Species_Percentage', descending=True)
            .select([
                'Entry',
                'Interpro_ID',
                'Species_Count',
                'Species_Percentage',
                'Combined_Log_Odds_Ratio',
                'Combined_SE',
#                 'Cohens_d',
#                 'Yules_Q',
#                 'Tau_Squared',
#                 'Q_Statistic',
#                 'I_Squared',
                'P_Value',
                'Adjusted_Meta_P_Value',
                'Log_Odds_Ratio_List',
                'SE_Log_Odds_Ratio_List'
            ])
        )

        return significant_enrichment_df

# Constants
go_categories = ["biological_process", "molecular_function", "cellular_component"]
default_columns = [
    "Entry",
    'Interpro_ID',
    "Description", 
    "Class",
    "Species_Count",
    "Species_Percentage",
    "Combined_Log_Odds_Ratio",
    "Combined_SE",
    "P_Value",
    "Adjusted_Meta_P_Value",
    "Log_Odds_Ratio_List",
    "SE_Log_Odds_Ratio_List"
]

schema = {
    "Entry": pl.Utf8,
    'Interpro_ID': pl.Utf8,
    "Description": pl.Utf8,
    "Class": pl.Utf8,
    "Species_Count": pl.Int64, 
    "Species_Percentage": pl.Float64,
    "Combined_Log_Odds_Ratio": pl.Float64,
    "Combined_SE": pl.Float64,
    "P_Value": pl.Float64,
    "Adjusted_Meta_P_Value": pl.Float64,
    "Log_Odds_Ratio_List": pl.List(pl.Float64),
    "SE_Log_Odds_Ratio_List": pl.List(pl.Float64)
}

default_row = {
    "Entry": "GO:0000000",
    "Interpro_ID": "IPR000000",
    "Description": "No significant terms found",
    "Species_Count": 0,
    "Species_Percentage": 0.0,
    "Combined_Log_Odds_Ratio": 0.0,
    "Combined_SE": 0.0,
    "P_Value": 1.0,
    "Adjusted_Meta_P_Value": 1.0,
    "Log_Odds_Ratio_List": [0.0],
    "SE_Log_Odds_Ratio_List": [0.0]
}

column_mapping = {
    "GO_term": "Entry",
    "P_value": "P_Value",
    "Odds_ratio": "Odds_Ratio",
    "Log_odds_ratio": "Log_Odds_Ratio",
    "SE_log_odds_ratio": "SE_Log_Odds_Ratio"
}

def create_empty_df(category: Optional[str] = None) -> pl.DataFrame:
    """
    Creates an empty DataFrame with default values.
    If category is provided, creates single row for that category.
    If not, creates rows for all categories.
    """
    categories = [category] if category else go_categories
    rows = []
    
    for cat in categories:
        row = default_row.copy()
        row["Class"] = cat
        rows.append(row)
    
    return pl.DataFrame(rows,schema=schema).select(default_columns)

def capitalize_first_letter(text: str) -> str:
    """Capitalizes the first letter of a string if it exists."""
    if isinstance(text, str) and text:
        return text[0].upper() + text[1:]
    return text

def process_goea_results(goea_df: pl.DataFrame, phylum_name: str) -> pl.DataFrame:
    """
    Processes Gene Ontology Enrichment Analysis (GOEA) results from multiple species per unique Phylum.
    Handles missing categories by adding empty rows with correct category labels.
    """
    if goea_df.is_empty():
        return create_empty_df()

    
    goea_df = goea_df.with_columns(
        pl.lit("IPR000000").alias("Interpro_ID")
    )
    # Preprocess the input DataFrame
    processed_df = (
        goea_df
        .with_columns([
            # Handle zero p-values
            pl.when(pl.col("P_value") == 0)
            .then(np.nextafter(0,1))
            .otherwise(pl.col("P_value"))
            .alias("P_value"),
            # Capitalize descriptions
            pl.col("Description")
            .map_elements(capitalize_first_letter, return_dtype=pl.Utf8)
            .alias("Description")
        ])
        .sort("P_value")
    )
    
    # Create entry-description mapping
    entry_description_mapping = (
        processed_df
        .select(["GO_term", "Description", 'Interpro_ID'])
        .unique()
        .rename({"GO_term": "Entry"})
    )
    
    processed_categories: Dict[str, pl.DataFrame] = {}
    
    for category in go_categories:
        try:
            category_df = processed_df.filter(pl.col("Class") == category)
            
            if category_df.is_empty():
                processed_categories[category] = create_empty_df(category)
                continue
            
            # Process the category
            enrichment_df = process_enrichment_results(
                category_df.rename(column_mapping)
            )
            
            if enrichment_df.is_empty():
                processed_categories[category] = create_empty_df(category)
                continue
            
            # Merge with descriptions and ensure all required columns
            processed_categories[category] = (
                enrichment_df
                .join(entry_description_mapping, on="Entry", how="left")
                .with_columns([
                    pl.col("Description").fill_null("No description available"),
                    pl.lit(category).alias("Class")
                ])
                .select(default_columns).cast(schema)
            )
            
        except Exception as e:
            print(f"Error processing {category} for phylum {phylum_name}: {str(e)}")
            processed_categories[category] = create_empty_df(category)
    
    # Combine all categories
    try:
        significant_go_terms = pl.concat(
            [df for df in processed_categories.values()],
            how="vertical"
        )
        
        return (significant_go_terms if not significant_go_terms.is_empty() 
                else create_empty_df())
        
    except Exception as e:
        print(f"Error combining results for phylum {phylum_name}: {str(e)}")
        return create_empty_df()

# Dictionary to store results
enrichment_results_by_phylum = {}

# Read each file from the families, domains enrichment results directories, and gene ontology results
enrichment_types_dirs = {
    "families": entry_results_dir.get("families"),
    "domains": entry_results_dir.get("domains"),
    "gene_ontology": msgoea_results_dir
}

# Iterate through all directories and files
for enrichment_type, dir_path in enrichment_types_dirs.items():
    if enrichment_type in ["families", "domains"]:
        for file_path in dir_path.glob("*_enrichment_results.txt"):
            # Get the filename from the file path
            file_name = file_path.name

            # Extract phylum name from filename by splitting on the appropriate enrichment type
            phylum_name = file_name.split(f"_{enrichment_type}")[0]

            # Load the enrichment data into a Polars DataFrame
            enrichment_df = pl.read_csv(file_path, separator="\t")

            # Process the enrichment results using the functions provided
            significant_enrichment_df = process_enrichment_results(enrichment_df)

            # Save the significant results DataFrame regardless of whether it's empty
            output_filename = f"{phylum_name}_{enrichment_type}_combined_significant_enrichment_results.parquet"
            output_path = output_dirs[f"combined_{enrichment_type}"] / output_filename
            significant_enrichment_df.write_parquet(output_path)

            # Store the processed DataFrame in the dictionary with the phylum name as the key
            if phylum_name not in enrichment_results_by_phylum:
                enrichment_results_by_phylum[phylum_name] = {}

            # Store the results based on enrichment type (domains or families)
            enrichment_results_by_phylum[phylum_name][enrichment_type] = significant_enrichment_df

    elif enrichment_type == "gene_ontology":
        # Get all files in directory and group by phylum
        phylum_files = {}
        for file in os.listdir(dir_path):
            if file.endswith("_results.txt"):
                # Extract phylum name (everything before *multi or *empty)
                phylum_name = file.split('_multi')[0].split('_empty')[0]
                if phylum_name not in phylum_files:
                    phylum_files[phylum_name] = {'multi': None, 'empty': None}
                if '_multi_species_goea_results.txt' in file:
                    phylum_files[phylum_name]['multi'] = file
                elif '_empty_results.txt' in file:
                    phylum_files[phylum_name]['empty'] = file

        # Process each phylum for gene ontology results
        for phylum_name, files_dict in phylum_files.items():
            try:
                # Ensure the phylum exists in the results dictionary without overwriting previous data
                if phylum_name not in enrichment_results_by_phylum:
                    enrichment_results_by_phylum[phylum_name] = {}

                # Add or update the gene_ontology_results key for this phylum
                enrichment_results_by_phylum[phylum_name].update({"gene_ontology_results": None})

                if files_dict['multi'] is not None:
                    # Load and process multi-species results
                    file_path = dir_path / files_dict['multi']
                    try:
                        enrichment_df = pl.read_csv(file_path, separator="\t")
                        significant_go_df = process_goea_results(enrichment_df, phylum_name)
                    except Exception as e:
                        print(f"Error processing multi-species results for {phylum_name}: {str(e)}")
                        significant_go_df = create_empty_df()
                else:
                    # Create empty DataFrame for empty results or missing files
                    significant_go_df = create_empty_df()
                    if not files_dict['empty']:
                        print(f"Warning: No results file found for {phylum_name}")

                # Save results
                output_filename = f"{phylum_name}_combined_significant_go_terms.parquet"
                output_path = output_dirs["combined_go_terms"] / output_filename
                significant_go_df.write_parquet(output_path)

                # Store in results dictionary
                enrichment_results_by_phylum[phylum_name]["gene_ontology_results"] = significant_go_df

            except Exception as e:
                print(f"Error handling phylum {phylum_name}: {str(e)}")
                # Ensure we still have an entry with empty results if an error occurs
                if phylum_name not in enrichment_results_by_phylum:
                    enrichment_results_by_phylum[phylum_name] = {}
                enrichment_results_by_phylum[phylum_name]["gene_ontology_results"] = create_empty_df()
