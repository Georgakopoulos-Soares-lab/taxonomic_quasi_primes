import os
import re
import glob
import numpy as np
import polars as pl
from typing import *
import multiprocessing
from pathlib import Path
from collections import defaultdict
from scipy.stats import fisher_exact

# Base directory for data storage
base_dir = Path("/storage/group/izg5139/default/lefteris/")

# Set input and output directories
entry_enrichment_input_dir = base_dir / "multi_species_entry_enrichment_files"
entry_enrichment_output_dir = entry_enrichment_input_dir / "multi_species_entry_enrichment_results"

# Define directories containing the study and background populations for the protein domains and protein families
analysis_dirs = {
    "study_domains": entry_enrichment_input_dir / 'phylum_study_domains',
    "study_families": entry_enrichment_input_dir / 'phylum_study_families',
    "background_domains": entry_enrichment_input_dir / 'phylum_background_domains',
    "background_families": entry_enrichment_input_dir / 'phylum_background_families'
}

# Define directories meant to store the protein domains and protein families enrichment results 
output_dirs = {
    "domains": entry_enrichment_output_dir / 'domain_enrichment_results',
    "families": entry_enrichment_output_dir / 'family_enrichment_results'
}

# Create all output directories
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

def load_phylum_names() -> List[str]:
    """
    Load phylum names from data files into a list.
    Handles special phylum names like 'Candidate_division_nc10' and 'Candidatus_absconditabacteria'.
    """
    # Initialize list to store phylum names
    phylum_names = []
    
    # Get all txt files in the directory
    files = list(analysis_dirs["study_domains"].glob("*.txt"))
    
    for file_path in files:
        try:
            # Extract phylum name from filename
            file_stem = file_path.stem
            
            # Split the filename by last underscore to separate phylum name and analysis suffix
            parts = file_stem.rsplit('_', 1)
            if len(parts) == 2:
                phylum_name = parts[0].replace('_study', '')  # Remove '_study' suffix
                # Add phylum name to list if not already present
                if phylum_name not in phylum_names:
                    phylum_names.append(phylum_name)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return phylum_names

phylum_list = load_phylum_names()

def calculate_enrichment_stats(study_count, total_study, background_count, total_background):
    """
    Helper function to calculate enrichment statistics with Haldane-Anscombe correction
    """
    # Calculate the number of entries and non-entries in the study and background sets
    study_entry_count = study_count  # Number of entries in the study set
    study_non_entry_count = total_study - study_count  # Number of non-entries in the study set
    background_entry_count = background_count  # Number of entries in the background set
    background_non_entry_count = total_background - background_count  # Number of non-entries in the background set
    
    # Perform Fisher's exact test to calculate p-value for enrichment
    _, p_value = fisher_exact(
        [[study_entry_count, study_non_entry_count],
         [background_entry_count, background_non_entry_count]],
        alternative='greater'  # One-sided test to see if study set is enriched compared to background
    )
    
    # Apply Haldane-Anscombe correction by adding 0.5 to all counts to handle zero counts and reduce bias
    ha_study_entry = study_entry_count + 0.5
    ha_study_non_entry = study_non_entry_count + 0.5
    ha_background_entry = background_entry_count + 0.5
    ha_background_non_entry = background_non_entry_count + 0.5
    
    # Calculate Odds Ratio using the corrected counts
    odds_ratio = (ha_study_entry * ha_background_non_entry) / (ha_study_non_entry * ha_background_entry)
    # Calculate the log of the Odds Ratio (log-odds ratio), set to NaN if odds ratio is zero or negative
    log_or = np.log(odds_ratio) if odds_ratio > 0 else np.nan
    
    # Calculate the standard error of the log-odds ratio
    se_log_or = np.sqrt(
        1/ha_study_entry + 1/ha_study_non_entry +
        1/ha_background_entry + 1/ha_background_non_entry
    )
    
    # Calculate frequency of entries in study and background sets
    study_freq = study_entry_count / total_study if total_study > 0 else 0
    background_freq = background_entry_count / total_background if total_background > 0 else 0
    # Calculate fold enrichment as the ratio of study frequency to background frequency
    fold_enrichment = study_freq / background_freq if background_freq > 0 else np.nan
    
    # Return all the calculated statistics in a dictionary
    return {
        'Study_Entry_Counts': study_entry_count,
        'Study_Non_Entry_Counts': study_non_entry_count,
        'Background_Entry_Counts': background_entry_count,
        'Background_Non_Entry_Counts': background_non_entry_count,
        'P_Value': p_value,  # p-value from Fisher's exact test
        'Odds_Ratio': odds_ratio,  # Odds Ratio for enrichment
        'Log_Odds_Ratio': log_or,  # Log-transformed Odds Ratio
        'SE_Log_Odds_Ratio': se_log_or,  # Standard Error of Log-Odds Ratio
        'Fold_Enrichment': fold_enrichment  # Fold enrichment compared to background
    }

def process_entries(study_df, background_df, phylum_name, taxon_id):
    """
    Process entries for either domains or families, preserving Interpro_ID information
    """
    # Initialize an empty list to store results
    results = []
    
    # Filter and group the study dataframe, keeping both Entry and Interpro_ID
    study_entry_counts = (
        study_df.filter(pl.col("Taxon_ID") == taxon_id)
        .group_by(['Entry', 'Interpro_ID'])
        .len()
        .rename({'len': 'Study_Count'})
    )
    
    # Filter and group the background dataframe, keeping both Entry and Interpro_ID
    background_entry_counts = (
        background_df.filter(pl.col("Taxon_ID") == taxon_id)
        .group_by(['Entry', 'Interpro_ID'])
        .len()
        .rename({'len': 'Background_Count'})
    )
    
    # Perform a left join of background counts with study counts on both 'Entry' and 'Interpro_ID'
    merged_counts = background_entry_counts.join(
        study_entry_counts, 
        on=['Entry', 'Interpro_ID'], 
        how='left'
    ).fill_null(0)
    
    # Calculate totals as before
    total_study = study_df.filter(pl.col("Taxon_ID") == taxon_id).shape[0]
    total_background = background_df.filter(pl.col("Taxon_ID") == taxon_id).shape[0]
    
    # Iterate over the merged counts
    for row in merged_counts.iter_rows():
        # Unpack the entry name, interpro_id, background count, and study count from the row
        entry, interpro_id, background_count, study_count = row
        
        try:
            # Calculate enrichment statistics
            stats = calculate_enrichment_stats(
                int(study_count),
                total_study,
                int(background_count),
                total_background
            )
            
            # Append results including the Interpro_ID
            results.append({
                'Phylum': phylum_name,
                'Taxon_ID': taxon_id,
                'Entry': entry,
                'Interpro_ID': interpro_id,  # Include Interpro_ID in the output
                **stats
            })
        except Exception as e:
            print(f"Error processing entry {entry} (Interpro_ID: {interpro_id}) for taxon {taxon_id}: {str(e)}")
    
    return results

def save_results(results, output_path):
    """
    Save results to TXT file with multiple testing correction, even if empty
    """
    try:
        if not results:
            # Create an empty DataFrame with all expected columns
            empty_df = pl.DataFrame(schema={
                'Phylum': pl.Utf8,
                'Taxon_ID': pl.Int64,
                'Entry': pl.Utf8,
                'Study_Entry_Counts': pl.Int64,
                'Study_Non_Entry_Counts': pl.Int64,
                'Background_Entry_Counts': pl.Int64,
                'Background_Non_Entry_Counts': pl.Int64,
                'Odds_Ratio': pl.Float64,
                'Log_Odds_Ratio': pl.Float64,
                'SE_Log_Odds_Ratio': pl.Float64,
                'P_Value': pl.Float64,
                'Fold_Enrichment': pl.Float64
            })
            # Save the empty DataFrame
            empty_df.write_csv(output_path, separator='\t')
            return

        # Create a DataFrame from the results list and cast columns to appropriate types
        df = pl.DataFrame(results).with_columns([
            pl.col('Taxon_ID').cast(pl.Int64),
            pl.col('Entry').cast(pl.Utf8),
            pl.col('Study_Entry_Counts').cast(pl.Int64),
            pl.col('Study_Non_Entry_Counts').cast(pl.Int64),
            pl.col('Background_Entry_Counts').cast(pl.Int64),
            pl.col('Background_Non_Entry_Counts').cast(pl.Int64),
            pl.col('Odds_Ratio').cast(pl.Float64),
            pl.col('Log_Odds_Ratio').cast(pl.Float64),
            pl.col('SE_Log_Odds_Ratio').cast(pl.Float64),
            pl.col('P_Value').cast(pl.Float64),
            pl.col('Fold_Enrichment').cast(pl.Float64)
        ])
        
        # Filter for p-value < 0.1
        filtered_df = df.filter(pl.col('P_Value') < 0.1)
        
        # If filtered DataFrame is empty, save the empty DataFrame with column headers
        if filtered_df.is_empty():
            # Create empty DataFrame with same schema
            empty_df = pl.DataFrame(schema=df.schema)
            empty_df.write_csv(output_path, separator='\t')
        else:
            # Save the filtered DataFrame
            filtered_df.write_csv(output_path, separator='\t')

    except Exception as e:
        print(f"Error saving results to {output_path}: {str(e)}")
        # Create and save empty DataFrame even in case of error
        empty_df = pl.DataFrame(schema={
            'Phylum': pl.Utf8,
            'Taxon_ID': pl.Int64,
            'Entry': pl.Utf8,
            'Study_Entry_Counts': pl.Int64,
            'Study_Non_Entry_Counts': pl.Int64,
            'Background_Entry_Counts': pl.Int64,
            'Background_Non_Entry_Counts': pl.Int64,
            'Odds_Ratio': pl.Float64,
            'Log_Odds_Ratio': pl.Float64,
            'SE_Log_Odds_Ratio': pl.Float64,
            'P_Value': pl.Float64,
            'Fold_Enrichment': pl.Float64
        })
        empty_df.write_csv(output_path, separator='\t')

        
def perform_phylum_enrichment_analysis(phylum_name: str) -> str:
    """
    Performs enrichment analysis for both domains and families for a specific phylum.
    """
    
    # Read input data
    study_domains = pl.read_csv(analysis_dirs["study_domains"] / f"{phylum_name}_study_domains.txt", separator='\t').with_columns([pl.col("Taxon_ID").cast(pl.Int64)])
    background_domains = pl.read_csv(analysis_dirs["background_domains"] / f"{phylum_name}_background_domains.txt", separator='\t').with_columns([pl.col("Taxon_ID").cast(pl.Int64)])
    study_families = pl.read_csv(analysis_dirs["study_families"] / f"{phylum_name}_study_families.txt", separator='\t').with_columns([pl.col("Taxon_ID").cast(pl.Int64)])
    background_families = pl.read_csv(analysis_dirs["background_families"] / f"{phylum_name}_background_families.txt", separator='\t').with_columns([pl.col("Taxon_ID").cast(pl.Int64)])
    
    # Get unique taxon IDs
    taxon_ids = set(np.concatenate([
        study_domains['Taxon_ID'].unique(),
        study_families['Taxon_ID'].unique()
    ]))
    
    
    domain_results = []
    family_results = []
    
    # Process each taxon ID
    for taxon_id in taxon_ids:
        try:
            # Process domains
            domain_results.extend(
                process_entries(study_domains, background_domains, phylum_name, taxon_id)
            )
            
            # Process families
            family_results.extend(
                process_entries(study_families, background_families, phylum_name, taxon_id)
            )
        except Exception as e:
            print(f"Error processing taxon {taxon_id} in phylum {phylum_name}: {str(e)}")
            continue
    
    # Save results
    save_results(domain_results, output_dirs['domains'] / f"{phylum_name}_domains_enrichment_results.txt")
    save_results(family_results, output_dirs['families'] / f"{phylum_name}_families_enrichment_results.txt")
    
    return f"Completed {phylum_name}"

# Main execution block
if __name__ == '__main__':
    try:
        # Get list of all phyla and their data
        total_phyla = len(phylum_list)

        print(f"\nStarting processing {total_phyla} phyla...")

       # Create process pool and process phyla
        with multiprocessing.Pool() as pool:
            # Use imap_unordered to process phyla in parallel
            for i, _ in enumerate(pool.imap_unordered(perform_phylum_enrichment_analysis, phylum_list), 1):
                print(f"Completed {i}/{total_phyla} phyla")


    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
