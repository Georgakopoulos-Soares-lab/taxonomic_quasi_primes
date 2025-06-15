import os
import glob
import polars as pl 
from typing import *
from pathlib import Path

# Base directory for data storage
base_dir = Path("/storage/group/izg5139/default/lefteris/")

# Set input and output directories
entry_enrichment_input_dir = base_dir / "multi_species_entry_enrichment_files"
entry_enrichment_output_dir = entry_enrichment_input_dir / "multi_species_entry_enrichment_results"
formated_reference_mappings_dir = base_dir / 'qp_peptides_over_90_per_phylum/formated_reference_mappings'

# Output directories setup
output_dirs = {
    'multi_species_entry_enrichment_results': entry_enrichment_input_dir / "multi_species_entry_enrichment_results",
    'phylum_study_domains': entry_enrichment_input_dir / 'phylum_study_domains',
    'phylum_study_families': entry_enrichment_input_dir / 'phylum_study_families',
    'phylum_background_domains': entry_enrichment_input_dir / 'phylum_background_domains',
    'phylum_background_families': entry_enrichment_input_dir / 'phylum_background_families'
}

# Create all output directories
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Read data
taxon_id_mappings = pl.read_parquet(base_dir / "taxid_mapping.parquet")
interpro_database = pl.read_parquet(entry_enrichment_input_dir / "interpro_database.parquet")

# Filter interpro database to keep only domain and family entries
domains_and_families = interpro_database.filter(
    pl.col('Type').is_in(['Domain', 'Family'])).drop('Database', 'Signature', 'Description', 'Representative')

# Add Taxon IDs to the Interpro database
interpro_database_with_taxids = domains_and_families.join(
            taxon_id_mappings,
            on="Protein_accession",
            how="left"
        )

def process_mapping_files(directory_path: str) -> pl.DataFrame:
    """
    Process all files matching *_formated_reference_mappings.txt pattern in the given directory and store results in a dictionary.
    """
    # Dictionary to store results
    results = {}
    
    # Pattern to match files
    pattern = os.path.join(directory_path, "*_formated_reference_mappings.txt")
    
    # Find all matching files
    matching_files = glob.glob(pattern)
        
    # Process each file
    for filepath in matching_files:
        # Extract the base filename
        filename = os.path.basename(filepath)
        
        # Extract the key (part before "reference")
        key = filename.split('_formated')[0].strip()
        
        # Process the file
        results[key] = pl.read_csv(filepath, separator ='\t').rename({'Protein_Name' : 'Protein_name'})
    
    return results

processed_reference_mappings_dict = process_mapping_files(formated_reference_mappings_dir)

def filter_interpro_by_mappings(
    interpro_database_with_taxids: pl.DataFrame,
    processed_reference_mappings_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    """
    Filters InterPro database based on the Taxon IDs present in the Peptide Match reference proteomes mappings dictionary.
    """
    # Pre-compute the set of all unique Taxon_IDs across all mappings
    all_taxons = set()
    taxon_dict = {}
    
    # Collect all unique taxons and prepare dictionary
    for key, mapping_df in processed_reference_mappings_dict.items():
        taxons = set(mapping_df['Taxon_ID'].unique())
        taxon_dict[key] = taxons
        all_taxons.update(taxons)
    
    # Single filter operation for all taxons
    filtered_base = interpro_database_with_taxids.filter(
        pl.col('Taxon_ID').is_in(all_taxons)
    )
    
    # Create filtered results using the filtered base
    filtered_results = {}
    
    for key, taxons in taxon_dict.items():
        filtered_results[key] = filtered_base.filter(
            pl.col('Taxon_ID').is_in(taxons)
        )
    return filtered_results

filtered_interpro_dict = filter_interpro_by_mappings(
    interpro_database_with_taxids,
    processed_reference_mappings_dict
)

def process_and_save_background_populations(
    filtered_dict: Dict[str, pl.DataFrame]) -> Dict[str, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Process each Phylum dataframe in the filtered dictionary to extract family and domain populations
    and save results to TXT files.
    """
    results = {}
    
    def extract_phylum_specific_background_populations(phylum_interpro_database: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        This is a helper function meant to be used to extract Protein Domains and Families for each Phylum.
        """
        def extract_population(Type: str) -> pl.DataFrame:
            return (
                phylum_interpro_database 
                .filter(pl.col('Type') == Type)
                .select(['Protein_name', 'Interpro_description', 'Taxon_ID', "Interpro_ID"])
                .unique() 
                .sort('Protein_name')
                .rename({
                    'Protein_name': "Protein", 
                    'Interpro_description': "Entry"
                })
            )
        
        families = extract_population('Family')
        domains = extract_population('Domain')
        
        return families, domains
    
    # Process each phylum and save results
    for phylum, df in filtered_dict.items():
        # Extract background populations and store to the corresponding dictionary
        families, domains = extract_phylum_specific_background_populations(df)
        results[phylum] = (families, domains)
        
        # Save families to file
        families_file = Path(output_dirs['phylum_background_families']) / f"{phylum}_background_families.txt"
        families.write_csv(
            families_file,
            separator="\t"
        )
        
        # Save domains to file
        domains_file = Path(output_dirs['phylum_background_domains']) / f"{phylum}_background_domains.txt"
        domains.write_csv(
            domains_file,
            separator="\t"
        )
        
    return results

background_populations = process_and_save_background_populations(
    filtered_interpro_dict
)

def combine_mapping_and_interpro_data(
    mapping_dict: Dict[str, pl.DataFrame],
    interpro_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    """
    Combines Peptide Match mapping data with InterPro data for each phylum.
    """
    def combine_peptide_match_and_interpro_data(
        phylum_mappings: pl.DataFrame, 
        phylum_interpro_database: pl.DataFrame) -> pl.DataFrame:
        """
        Helper function to combine Dataframes.
        """
        return (phylum_mappings.join(
            phylum_interpro_database, 
            on="Protein_name", 
            how="left"
        )
        .sort('Protein_name')
        .filter(pl.col('Start').is_not_null())
        .filter(pl.col('Interpro_ID').is_not_null()))
    
    combined_results = {}
    
    # Ensure we have matching keys in both dictionaries
    phyla = set(mapping_dict.keys()) & set(interpro_dict.keys())
    
    for phylum in phyla:
        
        # Get the corresponding dataframes
        mapping_df = mapping_dict[phylum]
        interpro_df = interpro_dict[phylum]
        
        # Combine the data
        combined_df = combine_peptide_match_and_interpro_data(
            mapping_df,
            interpro_df
        )
        
        combined_results[phylum] = combined_df
        
    missing_keys = set(mapping_dict.keys()) ^ set(interpro_dict.keys())
    if missing_keys:
        print(f"Warning: The following phyla were not found in both dictionaries: {missing_keys}")
    
    return combined_results

mappings_with_entries_dict = combine_mapping_and_interpro_data(
    processed_reference_mappings_dict,
    filtered_interpro_dict
)

def extract_unique_family_combinations(
    mappings_with_entries_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    """
    Creates a new dictionary containing unique combinations of protein families, QP peptides, and protein names for each phylum.
    """
    def extract_unique_families(df: pl.DataFrame) -> pl.DataFrame:
        """
        Helper function to extract unique protein families
        """
        return (
            df
            .filter(pl.col("Type") == "Family")
            .select([
                'QP_peptide',
                'Protein_name',
                'Interpro_description',
                'Interpro_ID',
                'Taxon_ID'
            ])
            .unique()
            .sort(['Protein_name', 'QP_peptide'])
        )
    
    unique_families_dict = {}
    
    for phylum, df in mappings_with_entries_dict.items():
    
        # Extract unique combinations
        unique_df = extract_unique_families(df)
        unique_families_dict[phylum] = unique_df
        
    return unique_families_dict

phylum_protein_families_dict = extract_unique_family_combinations(
    mappings_with_entries_dict
)

def extract_domains_from_dictionary(
    mappings_with_entries_dict: Dict[str, pl.DataFrame]) -> Dict[str, pl.DataFrame]:
    """
    Extracts QP peptide domains from each Peptide Match dataframe in the dictionary.
    """
    def extract_qp_peptide_domains(peptides_with_interpro_data: pl.DataFrame) -> pl.DataFrame:
        """
        Helper function to check if the Quasi Prime peptide is inside the protein domain or not.
        """
        return (
            peptides_with_interpro_data
            .filter(
                (pl.col("Type") == "Domain") &
                (pl.col("Match_start") >= pl.col("Start")) &
                (pl.col("Match_end") <= pl.col("End"))
            )
            .drop('Protein_length', 'Protein_accession_right', 'Protein_length_right', 'Start', 'End', 'Taxon_ID_right')
        )
    
    processed_results = {}
    
    for phylum, df in mappings_with_entries_dict.items():
        
        processed_df = extract_qp_peptide_domains(df)
        processed_results[phylum] = processed_df
        
    return processed_results

phylum_protein_domains_dict = extract_domains_from_dictionary(
    mappings_with_entries_dict
)

def process_and_save_study_populations(
    phylum_protein_domains_dict: Dict[str, pl.DataFrame],
    phylum_protein_families_dict: Dict[str, pl.DataFrame]) -> Dict[str, Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Process each phylum's data to create domain and family populations.
    """
    def format_families(family_df: pl.DataFrame) -> pl.DataFrame:
        """
        Format the unique families dataframe to match the required structure.
        """
        return (
            family_df
            .select(['QP_peptide', 'Protein_name', 'Interpro_description', 'Taxon_ID', 'Interpro_ID'])
            .rename({
                'Protein_name': "Protein", 
                'Interpro_description': "Entry"
            })
        )
    
    def remove_duplicates_domains(duplicated_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Deduplicates domain data.
        """
        return (
            duplicated_dataframe
            .filter(pl.col('Type') == 'Domain')
            .select(['QP_peptide', 'Protein_name', 'Interpro_description', 'Taxon_ID', 'Interpro_ID'])
            .unique()
            .sort('Protein_name')
            .rename({
                'Protein_name': "Protein", 
                'Interpro_description': "Entry"
            })
        )
    
    results = {}
    
    # Ensure we have matching keys in both dictionaries
    phyla = set(phylum_protein_domains_dict.keys()) & set(phylum_protein_families_dict.keys())
    
    for phylum in phyla:
        
        # Get the corresponding dataframes
        domains_df = phylum_protein_domains_dict[phylum]
        families_df = phylum_protein_families_dict[phylum]
        
        # Process families and domains
        families = format_families(families_df)
        domains = remove_duplicates_domains(domains_df)
        
        results[phylum] = (families, domains)
        
        # Save families to file
        families_file = output_dirs['phylum_study_families'] / f"{phylum}_study_families.txt"
        families.write_csv(
            families_file,
            separator="\t"
        )
        
        # Save domains to file
        domains_file = output_dirs['phylum_study_domains'] / f"{phylum}_study_domains.txt"
        domains.write_csv(
            domains_file,
            separator="\t"
        )
    
    # Check if any keys were missing
    missing_keys = set(phylum_protein_domains_dict.keys()) ^ set(phylum_protein_families_dict.keys())
    if missing_keys:
        print(f"Warning: The following phyla were not found in both dictionaries: {missing_keys}")
    
    return results

study_populations = process_and_save_study_populations(
    phylum_protein_domains_dict,
    phylum_protein_families_dict
)
