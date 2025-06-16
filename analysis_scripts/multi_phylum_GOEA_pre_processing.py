import os
import glob
import json
from typing import *
from pathlib import Path
import polars as pl
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

# Analysis directories
reference_mappings_dir = Path(config['peptide_match_dir'])
multi_species_goea_files_dir = Path(config['multi_species_goea_files_dir'])

# Output directories setup
output_dirs = {
    'formated_reference_mappings': reference_mappings_dir / "formated_reference_mappings",
    'goea_results': multi_species_goea_files_dir / 'multi_species_goea_results',
    'phylum_associations': multi_species_goea_files_dir / 'phylum_associations',
    'phylum_study_populations': multi_species_goea_files_dir / 'phylum_study_populations',
    'phylum_background_populations': multi_species_goea_files_dir / 'phylum_background_populations'
}

# Create all output directories
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Contains associations of UniProt protein accessions to Gene Ontology terms
uniprot_gaf = pl.read_parquet(config['goa_db_parquet_file'])

# Contains mappins of UniProt protein accessions to Taxon IDs
taxon_id_mappings = pl.read_parquet(config['taxid_mappings_parquet'])

def read_peptide_match_results(file_path: str) -> pl.DataFrame:
    """
    Reads and formats Peptide Match results into a Polars DataFrame.
    """
    mappings = pl.read_csv(
        file_path,
        separator='\t',
        comment_prefix='#',
        new_columns=["QP_peptide", "Mappings", "Protein_length", "Match_start", "Match_end"]
    )

    mappings = mappings.with_columns([
        pl.col("Mappings").str.split("|").list.get(1).alias("Protein_accession"),
        pl.col("Mappings").str.split("|").list.get(2).alias("Protein_Name")
    ])

    mappings = mappings.select(
        "QP_peptide", "Protein_accession", "Protein_Name",
        "Protein_length", "Match_start", "Match_end"
    )

    return mappings

def add_taxon_id(peptide_match_mappings: pl.DataFrame, taxon_ids: pl.DataFrame) -> pl.DataFrame:
    """
    Adds a Taxon_ID column to the Peptide Match results DataFrame based on matching Protein_accession values.
    """
    peptide_match_mappings_with_taxon_ids = peptide_match_mappings.join(
        taxon_ids,
        left_on="Protein_accession",
        right_on="Protein_accession",
        how="left"
    )
    
    return peptide_match_mappings_with_taxon_ids.drop_nulls() # Nulls in taxon id show isoforms

def filter_and_count_accessions(mappings: pl.DataFrame, min_count: int = 10):
    """
    Filters Taxon IDs based on the number of proteins that contain Quasi Prime peptides. 
    Taxon IDs with a protein count less than 10 are filtered so that downstream analyses are statisticaly reliable. 
    Also, reduces noise.
    """
    # Count the number of proteins for  each Taxon ID
    accession_counts = mappings.group_by('Taxon_ID').agg(
        pl.count('Protein_accession').alias('accession_count')
    )
    accession_counts = accession_counts.sort('accession_count', descending=True)
    accession_counts = accession_counts.filter(pl.col('accession_count') > min_count)

    # Extract unique Taxon IDs
    unique_ids = accession_counts['Taxon_ID'].unique()
    
    # Filter Peptide Match results
    filtered_mappings = mappings.filter(pl.col('Taxon_ID').is_in(unique_ids))
    
    return filtered_mappings

def process_mapping_files(directory_path):
    """
    Process all files matching *_reference_mappings.txt pattern in the given directory, 
    store results in a dictionary, save each formated dataframe to TXT file, and filter Taxon IDs that have a low protein count.
    """
    # Dictionary to store results
    results = {}
    
    # Pattern to match files
    pattern = os.path.join(directory_path, "*_reference_mappings.txt")
    
    # Find all matching files
    matching_files = glob.glob(pattern)
        
    # Process each file
    for filepath in matching_files:
        # Extract the base filename
        filename = os.path.basename(filepath)
        
        # Extract the key (part before "reference")
        key = filename.split('_reference')[0].strip()
        
        # Process the file
        results[key] = read_peptide_match_results(filepath)
        results[key] = add_taxon_id(results[key], taxon_id_mappings)
        
        # Save to TXT
        output_path = os.path.join(output_dirs['formated_reference_mappings'], f"{key}_formated_reference_mappings.txt")
        results[key].write_csv(output_path, separator = '\t')
        
        # Filter low protein count Taxon IDs
        results[key] = filter_and_count_accessions(results[key])
    
    return results

processed_reference_mappings_dict = process_mapping_files(reference_mappings_dir)

def create_associations(taxon_ids: Union[pl.Series, List[int]]) -> Dict[int, Dict[str, List[str]]]:
    """
    Creates a nested dictionary mapping taxon IDs to their protein-GO term associations.
    The nested dictionary is of this format
    Phylum
        Taxon ID
            Protein
                Associated GO terms
    """
    # Creates a set with unique Taxon IDs for an efficient lookup
    taxon_id_set = set(taxon_ids.to_numpy())
    
    # Filters the original GAF file to keep only the needed Taxon IDs
    filtered_uniprot_gaf = uniprot_gaf.filter(pl.col("Taxon_ID").is_in(taxon_id_set))
    
    # Group the filtered data by Taxon_ID and Protein_accession and aggregate GO terms for each protein within each taxon
    grouped_uniprot_gaf = (filtered_uniprot_gaf
                           .group_by(['Taxon_ID', 'Protein_accession'])
                           .agg(pl.col('GO_term').alias('GO_terms'))
                           .sort('Taxon_ID'))
    associations = {}
    
    # Process each Taxon ID
    for taxon_id, group in grouped_uniprot_gaf.group_by('Taxon_ID'):
        protein_go_dict = {protein: go_terms.to_list() for protein, go_terms in zip(group['Protein_accession'], group['GO_terms'])}
        associations[int(taxon_id[0])] = protein_go_dict
        
    return associations

def process_and_save_taxonomic_associations(
    processed_reference_mappings_dict: Dict[str, pl.DataFrame]) -> Dict[str, Dict[int, Dict[str, List[str]]]]:
    """
    Process each Peptide Match results DataFrame, create protein-GO term associations, and save results to JSON files.
    """
    # Initialize result dictionary
    taxonomic_associations = {}
       
    # Process each phylum's DataFrame and save results
    for phylum, df in processed_reference_mappings_dict.items():
        # Get unique taxon IDs for this phylum
        taxon_ids = df['Taxon_ID'].unique()
        
        # Create associations for these taxon IDs
        phylum_associations = create_associations(taxon_ids)
        
        # Store in result dictionary
        taxonomic_associations[phylum] = phylum_associations
        
        # Save this phylum's associations to a JSON file
        output_file = output_dirs['phylum_associations'] / f"{phylum}_associations.json"
        with open(output_file, 'w') as f:
            json.dump(phylum_associations, f, indent=2)
    
    return taxonomic_associations

taxonomic_associations = process_and_save_taxonomic_associations(
    processed_reference_mappings_dict
)

def create_populations(mappings: pl.DataFrame) -> Dict[int, List[str]]:
    """
    Creates a dictionary mapping taxon IDs to their associated protein accessions.
    The dictionary is of this format
    Phylum
        Taxon ID
            Proteins
    """
    # Group the data by Taxon_ID and collect unique protein accessions for each taxon
    taxon_populations = (
        mappings.group_by("Taxon_ID")
        .agg(pl.col("Protein_accession").unique().alias("Protein_accessions"))
        .sort("Taxon_ID")
    )
    # Convert the grouped data into a dictionary:
    populations = dict(zip(
        taxon_populations["Taxon_ID"].to_list(),
        taxon_populations["Protein_accessions"].to_list()
    ))
    
    return populations

def process_and_save_populations(
    processed_reference_mappings_dict: Dict[str, pl.DataFrame]) -> Tuple[Dict[str, Dict[int, List[str]]], Dict[str, Dict[int, List[str]]]]:
    """
    Process each Peptide Match result DataFrame alongside the GAF file to create both study and background populations,
    and save them to separate JSON files by phylum.
    
    The Quasi Prime containing proteins per Phylum are considered as study populations while 
    all the proteins that have associated GO terms per Phylum are considered as background populations 
    """
    # Initialize study and background dictionaries
    study_populations = {}
    background_populations = {}
        
    # Process each phylum
    for phylum, df in processed_reference_mappings_dict.items():
        
        # Create study populations for this phylum
        phylum_study_populations = create_populations(df)
        study_populations[phylum] = phylum_study_populations
        
        # Get unique taxon IDs for this phylum
        taxon_ids = list(phylum_study_populations.keys())
        
        # Filter uniprot_gaf for these taxon IDs and create background populations
        filtered_uniprot = (
            uniprot_gaf
            .filter(pl.col("Taxon_ID").is_in(taxon_ids))
            .group_by("Taxon_ID")
            .agg(pl.col("Protein_accession").unique().alias("Protein_accessions"))
            .sort("Taxon_ID")
        )
        
        phylum_background_populations = dict(zip(
            filtered_uniprot["Taxon_ID"].to_list(),
            filtered_uniprot["Protein_accessions"].to_list()
        ))
        background_populations[phylum] = phylum_background_populations
        
        # Save study populations
        with open(output_dirs['phylum_study_populations'] / f"{phylum}_study_populations.json", 'w') as f:
            json.dump(phylum_study_populations, f, indent=2)
            
        # Save background populations
        with open(output_dirs['phylum_background_populations'] / f"{phylum}_background_populations.json", 'w') as f:
            json.dump(phylum_background_populations, f, indent=2)
    
    return study_populations, background_populations

study_populations, background_populations = process_and_save_populations(
    processed_reference_mappings_dict
)
