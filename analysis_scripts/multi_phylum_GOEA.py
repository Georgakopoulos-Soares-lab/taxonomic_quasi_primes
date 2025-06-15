import os
import json
import numpy as np
import polars as pl
from typing import *
import multiprocessing
from pathlib import Path
from goatools.obo_parser import GODag
from goatools.go_enrichment import GOEnrichmentStudy

# Base directory where files are stored
base_dir = Path('/storage/group/izg5139/default/lefteris/')

# Multi Species Gene Ontology Enrichmert Analysis input directory
multi_species_goea_files_dir = base_dir / 'multi_species_goea_files'

# Output directories setup
input_dirs = {
    'goea_results': multi_species_goea_files_dir / 'multi_species_goea_results',
    'phylum_associations': multi_species_goea_files_dir / 'phylum_associations',
    'phylum_study_populations': multi_species_goea_files_dir / 'phylum_study_populations',
    'phylum_background_populations': multi_species_goea_files_dir / 'phylum_background_populations'
}

# Multi Species Gene Ontology Enrichmert Analysis output directory
multi_species_goea_results_dir = multi_species_goea_files_dir / "multi_species_goea_results"

def get_phyla_names(directory: str) -> List[str]:
    """
    Extracts the Phyla names that will be used later for the Multi-Species Gene Ontology Enrichment Analysis
    """
    # Get all files ending in .json
    phylum_json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    # Split filenames on _study and store first part
    phyla_names = [f.split('_study')[0] for f in phylum_json_files]
    
    return phyla_names

phyla_names = get_phyla_names(input_dirs['phylum_study_populations'])

def perform_multi_species_goea(study_populations, background_populations, associations, taxon_id: int) -> pl.DataFrame:
    """
    Performs Gene Ontology enrichment analysis (GOEA) for a specific species/taxon.
    It is meant to be used later for a Multi Species Gene Ontology Enrichment Analysis.
    
    This function analyzes whether certain GO terms are statistically overrepresented
    in a study set comprised of Quasi Prime containting proteins compared to a background set of GO Term annotated proteins.
    
    """
    # Load the Gene Ontology graph structure from OBO file
    go_obo = GODag(multi_species_goea_files_dir / "go-basic.obo", prt=None)
    
    # Extract the relevant data for the specified Taxon ID
    taxon_study_population = study_populations[taxon_id]
    taxon_background_population = background_populations[taxon_id]
    taxon_associations = associations[taxon_id]
    
    # Initialize the GO enrichment study object
    go_study = GOEnrichmentStudy(
        taxon_background_population, 
        taxon_associations, 
        go_obo,
        propagate_counts=False, # Don't propagate counts up the GO hierarchy
        alpha=0.05, # Significance threshold
        methods=['fdr_bh'] # Use Benjamini-Hochberg FDR correction
    )
    
    # Perform the enrichment analysis
    go_results = go_study.run_study(taxon_study_population, prt=None)
    # Filter for nominally significant results (p < 0.05)
    go_results = [r for r in go_results if r.p_uncorrected < 0.05]
    
    # Return an empty Polars DataFrame if no results are obtained
    if not go_results:
        return pl.DataFrame()

    # Process results and calculate additional statistics
    go_results_list = []
    N_background = len(taxon_background_population)
    
    for r in go_results:
    # Calculate odds ratio components with continuity correction (+0.5)
        a = r.study_count + 0.5  # Number in study set with GO term
        b = r.study_n - r.study_count + 0.5 # Number in study set without GO term
        c = r.pop_count - r.study_count + 0.5 # Number in background with GO term (excluding study set)
        d = N_background - r.pop_count - (r.study_n - r.study_count) + 0.5 # Number in background without GO term

        # Calculate effect size statistics
        odds_ratio = (a * d) / (b * c)
        log_odds_ratio = np.log(odds_ratio)
        se_log_odds_ratio = np.sqrt(1/a + 1/b + 1/c + 1/d) # Standard error of log odds ratio

        # Compile results for this GO term
        result_dict = {
            'GO_term': r.goterm.id,
            'Description': r.goterm.name,
            'Class': r.goterm.namespace,  # Biological Process, Molecular Function, or Cellular Component
            'P_value': r.p_uncorrected,
            'Adjusted_p_value': r.p_fdr_bh,  # Benjamini-Hochberg corrected p-value
            'N_protein': r.study_count,  # Number of proteins with this GO term in study set
            'N_study': r.study_n,  # Total number of proteins in study set
            'N_GO': r.pop_count,  # Number of proteins with this GO term in background
            'N_background': N_background,  # Total number of proteins in background
            'Fold_enrichment': (r.study_count * N_background) / (r.study_n * r.pop_count),  # Enrichment ratio
            'Odds_ratio': odds_ratio,
            'Log_odds_ratio': log_odds_ratio,
            'SE_log_odds_ratio': se_log_odds_ratio,
            'Taxon_ID': taxon_id
        }
        go_results_list.append(result_dict)

    # Convert results to polars DataFrame
    go_results_df = pl.DataFrame(go_results_list)

    return go_results_df

def process_phylum(phylum_name: str):
    """
    Process Gene Ontology Enrichment Analysis (GOEA) for all species within a phylum.
    
    This function loads the necessary data files for a given phylum, performs GOEA
    for each species (taxon) within that phylum, and writes the results to files.
    
    Also, stores taxa that do not have results and writes to an empty results file per Phylum.
    """
    
    # Construct file paths for input data using predefined directory structure
    phylum_associations_path = input_dirs['phylum_associations'] / f"{phylum_name}_associations.json"
    phylum_study_populations_path = input_dirs['phylum_study_populations'] / f"{phylum_name}_study_populations.json"
    phylum_background_populations_path = input_dirs['phylum_background_populations'] / f"{phylum_name}_background_populations.json"
                                               
    # Define output file paths for results and empty results
    results_file = multi_species_goea_results_dir / f"{phylum_name}_multi_species_goea_results.txt"
    empty_results_file = multi_species_goea_results_dir / f"{phylum_name}_empty_results.txt"
    
    # Load protein-GO term associations for all taxa in the phylum
    # Convert taxon IDs from strings to integers for consistency
    with open(phylum_associations_path, 'r') as f:
        associations = json.load(f)
        associations = {
            int(taxon_id): protein_dict 
            for taxon_id, protein_dict in associations.items()
        }
    
    # Load study population (proteins of interest) for each taxon
    with open(phylum_study_populations_path, 'r') as f:
        study_populations = json.load(f)
        study_populations = {
            int(taxon_id): protein_list 
            for taxon_id, protein_list in study_populations.items()
        }
    
    # Load background population (all proteins) for each taxon
    with open(phylum_background_populations_path, 'r') as f:
        background_populations = json.load(f)
        background_populations = {
            int(taxon_id): protein_list 
            for taxon_id, protein_list in background_populations.items()
        }
    
    # Track taxa with no significant results or errors
    empty_results = []
    all_results = []  # Store all valid results to write at once
    
    # Process each taxon in the phylum
    taxon_ids = list(study_populations.keys())
    for taxon_id in taxon_ids:
        try:
            # Perform GO enrichment analysis for this taxon
            result = perform_multi_species_goea(study_populations, background_populations, associations, taxon_id)
            
            if not result.is_empty():
                # Store valid results
                all_results.append(result)
            else:
                # Track taxa with no significant results
                empty_results.append({"Phylum": phylum_name, "Taxon_ID": taxon_id})
                
        except Exception as e:
            # Log and track any errors during processing
            print(f"Error processing taxon_id {taxon_id} in {phylum_name}: {str(e)}")
            empty_results.append({"Phylum": phylum_name, "Taxon_ID": taxon_id})
    
    # Write all valid results at once if any exist
    if all_results:
        # Concatenate all results into a single DataFrame
        combined_results = pl.concat(all_results)
        # Write to file with tab separator
        combined_results.write_csv(results_file, separator='\t')
    else:
        # No valid results for any taxa
        empty_results.append({"Phylum": phylum_name, "Taxon_ID": "All"})
    
    # Write list of taxa with no results to separate file if any exist
    if empty_results:
        empty_df = pl.DataFrame(empty_results)
        empty_df.write_csv(empty_results_file, separator='\t')

# Main execution block for parallel processing
if __name__ == '__main__':
    total_phyla = len(phyla_names)
    
    print(f"Starting processing {total_phyla} phyla")
    
    # Create a process pool and distribute phyla processing across cores
    with multiprocessing.Pool() as pool:
        # Process results as they complete (order doesn't matter), hence the imap_unordered
        for i, _ in enumerate(pool.imap_unordered(process_phylum, phyla_names), 1):
            print(f"Completed {i}/{total_phyla} phyla")
            
    print("All processing completed")
    