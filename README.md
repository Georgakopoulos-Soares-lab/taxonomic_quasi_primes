# Taxonomic quasi-primes: peptides charting lineage-specific adaptations and disease-relevant loci

## Overview
This repository contains the analysis pipeline and findings for identifying unique peptide sequences, termed **taxonomic quasi-prime peptides**, that are specific to taxonomic groups. By analyzing proteomes across 24,073 species, the study identifies these peptides unique to superkingdoms, kingdoms, and phyla, offering insights into evolutionary divergence and functional roles.

---

## Graphical Abstract
<p align="center">
  <img src="graphical_abstract.png" alt="Abstract" width="1000"/>
</p>

## How to reproduce
1. **Clone this repository:**
   
   ```bash
    git clone https://github.com/Georgakopoulos-Soares-lab/taxonomic_quasi_primes
    cd taxonomic_quasi_primes
    ```

2.  **Obtain taxonomic quasi-prime raw data for the desired taxonomic rank:**
    
    * Download the extracted taxonomic quasi-prime raw data used in this manuscript from our [Zenodo repository](https://zenodo.org/records/14385095).

3. **Install the required libraries used for the analysis:**
    
    ```bash
    pip install -r requirements.txt
    ```
4. **Install Peptide Match to correctly identify the proteins containing taxonomic quasi-prime peptides:**
     
     Instructrions to install the Peptide Match commandline tool can be found in the [Peptide Match documentation](https://research.bioinformatics.udel.edu/peptidematch/commandlinetool.jsp)

5. **Download reference proteome data from the UniProt FTP site:**
     
     We have used the reference proteomes of UniProt release 2024_01, downloaded from the [UniProt FTP site](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/)

6. **Build the Peptide Match index:**

     After install the Peptide Match command line tool, you must correctly build the Lucene index needed for the tool. You must combine all proteome FASTA files into one (e.g. all_proteomes.fasta). Then the index can be built using the command:
     
     ```bash
     java -Xms250G -Xmx250G -jar path/to/peptide_match.jar -a index -d path/to/all_proteomes.fasta -i path/to/peptide_match_index_dir -f
     ```

     Make sure you update the `-Xms250G -Xmx250G` to reflect your resources, path for the peptide_match executable, and path for the peptide_match_index directory.

7. **Install STRIDE for the secondary structure evaluation:**
    
    Instructrions to install the STRIDE (Structural identification) algorithm can be found in the [ssbio documentation](https://ssbio.readthedocs.io/en/latest/instructions/stride.html)

8. **Install GNU parallel, which is used to process PDB files in parallel in the secondary structure evaluation step.**

9. **Make the C program to extract taxonomic quasi-prime regions from PDB files:**
    
    ```bash
    gcc -fopenmp -O2 -o pdb_quasi_prime_extractor pdb_quasi_prime_extractor.c
    ```

10. **Configure the `config.yaml` file to contain the paths specific to your enviroment.**

11. **Execute the Python scripts in this order:**
    1. `descriptive_statistics.py`
       * This script loads the extracted taxonomic quasi-primes for all three taxonomic ranks, calculates statistics and plots Figure 2 and 3a.
          
          Run using the command:
          
          ```bash
           python descriptive_statistics.py <path to the config.yaml file>
           ```
        
        * This script also identifies the taxonomic quasi-prime peptides with ε-score greater than 90.00% for each Phylum and saves them.
    
    2. Run Peptide Match using the generated taxonomic quasi-primes with over 90.00% ε-score.
         * You can build a SHELL script to automate this process. The main command for this task is:

             ```bash
             java  -Xms250G -Xmx250G -jar path/to/peptide_match.jar -a query -i path/to/peptide_match_index_dir -Q path/to/phylum_taxonomic_quasi_primes.txt -l -o path/to/mapped_taxonomic_quasi_primes.txt
             ```

             Make sure you update the `-Xms250G -Xmx250G` to reflect your resources, path for the peptide_match executable, path for the peptide_match_index directory, path to the phylum taxonomic quasi_primes, and for the corresponding output. Also, the resulting Peptide Match file must be named using the phylum name followed by the `_reference_mappings.txt` prefix (e.g. `Chordata_reference_mappings.txt`)

    3. `umap_algorithm.py`
         * This script loads all phylum taxonomic quasi-prime seven-mers and their corresponding proteins containing them, filters to keep the taxonomic quasi-primes with ε-score above the 50th percentile and creates the UMAP plot present in Figure 3b.
            
            Run using the command:
            
             ```bash
             python umap_algorithm.py <path to the config.yaml file>
             ```
            
         * For this script to work you need two files: 
            * A tab-separated file containing taxonomic quasi-primes with their mapped proteomes and the corresponding protein. For example:

                 ```text
                 kmer	proteomeID	protein_name
                 NPWWMC	UP001108240_630221	A0A8C0Y3L4_CYPCA
                 PWWPCH	UP001108240_630221	A0A8C1BLA7_CYPCA
                 CKHHKW	UP001108240_630221	A0A8C1BLB4_CYPCA
                 ```

            * A tab-separated file mapping the proteome to the corresponding superkingdom. For example:
                 
                 ```text
                 UP000000216_1235689 viruses
                 UP000000226_3885 eukaryota
                 UP000000231_312153 bacteria
                 ```

    4. `multi_phylum_GOEA_pre_processing.py`
         * This script preprocesses the Peptide Match extraction files, creating study population and background population files meant to be used for the multi-phylum Gene Ontology enrichment analysis.
        
         Run using the command:
         
         ```bash
         python multi_phylum_GOEA_pre_processing.py <path to the config.yaml file>
         ```

         * For this script to work you need two files: 
             * A Gene Ontology annotation file downloaded from the [GOA database FTP repository](https://ftp.ebi.ac.uk/pub/databases/GO/goa/proteomes/)

             You can download one file for each analysed proteome, and then you must combine them into one final Parquet file.

             * A tab-separated file which maps UniProt entry to the corresponding NCBI taxon ID, also saved in Parquet format.


    5. `multi_phylum_GOEA.py`
         * This scripts performs the multi-phylum Gene Ontology enrichment analysis in parallel:

         Run this command:

         ```bash
         python multi_phylum_GOEA.py <path to the config.yaml file>
         ```

    6. `multi_phylum_protein_entry_analysis_pre_processing.py`
         * This script preprocesses the Peptide Match extraction files, creating study populations meant to be used for the protein entry enrichment analysis.

         Run this command:
         
         ```bash 
         python multi_phylum_protein_entry_analysis_pre_processing.py <path to the config.yaml file>
         ```
         
         * For this script to work you need:
             * A Parquet file containing information from the InterPro database. Raw data can be obtained from the [InterPro FTP repository](https://ftp.ebi.ac.uk/pub/databases/interpro/) and can be later formated into a Parquet file.
             * A tab-separated file which maps UniProt entry to the corresponding NCBI taxon ID, also saved in Parquet format.


    7. `multi_phylum_protein_entry_analysis.py`
         * This script performs the protein entry enrichment analysis across Phyla in parallel:
         
         Run this command:
         
         ```bash
         python multi_phylum_GOEA.py <path to the config.yaml file>
         ```

    8. `combine_enrichment_results.py`
         * This script combines the single-species results, for the multi-phylum Gene Ontology enrichment analysis and protein entry enrichment analysis, into one file per phylum, one for each analysis.

         Run this command:
         
         ```bash
         python combine_enrichment_results.py <path to the config.yaml file>
         ```
        
    9. `multi_phylum_enrichment_plots.ipynb`
         * This script creates the combined enrichment plots present in figures 4, 5, 6, 7.

         Run this command:
         
         ```bash
         python multi_phylum_enrichment_plots.py <path to the config.yaml file>
         ```
    
    10. Download the model and global health risk organism PDB files from the [AlphaFold downloads page](https://alphafold.ebi.ac.uk/download)

    11. Extract the taxonomic quasi-prime regions from the downloaded PDB files using the `pdb_quasi_prime_extractor.c` script. 
         
         * After compiling, the script is run using the following command:

         ```bash
         ./pdb_quasi_prime_extractor <path_to_mappings_file> <path_to_input_PDB_directory> <path_to_output_PDB_directory>
         ```

         * The mappings file is a tab-separated file used to give the information about where inside each protein the taxonomic quasi-prime is located and has the following format:

         ```text
         P0DPI2	100	125
         Q9Y2X8	50	75
         A0A024R1R8	210	235
         ```

         Each row has 3 fields, the UniProt entry, the start and end coordinates of the taxonomic quasi-primes present in the proteins. T

         * The input and output PDB directories, contain the original PDB files downloaded from AlphaFold and the PDB files generated which hold only the taxonomic quasi-prime peptide. 
    
    12. After extracting the PDB for the taxonomic quasi-primes you must run the STRIDE algorithm to determine the secondary structure of the peptide. This is done using the `protein_secondary_struction_prediction.sh` script.

         * Make the `protein_secondary_struction_prediction.sh` script executable:

         ```bash
         chmod +x protein_secondary_struction_prediction.sh
         ```

         * Run using the command:

         ```bash
         ./protein_secondary_struction_prediction <path to taxonomic quasi-prime PDB directory>
         ```

         * The input for the script is the directory created in the previous step containing the taxonomic quasi-prime PDB file.

    13. `sec_struct_clustermaps.py`
         * This script generates the clustermaps presenting how taxonomic quasi-primes are organized based on their secondary structure.

         Run this command:
         
         ```bash
         python sec_struct_clustermaps.py <path to the config.yaml file>
         ```

    14. `alphamissense_analysis.py`
         * This script performs the pathogenicity analysis presented in Figure 9.

         Run this command:
         
         ```bash
         python alphamissense_analysis.py <path to the config.yaml file>
         ```    
         * For this script to work you need:
             * The AlphaMissense hg38 data, which can be downloaded from their corresponding [Zenodo repository](https://zenodo.org/records/10813168)

    15. `substitution_matrix_analysis_pre_processing.py`
         * This script generates all BLOSUM62 single amino acid variant peptides, that have a positive score based on the Chordata taxonomic quasi-prime seven-mers with an ε-score greater than 90.00%.

         Run this command:
         
         ```bash
         python substitution_matrix_analysis_pre_processing.py <path to the config.yaml file>
         ```        

    16. Run Peptide Match on the generated variant data.

    17. `substitution_matrix_analysis.py`
         * This script performs the substitution matrix analysis, which checks if the Chordata quasi-prime variants are found in other phyla.

         Run this command:
         
         ```bash
         python substitution_matrix_analysis.py <path to the config.yaml file>
         ``` 
         * For this script to work you need:
             * A tab-separated file which maps the UniProt entries to the corresponding NCBI taxon ID and is of this format:

             ```text
             A4YCM6  NCBI_TaxID      399549
             A4YCM7  NCBI_TaxID      399549
             A4YCM8  NCBI_TaxID      399549
             ```
             * A file which shows the taxonomic lineage for each NCBI taxon id. This file can be obtaied from the [NCBI FTP repository](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/), on the new_taxdump folder. The file used by the script is the `rankedlineage.dmp`.

             * The `merged.dmp` file from the same NCBI repository to update any outdated NCBI taxon ids.

---

## Citation

The citation will be placed here after publication.

## Contact
For any questions or support, please contact:
* izg5139@psu.edu
* left.bochalis@gmail.com