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
4. **Install STRIDE for the secondary structure evaluation:**
    Instructrions to install the STRIDE (Structural identification) algorithm can be found in the [ssbio documentation](https://ssbio.readthedocs.io/en/latest/instructions/stride.html)

5. **Make the C program to extract taxonomic quasi-prime regions from PDB files:**
    ```bash
    gcc -fopenmp -O2 -o pdb_quasi_prime_extractor pdb_quasi_prime_extractor.c
    ```
6. **Configure the `config.yaml` file to contain the paths specific to your enviroment**
7. **Execute the Python scripts in this order**
    1. descriptive_statistics.py
       * This script loads the extracted taxonomic quasi-primes for all three taxonomic ranks, calculates statistics and plots Figure 2 and 3a.
          
          Run using the command:
          
          ```bash
           python descriptive_statistics.py <path to the config.yaml file>
           ```
           
    2. multi_phylum_GOEA_pre_processing.ipynb
    3. multi_phylum_GOEA.ipynb
    4. multi_phylum_protein_entry_analysis_pre_processing.ipynb
    5. multi_phylum_protein_entry_analysis.ipynb
    6. combine_enrichment_results.ipynb
    7. multi_phylum_enrichment_plots.ipynb
    8. sec_struct_clustermaps.ipynb
    9. alphamissense_analysis.ipynb
    10. substitution_pre
    11. substitution analysis

---

## Citation

The citation will be placed here after publication.

## Contact
For any questions or support, please contact:
* izg5139@psu.edu
* left.bochalis@gmail.com