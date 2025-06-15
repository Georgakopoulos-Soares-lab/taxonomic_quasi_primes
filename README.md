# Taxonomic quasi-primes: peptides charting lineage-specific adaptations and disease-relevant loci

## Graphical Abstract
<p align="center">
  <img src="graphical_abstract.png" alt="Abstract" width="1000"/>
</p>

## Overview
This repository contains the analysis pipeline and findings for identifying unique peptide sequences, termed **taxonomic quasi-prime peptides**, that are specific to taxonomic groups. By analyzing proteomes across 24,073 species, the study identifies these peptides unique to superkingdoms, kingdoms, and phyla, offering insights into evolutionary divergence and functional roles.

---

## How to use
Clone this repository and then run the files in this order:
1. Perform the taxonomic quasi-prime peptide extraction, using the files in the extraction_scripts directory
2. descriptive_statistics.ipynb
3. multi_phylum_GOEA_pre_processing.ipynb
4. multi_phylum_GOEA.ipynb
5. multi_phylum_protein_entry_analysis_pre_processing.ipynb
6. multi_phylum_protein_entry_analysis.ipynb
7. combine_enrichment_results.ipynb
8. multi_phylum_enrichment_plots.ipynb
9. sec_struct_clustermaps.ipynb
10. alphamissense_analysis.ipynb

Supplementary scripts for taxonomic quasi-prime peptide extraction from PDB files and the secondary structure prediction using STRIDE are also provided.

---

## Key Objectives
1. **Identification of Taxonomic Quasi-Prime Peptides:**
   - Define k-mer peptides (short peptide sequences of length k) exclusive to specific taxonomic groups.
   - Assess their frequency and uniqueness across taxonomic levels (e.g., superkingdom, kingdom, phylum).

2. **Functional Analysis:**
   - Map quasi-prime peptides to proteins.
   - Perform multi-Phylum Gene Ontology (GO) enrichment to uncover their biological significance.
   - Perform multi-Phylum protein domain and protein family enrichment analyses integrating InterPro data.

3. **Structural Insights:**
   - Analyze secondary and tertiary structures of proteins containing quasi-prime peptides.
   - Explore taxon-specific protein modifications and adaptations.

4. **Applications:**
   - Highlight the potential of quasi-prime peptides in healthcare, biotechnology, and evolutionary research.

---

## Methods

### Data Sources
- **Proteomes:** Reference proteomes obtained from UniProt, encompassing 24,073 species across archaea, bacteria, eukaryotes, and viruses.

### Analysis Pipeline
1. **Peptide Extraction and Scoring:**
   - Extract k-mer sequences (lengths 5-7 amino acids).
   - Calculate ε-scores to measure taxonomic specificity.

2. **Functional Enrichment Analyses:**
   - Perform Gene Ontology (GO) enrichment at the phylum level using GOATools.
   - Conduct protein family and domain enrichment analysis.

3. **Visualization and Clustering:**
   - Employ UMAP for clustering species by quasi-prime peptide composition.
   - Use hierarchical clustering to group enriched GO terms and domains.

4. **Structural Insights:**
   - Secondary and tertiary structure analyses using STRIDE and AlphaFold datasets.
   - Multiple sequence alignment to reveal evolutionary patterns.

---

## Key Findings

### Evolutionary Insights
- Identification of thousands of taxon-specific peptides revealing lineage-specific adaptations.
- Eukaryotes exhibit the greatest diversity of taxonomic quasi-prime peptides due to their complex proteomes.

### Functional Roles
- Enriched in proteins linked to key traits, such as neuronal functions in Chordata and redox reactions in Archaea and Bacteria.
- Significant roles in transmembrane transport, ion binding, and metabolic pathways.

### Structural Adaptations
- Predominantly localized in coiled regions of proteins and critical protein domains.
- Stabilize protein complexes and facilitate species-specific traits.

### Healthcare Implications
- Human taxonomic quasi-prime loci are enriched for pathogenic missense variants, making them promising targets for diagnostics and therapeutics.

---

## Applications
1. **Diagnostics and Pathogen Detection:**
   - Use as biomarkers for identifying pathogens or monitoring microbiomes.

2. **Healthcare Innovations:**
   - Target quasi-prime peptides for drug development and vaccine design.

3. **Biotechnology:**
   - Integrate these peptides into proteomics workflows for precision studies.

---

## Acknowledgments
Research supported by the National Institute of General Medical Sciences (NIH). The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH.
