import re, glob, pathlib, asyncio, aiohttp, nest_asyncio, shutil, pickle
import numpy as np
import polars as pl
import pandas as pd 
from typing import *
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.ticker import FixedLocator
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from goatools.obo_parser import GODag
from goatools.semsim.termwise.wang import SsWang
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# Apply nest_asyncio to allow nested event loops in Jupyter
nest_asyncio.apply()

# Set the style for seaborn plots
sns.set_style("ticks",{'font.family':'serif', 'font.serif':'Microsoft Sans Serif'})
plt.style.use('seaborn-v0_8-ticks')
sns.set_context("paper",font_scale=2)

# Set up a dictionary to store the various colormaps that will be used later for the size- and color-encoded heatmaps
colormaps={
    'biological_process': sns.cubehelix_palette(start=0, rot=0.3, dark=0.2, light=0.93, gamma=1.5, as_cmap=True),
    'molecular_function': sns.cubehelix_palette(start=1, rot=0.1, dark=0.1, light=0.93, gamma=1.5, as_cmap=True),
    'cellular_component': sns.cubehelix_palette(start=2, rot=0.3, dark=0.2, light=0.93, gamma=1.7, as_cmap=True),
    'domains': sns.cubehelix_palette(start=3, rot=0.6, dark=0.1, light=0.9, gamma=1.3, as_cmap=True),
    'families': sns.cubehelix_palette(start=0, rot=0, dark=0.0, light=0.88, gamma=1.3, as_cmap=True),
}

# Base directory for data storage
mspeea_results_dir = pathlib.Path("/storage/group/izg5139/default/lefteris/multi_species_entry_enrichment_files/multi_species_entry_enrichment_results")
msgoea_results_dir = pathlib.Path("/storage/group/izg5139/default/lefteris/multi_species_goea_files/multi_species_goea_results")

combined_enrichment_dirs = {
    "combined_domains": mspeea_results_dir / 'combined_domain_enrichment_results',
    "combined_families": mspeea_results_dir / 'combined_family_enrichment_results',
    "combined_go_terms": msgoea_results_dir / 'combined_gene_ontology_results'
}

# File patterns for each type
patterns = {
    "domains": "_domains_combined_significant_enrichment_results.parquet",
    "families": "_families_combined_significant_enrichment_results.parquet",
    "go_terms": "_combined_significant_go_terms.parquet"
}

# Create a directory to save the plots
plots = pathlib.Path("plots")
plots.mkdir(exist_ok=True)

# Load GO DAG
godag = GODag('/storage/group/izg5139/default/lefteris/multi_species_goea_files/go-basic.obo', optional_attrs={'relationship'})

go_term_mapping = {
    "Beta-ureidopropionase activity": "Beta-ureidopropionase",
    "Pyruvate dehydrogenase kinase activity": "Pyruvate dehydrogenase kinase",
    "Ligase activity": "Ligase",
    "TRNA-N1)-methyltransferase activity": "tRNA N1-methyltransferase",
    "Hydroxymethylpyrimidine kinase activity": "HMP-kinase",
    "1,4-alpha-oligoglucan phosphorylase activity": "Oligoglucan phosphorylase",
    "Iron-sulfur cluster binding": "Fe-S cluster binding",
    "Methyltransferase activity": "Methyltransferase",
    "Hydrolase activity, acting on glycosyl bonds": "Glycosyl hydrolase",
    "Threonine-phosphate decarboxylase activity": "Threonine decarboxylase",
    "Magnesium chelatase activity": "Magnesium chelatase",
    "Sequence-specific DNA binding": "DNA binding (specific)",
    "Ribonucleoside-triphosphate reductase activity": "rNTP reductase",
    "Epoxyqueuosine reductase activity": "Epoxyqueuosine reductase",
    "Proline-tRNA ligase activity": "Proline-tRNA ligase",
    "4 iron, 4 sulfur cluster binding": "4Fe-4S cluster binding",
    "Aminoacyl-tRNA ligase activity": "Aminoacyl-tRNA ligase",
    "Oxidoreductase activity": "Oxidoreductase",
    "Acyltransferase activity, transferring groups other than amino-acyl groups": "Non-amino acyltransferase",
    "Dopamine beta-monooxygenase activity": "Dopamine monooxygenase",
    "Mitochondrion targeting sequence binding": "Mitochondrial sequence binding",
    "ABC-type ferric iron transporter activity": "Ferric iron transporter",
    "AMPylase activity": "AMPylase",
    "Cysteine-type exopeptidase activity": "Cysteine exopeptidase",
    "Phosphoribosylglycinamide formyltransferase 2 activity": "GAR transformylase 2",
    "DCTP deaminase activity": "dCTP deaminase",
    "Glycosyltransferase activity": "Glycosyltransferase",
    "Starch synthase activity": "Starch synthase",
    "Exoribonuclease II activity": "Exoribonuclease II",
    "Glucose 6-phosphate:phosphate antiporter activity": "Glucose-phosphate antiporter",
    "Transmembrane transporter activity": "Transmembrane transporter",
    "G protein-coupled serotonin receptor activity": "Serotonin GPCR",
    "Clathrin binding": "Clathrin binding",
    "Ephrin receptor binding": "Ephrin receptor binding",
    "ATP-dependent peptidase activity": "ATP-dependent peptidase",
    "Hsp70 protein binding": "Hsp70 binding",
    "Diacylglycerol-dependent serine/threonine kinase activity": "Diacylglycerol Ser/Thr kinase",
    "First spliceosomal transesterification activity": "Spliceosomal transesterification",
    "Dihydrofolate synthase activity": "Dihydrofolate synthase",
    "S-glutathione dehydrogenase [NAD+] activity": "S-glutathione dehydrogenase",
    "Myosin phosphatase activity": "Myosin phosphatase",
    "GTPase activity": "GTPase",
    "GTP binding": "GTP binding",
    "Protein phosphatase regulator activity": "Phosphatase regulator",
    "Metal ion binding": "Metal ion binding",
    "Ubiquitin-protein transferase activity": "Ubiquitin transferase",
    "Monoatomic ion channel activity": "Ion channel",
    "Estrogen response element binding": "Estrogen response binding",
    "RNA 7-methylguanosine cap binding": "RNA cap binding",
    "Dynein light intermediate chain binding": "Dynein binding",
    "Hydrolase activity, hydrolyzing O-glycosyl compounds": "O-glycosyl hydrolase",
    "Extracellular matrix binding": "ECM binding",
    "Potassium channel activity": "Potassium channel",
    "Extracellular ligand-gated monoatomic ion channel activity": "Ligand-gated ion channel",
    "Squalene synthase activity": "Squalene synthase",
    "N-acetylmuramoyl-L-alanine amidase activity": "NAM-L-Ala amidase",
    "NADH dehydrogenase activity": "NADH dehydrogenase",
    "RNA polymerase II activity": "RNA polymerase II",
    "SnRNA binding": "snRNA binding",
    "Cellulose synthase activity": "Cellulose synthase",
    "UDP-glucose 6-dehydrogenase activity": "UDP-glucose dehydrogenase",
    "O-acetyltransferase activity": "O-acetyltransferase",
    "G protein-coupled receptor activity": "GPCR",
    "Serine-type endopeptidase activity": "Serine endopeptidase",
    "Trehalose biosynthetic process": "Trehalose biosynthesis",
    "L-serine biosynthetic process": "L-serine biosynthesis",
    "Receptor guanylyl cyclase signaling pathway": "Guanylyl cyclase signaling",
    "L-lysine catabolic process to acetyl-CoA via saccharopine": "L-lysine catabolism (acetyl-CoA)",
    "Alternative mRNA splicing, via spliceosome": "Alternative mRNA splicing",
    "TRNA acetylation": "tRNA acetylation",
    "Respiratory electron transport chain": "Electron transport chain",
    "Lipopolysaccharide transport": "LPS transport",
    "Peptidyl-histidine phosphorylation": "p-Histidine phosphorylation",
    "Prenylcysteine catabolic process": "Prenylcysteine catabolism",
    "Nucleotide-excision repair": "Nucleotide excision repair",
    "2'-deoxyribonucleotide biosynthetic process": "Deoxyribonucleotide biosynthesis",
    "Cysteinyl-tRNA aminoacylation": "Cysteinyl-tRNA aminoacylation",
    "RNA catabolic process": "RNA catabolism",
    "Sulfur compound biosynthetic process": "Sulfur biosynthesis",
    "DNA conformation change": "DNA conformation change",
    "Transmembrane transport": "Transmembrane transport",
    "Methylation": "Methylation",
    "Interstrand cross-link repair": "Cross-link repair",
    "Phosphorus metabolic process": "Phosphorus metabolism",
    "Transcription initiation at RNA polymerase II promoter": "RNA pol II transcription initiation",
    "DTDP-rhamnose biosynthetic process": "DTDP-rhamnose biosynthesis",
    "Extracellular polysaccharide biosynthetic process": "Polysaccharide biosynthesis",
    "Phosphate ion transmembrane transport": "Phosphate ion transport",
    "S-adenosylhomocysteine metabolic process": "SAH metabolism",
    "Small molecule metabolic process": "Small molecule metabolism",
    "'de novo' IMP biosynthetic process": "de novo IMP biosynthesis",
    "Acetyl-CoA biosynthetic process": "Acetyl-CoA biosynthesis",
    "Glucose-6-phosphate transport": "G-6P transport",
    "Cell differentiation": "Cell differentiation",
    "Cell surface receptor protein tyrosine kinase signaling pathway": "Receptor tyrosine kinase signaling",
    "Phosphorylation": "Phosphorylation",
    "Intracellular signal transduction": "Intracellular signaling",
    "Protein refolding": "Protein refolding",
    "Mitotic DNA replication preinitiation complex assembly": "DNA replication preinitiation complex",
    "SiRNA processing": "siRNA processing",
    "Dihydrofolate biosynthetic process": "Dihydrofolate biosynthesis",
    "Protein stabilization": "Protein stabilization",
    "Cellular response to heat": "Heat response",
    "Microtubule-based process": "Microtubule process",
    "Phosphatidylinositol dephosphorylation": "PI dephosphorylation",
    "DNA repair": "DNA repair",
    "Estrogen receptor signaling pathway": "Estrogen signaling",
    "Monoatomic cation transmembrane transport": "Cation transport",
    "Intraciliary retrograde transport": "Ciliary retrograde transport",
    "Protein polyglycylation": "Protein glycine modification",
    "Phospholipase C-activating G protein-coupled receptor signaling pathway": "PLC-activating GPCR signaling",
    "Release of sequestered calcium ion into cytosol by sarcoplasmic reticulum": "SR Ca\u00B2\u207A release",
    "Stabilization of membrane potential": "Membrane potential stabilization",
    "Farnesyl diphosphate metabolic process": "Farnesyl diphosphate metabolism",
    "Cellular response to nutrient levels": "Nutrient response",
    "U5 snRNA 3'-end processing": "U5 snRNA processing",
    "Monoatomic ion transmembrane transport": "Ion transport",
    "Cellulose biosynthetic process": "Cellulose biosynthesis",
    "Amino acid transport": "Amino acid transport",
    "Auxin-activated signaling pathway": "Auxin signaling",
    "Regulation of membrane potential": "Membrane potential regulation",
    "G protein-coupled receptor signaling pathway": "GPCR signaling",
    "PAS complex": "PAS complex",
    "Sec62/Sec63 complex": "Sec62/63 complex",
    "Delta DNA polymerase complex": "Delta polymerase complex",
    "Chromosome, telomeric region": "Telomeric region",
    "DNA helicase complex": "DNA helicase complex",
    "Excinuclease repair complex": "Excinuclease repair complex",
    "Ribonuclease H2 complex": "RNase H2 complex",
    "External side of plasma membrane": "Plasma membrane (external side)",
    "Cytochrome complex": "Cytochrome complex",
    "Cytoskeleton": "Cytoskeleton",
    "Thylakoid": "Thylakoid",
    "Protein-containing complex": "Protein complex",
    "Cytosol": "Cytosol",
    "F-actin capping protein complex": "F-actin capping complex",
    "Postsynapse": "Postsynapse",
    "Synaptic vesicle": "Synaptic vesicle",
    "Nuclear RNA-directed RNA polymerase complex": "Nuclear RNA pol complex",
    "Axonemal microtubule": "Axonemal microtubule",
    "Hrd1p ubiquitin ligase ERAD-L complex": "Hrd1p ERAD-L complex",
    "Perinuclear region of cytoplasm": "Perinuclear cytoplasm",
    "Microtubule": "Microtubule",
    "Protein phosphatase type 2A complex": "PP2A complex",
    "Membrane": "Membrane",
    "Transcription factor TFIIH core complex": "TFIIH core complex",
    "Cul4A-RING E3 ubiquitin ligase complex": "Cul4A-RING complex",
    "Cytoskeleton of presynaptic active zone": "Presynaptic cytoskeleton",
    "Non-motile cilium": "Non-motile cilium",
    "Intracellular membrane-bounded organelle": "Intracellular organelle",
    "Cytoplasm": "Cytoplasm",
    "Spectrin": "Spectrin",
    "Gap junction": "Gap junction",
    "Transcription regulator complex": "Transcription regulator",
    "Nuclear lamina": "Nuclear lamina",
    "RNA polymerase II, core complex": "RNA pol II core complex",
    "Apoplast": "Apoplast",
    "Golgi apparatus": "Golgi apparatus",
    "Golgi membrane": "Golgi membrane",
    "Receptor complex": "Receptor complex",
    "Extracellular region": "Extracellular region",
    "Plasma membrane": "Plasma membrane",
    "DNA replication initiation": "DNA initiation",
    "[2Fe-2S] cluster assembly": "2Fe-2S assembly",
    "Regulation of DNA-templated transcription initiation": "Transcription initiation regulation",
    "Purine nucleotide biosynthetic process": "Purine biosynthesis",
    "Protein phosphorylation": "Protein phosphorylation",
    "Polysaccharide catabolic process": "Polysaccharide catabolism",
    "Aerobic respiration": "Aerobic respiration",
    "Regulation of transcription by RNA polymerase II": "Transcription regulation (RNA pol II)",
    "Signal transduction": "Signal transduction",
    "Protein deubiquitination": "Protein deubiquitination",
    "Box C/D sno(s)RNA 3'-end processing": "Box C/D snoRNA processing",
    "Carotenoid biosynthetic process": "Carotenoid biosynthesis",
    "DGTP catabolic process": "dGTP catabolism",
    "Regulation of DNA-templated transcription": "Transcription regulation",
    "Carbohydrate derivative metabolic process": "Carbohydrate derivative metabolism",
    "Lipid metabolic process": "Lipid metabolism",
    "Prenylated protein catabolic process": "Prenylated protein catabolism",
    "Valyl-tRNA aminoacylation": "Valyl-tRNA synthesis",
    "Positive regulation of transcription by RNA polymerase II": "Pos. transcription reg. (RNA pol II)",
    "Proteolysis": "Proteolysis",
    "Carbohydrate metabolic process": "Carbohydrate metabolism",
    "Cell wall organization": "Cell wall organization",
    "Peptide transport": "Peptide transport",
    "Chromatin remodeling": "Chromatin remodeling",
    "Synapse": "Synapse",
    "Ubiquitin ligase complex": "Ubiquitin ligase complex",
    "Lateral plasma membrane": "Lateral membrane",
    "Cytoplasmic vesicle": "Cytoplasmic vesicle",
    "Glycine cleavage complex": "Glycine cleavage complex",
    "Outer membrane": "Outer membrane",
    "Neuron projection": "Neuron projection",
    "Proteasome regulatory particle, base subcomplex": "Proteasome base subcomplex",
    "Myosin complex": "Myosin complex",
    "Extracellular space": "Extracellular space",
    "Anaerobic ribonucleoside-triphosphate reductase complex": "Anaerobic ribonucleotide reductase",
    "Elongator holoenzyme complex": "Elongator holoenzyme complex",
    "Monoatomic ion channel complex": "Ion channel complex",
    "NatA complex": "NatA complex",
    "Lysosome": "Lysosome",
    "Endoplasmic reticulum": "Endoplasmic reticulum",
    "TAT protein transport complex": "TAT transport complex",
    "Nucleus": "Nucleus",
    "Bicellular tight junction": "Tight junction",
    "Phosphorelay sensor kinase activity": "Phosphorelay kinase",
    "ABC-type transporter activity": "ABC transporter",
    "Protein kinase activity": "Protein kinase",
    "Hydrolase activity": "Hydrolase",
    "ATP binding": "ATP binding",
    "Catalytic activity": "Catalytic activity",
    "ATP-dependent chromatin remodeler activity": "Chromatin remodeler (ATP-dependent)",
    "Transferase activity": "Transferase",
    "Metallopeptidase activity": "Metallopeptidase",
    "Carboxypeptidase activity": "Carboxypeptidase",
    "D-malate dehydrogenase (decarboxylating) (NAD+) activity": "D-malate dehydrogenase (NAD+)",
    "Protein serine/threonine kinase activity": "Ser/Thr protein kinase",
    "RNA polymerase II cis-regulatory region sequence-specific DNA binding": "Pol II cis-regulatory DNA binding",
    "Oxidoreductase activity, acting on iron-sulfur proteins as donors": "Fe-S protein oxidoreductase",
    "DNA-binding transcription factor activity, RNA polymerase II-specific": "Pol II-specific transcription factor",
    "Zinc ion binding": "Zinc binding",
    "Heme binding": "Heme binding",
    "Guanyl-nucleotide exchange factor activity": "GEF activity",
    "Ubiquitin protein ligase activity": "Ubiquitin ligase",
    "S-(hydroxymethyl)glutathione dehydrogenase [NAD(P)+] activity": "S-HMG dehydrogenase"
    }

# # Save the dictionary
# with open("go_term_mapping.pkl", "wb") as file:
#     pickle.dump(go_term_mapping, file)
    
# with open("go_term_mapping.pkl", "rb") as file:
#     go_term_mapping = pickle.load(file)


def read_enrichment_files(base_dirs: Dict[str, pathlib.PosixPath], patterns: Dict[str, str]) -> Dict[str, pl.DataFrame]:
    """
    Reads all parquet files from the multi-species entry enrichment results and the multi-species GOEA results and adds Phylum information.
    """
    result_dfs = {}
    
    # Process each data type (domains, families, go_terms)
    for data_type, base_dir in base_dirs.items():
        file_pattern = patterns[data_type]
        parquet_files = list(base_dir.glob(f"*{file_pattern}"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found matching pattern '{file_pattern}' in {base_dir}")
        
        # Read the first file to get schema
        first_df = pl.read_parquet(parquet_files[0])
        schema = first_df.schema
        
        # Initialize list to store dataframes for current data type
        dfs = []
        
        # Process each file
        for file_path in parquet_files:
            # Extract phylum name from filename
            phylum = (file_path.name
                     .split(file_pattern.split("_")[1])[0]
                     .rstrip('_')
                     .replace("_", " "))
            
            # Read and process the parquet file
            df = (pl.read_parquet(file_path)
                 .cast(schema)
                 .with_columns(pl.lit(phylum).alias("Phylum")))
            
            dfs.append(df)
        
        # Combine all dataframes for current data type
        result_dfs[data_type] = pl.concat(dfs)
    
    return result_dfs

enrichment_dfs = read_enrichment_files(
    base_dirs={
        'domains': combined_enrichment_dirs["combined_domains"],
        'families': combined_enrichment_dirs["combined_families"],
        'go_terms': combined_enrichment_dirs["combined_go_terms"]
    },
    patterns={
        'domains': patterns["domains"],
        'families': patterns["families"],
        'go_terms': patterns["go_terms"]
    }
)

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
    
    # Sort domains
    for domain in sorted(domain_to_kingdom_to_phylum.keys()):
        sorted_dict[domain] = {}
        
        # Sort kingdoms within each domain
        for kingdom in sorted(domain_to_kingdom_to_phylum[domain].keys()):
            # Sort phyla within each kingdom
            sorted_phyla = sorted(domain_to_kingdom_to_phylum[domain][kingdom])
            sorted_dict[domain][kingdom] = sorted_phyla
            
    return sorted_dict

# Create the sorted dictionary
domain_to_kingdom_to_phylum = sort_dictionary(domain_to_kingdom_to_phylum)

async def fetch_short_name(session: aiohttp.ClientSession, 
                         interpro_id: str, 
                         semaphore: asyncio.Semaphore) -> Tuple[str, Optional[str]]:
    """
    Fetch short name for a single InterPro ID.
    Returns tuple of (interpro_id, short_name) to maintain order.
    """
    url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{interpro_id}"
    
    try:
        async with semaphore:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return interpro_id, data["metadata"]["name"].get("short")
                elif response.status == 410:
                    return interpro_id, None
                else:
                    return interpro_id, None
    except aiohttp.ClientError as e:
        print(f"Network error for {interpro_id}: {str(e)}")
        return interpro_id, None
    except Exception as e:
        print(f"Unexpected error for {interpro_id}: {str(e)}")
        return interpro_id, None

async def fetch_all_short_names(interpro_ids: List[str], 
                              max_concurrent: int = 10) -> Dict[str, Optional[str]]:
    """Fetch short names for all InterPro IDs concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    timeout = aiohttp.ClientTimeout(total=60, connect=30)
    
    connector = aiohttp.TCPConnector(limit=max_concurrent,
                                   force_close=False,
                                   enable_cleanup_closed=True)
    
    async with aiohttp.ClientSession(timeout=timeout,
                                   connector=connector,
                                   raise_for_status=False) as session:
        tasks = []
        for id_ in interpro_ids:
            task = asyncio.create_task(fetch_short_name(session, id_, semaphore))
            tasks.append(task)
        
        results_dict = {}
        try:
            # Create progress bar before starting task processing
            pbar = tqdm(total=len(tasks), desc="Fetching InterPro data")
            
            for coro in asyncio.as_completed(tasks):
                try:
                    interpro_id, short_name = await coro
                    results_dict[interpro_id] = short_name
                    pbar.update(1)  # Update progress bar
                except Exception as e:
                    print(f"Error processing result: {str(e)}")
                    pbar.update(1)  # Update progress even on error
                    continue
            pbar.close()
        except Exception as e:
            print(f"Error in processing tasks: {str(e)}")
        finally:
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete or be cancelled
            await asyncio.gather(*tasks, return_exceptions=True)
            
        return results_dict

def add_short_names_async(df: pl.DataFrame, max_concurrent: int = 10) -> pl.DataFrame:
    """
    Add short names from InterPro API to the DataFrame using async processing.
    """
    interpro_ids = df['Interpro_ID'].to_list()
    try:
        results_dict = asyncio.run(
            fetch_all_short_names(interpro_ids, max_concurrent=max_concurrent)
        )
    except RuntimeError as e:
        if "There is no current event loop in thread" in str(e):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results_dict = loop.run_until_complete(
                    fetch_all_short_names(interpro_ids, max_concurrent=max_concurrent)
                )
            finally:
                loop.close()
    
    # Create short names list in original order
    short_names = [results_dict.get(id_, None) for id_ in interpro_ids]
    
    # Add results to DataFrame
    return df.with_columns(pl.Series(name='Short_Name', values=short_names))

enrichment_dfs['domains'] = add_short_names_async(enrichment_dfs['domains'], max_concurrent=20)
enrichment_dfs['families'] = add_short_names_async(enrichment_dfs['families'], max_concurrent=20)

enrichment_dfs['domains']['Entry', 'Short_Name'].unique().write_csv('domains_short_names.csv')
enrichment_dfs['families']['Entry', 'Short_Name'].unique().write_csv('families_short_names.csv')

def add_domain_info(enrichment_df: pl.DataFrame) -> pl.DataFrame:
    """
    Adds the corresponding Domain to each Phylum.
    """
    # Intialize an empty dictionary to store the phylum to domain mappings 
    phylum_to_domain = {}

    # Create a look-up dictionary that can categorize phyla based on the greater domain level
    for domain, kingdoms in domain_to_kingdom_to_phylum.items():
        for kingdom, phyla in kingdoms.items():
            for phylum in phyla:
                phylum_to_domain[phylum] = domain

    # Single-pass categorization
    enrichment_df = enrichment_df.with_columns(
        pl.col('Phylum').replace(phylum_to_domain).alias('Domain')
    )
    return enrichment_df.sort('Domain')

for data_type in enrichment_dfs:
    enrichment_dfs[data_type] = add_domain_info(enrichment_dfs[data_type])
    if "Short_Name" in enrichment_dfs[data_type].columns:
        enrichment_dfs[data_type] = enrichment_dfs[data_type].with_columns(
            pl.col("Short_Name").str.replace_all("_", " ").alias("Short_Name"))
    if "Short_Name" not in enrichment_dfs[data_type].columns:
            enrichment_dfs[data_type] = enrichment_dfs[data_type].with_columns(Short_Name = pl.lit('None'))
            enrichment_dfs[data_type] = enrichment_dfs[data_type].rename({"Entry" : "GO_Term_ID", 'Description': "Entry"})       
    
def calculate_weighted_mean(log_odds_ratios: NDArray[np.float64], standard_errors: NDArray[np.float64]) -> Tuple[float, float]:
    """
    Calculate weighted mean using Winsorized weights based on standard errors.
    """
    
    if len(log_odds_ratios) == 0 or len(standard_errors) == 0:
        return np.nan, np.nan
        
    # Remove any entries where SE is 0 or invalid
    valid_mask = (standard_errors > 0) & ~np.isnan(standard_errors) & ~np.isnan(log_odds_ratios)
    if not np.any(valid_mask):
        return np.nan, np.nan
        
    log_odds_ratios = log_odds_ratios[valid_mask]
    standard_errors = standard_errors[valid_mask]
    
    # Compute initial weights
    weights = 1 / (standard_errors ** 2)
    
    # Winsorize weights at the 95th percentile
    weight_percentile = np.percentile(weights, 95)
    weights = np.minimum(weights, weight_percentile)
    
    # Calculate weighted mean and combined SE
    combined_lor = np.sum(weights * log_odds_ratios) / np.sum(weights)
    combined_se = np.sqrt(1 / np.sum(weights))
    
    return combined_lor, combined_se

def cluster_common_go_terms(enrichiment_df: pl.DataFrame,
                            class_to_analyse: str) -> pl.DataFrame:
    """
    Clusters semantically similar GO Terms using Wang's method and Scipy's hierarchial clustering. 
    """
    
    # Filters the dataframe to keep entries with a Adjusted P Value less than 0.05 and based on the class to analyse
    enrichiment_df = enrichiment_df.filter((pl.col("Adjusted_Meta_P_Value") <= 0.05) & (pl.col('Class') == class_to_analyse))
    
    # Selects only the relevant columns for the analysis
    enrichiment_terms = enrichiment_df.select(['GO_Term_ID', 'Entry', 'Adjusted_Meta_P_Value', 'Species_Percentage','Combined_Log_Odds_Ratio', 'Combined_SE'])
    
    # Check if enriched_terms is empty
    if enrichiment_terms.is_empty():
        raise ValueError("The enriched_terms DataFrame is empty.")

    # Get list of GO IDs
    go_ids = enrichiment_terms.select('GO_Term_ID').to_series().to_list()

    # Calculate Wang similarity matrix
    wang_calc = SsWang(set(go_ids), godag)
    n_terms = len(go_ids)
    similarity_matrix = np.zeros((n_terms, n_terms))

    for i in range(n_terms):
        for j in range(i, n_terms):
            sim = wang_calc.get_sim(go_ids[i], go_ids[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix

    # Ensure the diagonal elements are exactly zero
    np.fill_diagonal(distance_matrix, 0)

    # Convert to condensed distance matrix
    condensed_distance = squareform(distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method='average')
    
    # Set GO Class specific threshold values for the clusters
    if class_to_analyse == 'biological_process':
        clustering_threshold = 0.54
    elif class_to_analyse == 'molecular_function':
        clustering_threshold = 0.535
    elif class_to_analyse == 'cellular_component':
        clustering_threshold = 0.52
    
    # Adjust threshold calculation
    distance_threshold = 1 - clustering_threshold
    clusters = fcluster(linkage_matrix, t = distance_threshold, criterion='distance')

    # Process clusters
    results = []

    unique_clusters, counts = np.unique(clusters, return_counts=True)
    cluster_counts = dict(zip(unique_clusters, counts))

    for cluster_id in unique_clusters:
        # Get indices of terms in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_size = len(cluster_indices)

        # Get GO IDs and terms in this cluster
        cluster_go_ids = [go_ids[i] for i in cluster_indices]
        cluster_terms = enrichiment_terms.filter(pl.col('GO_Term_ID').is_in(cluster_go_ids))

        if cluster_size >= 2:
            # Select representative term (most significant p-value)
            min_p_value = cluster_terms['Adjusted_Meta_P_Value'].min()
            representative_terms = cluster_terms.filter(pl.col('Adjusted_Meta_P_Value') == min_p_value)

            # Among those terms, select the ones with highest Species_Percentage
            max_species_percentage = representative_terms['Species_Percentage'].max()
            representative_terms = representative_terms.filter(pl.col('Species_Percentage') == max_species_percentage)

            # Handle ties by selecting the term with the highest Combined_Log_Odds_Ratio
            max_combined_lor = representative_terms['Combined_Log_Odds_Ratio'].max()
            representative_terms = representative_terms.filter(pl.col('Combined_Log_Odds_Ratio') == max_combined_lor)

            # Finally, select the first term
            representative_term = representative_terms.row(0)

            # Improved calculation of mean similarity within cluster
            cluster_indices_array = np.array(cluster_indices)
            cluster_similarity_matrix = similarity_matrix[np.ix_(cluster_indices_array, cluster_indices_array)]
            np.fill_diagonal(cluster_similarity_matrix, np.nan)
            mean_similarity = np.nanmean(cluster_similarity_matrix)

            # Extract LORs and SEs
            lor_list = cluster_terms.select(pl.col('Combined_Log_Odds_Ratio')).to_series().to_numpy()
            se_list = cluster_terms.select(pl.col('Combined_SE')).to_series().to_numpy()
            species_percentage_list = cluster_terms.select(pl.col('Species_Percentage')).to_series().to_numpy()

            # Ensure SEs are positive and non-zero
            valid_indices = se_list > 0
            lor_list = lor_list[valid_indices]
            se_list = se_list[valid_indices]
            species_percentage_list = species_percentage_list[valid_indices]

            if len(lor_list) > 0:
                # Recalculate combined LOR and SE using Winsorized weights
                combined_lor, combined_se = calculate_weighted_mean(lor_list, se_list)
            else:
                # Handle case where all SEs are zero or invalid
                combined_lor = np.nan
                combined_se = np.nan

            # Calculate combined Species_Percentage as simple mean
            if len(species_percentage_list) > 0:
                combined_species_percentage = np.mean(species_percentage_list)
            else:
                combined_species_percentage = np.nan

            results.append({
                'GO_Term_ID': representative_term[0],
                'Representative_Description': representative_term[1], 
                'Cluster_Size': cluster_size,
                'Mean_Similarity': mean_similarity,
                'Representative_LOR': combined_lor,
                'Representative_SE': combined_se,
                'Representative_Percentage': combined_species_percentage,
                'Member_GOs': ', '.join(cluster_go_ids)
            })
        elif cluster_size == 1:
            # Handle singleton clusters
            term_idx = cluster_indices[0]
            term = enrichiment_terms.filter(pl.col('GO_Term_ID') == go_ids[term_idx]).row(0)
            if term[2] < 0.05:  
                results.append({
                    'GO_Term_ID': term[0],
                    'Representative_Description': term[1],  
                    'Cluster_Size': 1,
                    'Mean_Similarity': 1.0,
                    'Representative_LOR': term[4],  
                    'Representative_SE': term[5],  
                    'Representative_Percentage': term[3], 
                    'Member_GOs': term[0]
                })

    
    result_df = pl.DataFrame(results)
    
    # Create a final dataframe that combines the clusters to the original dataframe
    final_df = (enrichiment_df
           .filter(pl.col("GO_Term_ID").is_in(result_df['GO_Term_ID']))
           .drop("Entry", 
                 "Species_Percentage",
                 'Combined_Log_Odds_Ratio', 
                 'Combined_SE')
           .join(result_df, on='GO_Term_ID'))
           
    return final_df

# Define classes of GO terms to be used for later analyses
classes = ['biological_process', 'molecular_function', 'cellular_component']

def process_phylum_go_terms(df: pl.DataFrame) -> pl.DataFrame:
    """
    Process GO terms clustering for each phylum and class combination, then combine results.
    """
    
    # Get unique phyla and classes
    phyla = df.select('Phylum').unique().to_series().to_list()
    
    # Store results for each phylum
    phylum_results = []
    
    for phylum in phyla:        
        # Filter data for current phylum
        phylum_df = df.filter(pl.col('Phylum') == phylum)
        
        # Store results for each class within the phylum
        class_results = []
        
        for class_name in classes:
            try:
                # Process the class-specific clustering
                class_result = cluster_common_go_terms(
                    enrichiment_df=phylum_df,
                    class_to_analyse=class_name
                )
                
                if not class_result.is_empty():
                    # Add phylum and class information if not already present
                    class_result = class_result.with_columns([
                        pl.lit(phylum).alias('Phylum'),
                        pl.lit(class_name).alias('Class')
                    ])
                    
                    class_results.append(class_result)
                    
                    
            except ValueError as e:
                continue
        
        if class_results:
            phylum_result = pl.concat(class_results, how='vertical')
            phylum_results.append(phylum_result)
    
    # Combine results from all phyla
    if not phylum_results:
        raise ValueError("No results were generated for any phylum")
    
    final_results = pl.concat(phylum_results, how='vertical')
    
    # Sort the results by Phylum, Class, and cluster size
    final_results = final_results.sort(
        ['Phylum', 'Class', 'Cluster_Size', 'Representative_LOR'], 
        descending=[False, False, True, True]
    )
    
    return final_results

def process_go_enrichment(df: pl.DataFrame) -> pl.DataFrame:
    """
    Wrapper function to process GO enrichment data with error handling.
    """
    try:
        results = process_phylum_go_terms(df)
        return results
    except Exception as e:
        print(f"Error processing GO enrichment: {str(e)}")
        return pl.DataFrame()

clustered_go_terms = process_go_enrichment(enrichment_dfs['go_terms'])

def calculate_phylum_coverage(
    entry_enrichment_df: pl.DataFrame,
    id_type: Literal['go', 'interpro'] = 'interpro',
    default_class: str = 'biological_process') -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Calculate phylum coverage for either GO terms or InterPro IDs.
    """
    # Define ID column name based on type
    id_col = 'GO_Term_ID' if id_type == 'go' else 'Interpro_ID'
    
    # Define schema based on ID type
    base_schema = {
        'Entry': pl.Utf8,
        'Interpro_ID': pl.Utf8,
        'Species_Count': pl.Int64,
        'Species_Percentage': pl.Float64,
        'Combined_Log_Odds_Ratio': pl.Float64,
        'Combined_SE': pl.Float64,
        'P_Value': pl.Float64,
        'Adjusted_Meta_P_Value': pl.Float64,
        'Log_Odds_Ratio_List': pl.List(pl.Float64),
        'SE_Log_Odds_Ratio_List': pl.List(pl.Float64),
        'Phylum': pl.Utf8,
        'Short_Name': pl.Utf8,
        'Domain': pl.Utf8,
        'Entry_Domain_of_Origin': pl.Utf8,
        'Phyla_with_entry': pl.Int64,
        'Total_number_of_Phyla': pl.Int64,
        'Phylum_coverage': pl.Float64,
        'Counter': pl.Int64
    }
    
    # Add GO-specific fields if needed
    schema = base_schema.copy()
    if id_type == 'go':
        schema.update({
            'GO_Term_ID': pl.Utf8,
            'Class': pl.Utf8
        })
    
    # Filter to keep only statistically significant entries
    entry_enrichment_df = entry_enrichment_df.filter(pl.col("Adjusted_Meta_P_Value") < 0.05)

    # Calculate the total number of Phyla per Domain
    domain_phylum_counts = entry_enrichment_df.group_by('Domain').agg(
        n_phyla=pl.col('Phylum').n_unique()
    )

    entry_domain_phylum_counts = (
        entry_enrichment_df
        .filter(pl.col('Combined_Log_Odds_Ratio').is_not_null())
        .group_by(['Domain', id_col])
        .agg(
            n_phyla_entry=pl.col('Phylum').n_unique(),
            lor_list=pl.col('Combined_Log_Odds_Ratio'),
            se_list=pl.col('Combined_SE'),
            first_entry=pl.col('Entry').first(),
            first_short_name=pl.col('Short_Name').first()
        )
        .with_columns([
            pl.struct(['lor_list', 'se_list'])
            .map_elements(lambda x: calculate_weighted_mean(
                np.array(x['lor_list']), 
                np.array(x['se_list'])
            ), return_dtype=pl.List(pl.Float64))
            .alias('weighted_stats')
        ])
        .with_columns([
            pl.col('weighted_stats').list.get(0).alias('mean_combined_log_odds_ratio'),
            pl.col('weighted_stats').list.get(1).alias('combined_se')
        ])
        .drop(['lor_list', 'se_list', 'weighted_stats'])
    )

    entry_domain_phylum_counts = entry_domain_phylum_counts.sort(['Domain', "n_phyla_entry"], descending=[False, True])

    # Merge the counts to compute the percentage of Phyla each Entry covers in its Domain
    entry_phylum_coverage = entry_domain_phylum_counts.join(domain_phylum_counts, on='Domain')
    entry_phylum_coverage = entry_phylum_coverage.with_columns(
        phylum_coverage_percentage=(pl.col('n_phyla_entry') / pl.col('n_phyla') * 100)
    )

    # Rename columns
    entry_phylum_coverage = entry_phylum_coverage.rename({
        'n_phyla': "Total_number_of_Phyla",
        'n_phyla_entry': "Phyla_with_entry",
        'phylum_coverage_percentage': "Phylum_coverage",
        'mean_combined_log_odds_ratio': "Mean_Log_Odds_Ratio"
    })

    selected_ids = set()
    selected_rows = []

    for domain in sorted(list(entry_enrichment_df['Domain'].unique())):
        domain_entries = (entry_phylum_coverage
            .filter(pl.col("Domain") == domain)
            .sort(["Phylum_coverage", 'Mean_Log_Odds_Ratio'], descending=[True, True])
        )

        domain_entries_selected = 0

        for row in domain_entries.iter_rows(named=True):
            if domain_entries_selected >= 10:
                break

            if row[id_col] not in selected_ids:
                selected_ids.add(row[id_col])
                selected_rows.append(row)
                domain_entries_selected += 1

    # Create DataFrame with selected entries
    entries_to_keep_df = pl.DataFrame(selected_rows).sort(["Domain", "Phylum_coverage"], descending=[False, True])
    entries_to_keep_df = entries_to_keep_df.with_columns(pl.arange(1, entries_to_keep_df.height + 1).alias("Counter"))

    entries_to_keep = entries_to_keep_df.select([id_col, 'Counter'])
    entries_to_keep = dict(entries_to_keep.iter_rows())

    # Join the coverage information back to the original DataFrame
    # Modified to keep all necessary columns
    result_df = entry_enrichment_df.join(
        entry_phylum_coverage.select(pl.exclude(['first_entry'])), 
        on=['Domain', id_col]
    )

    # Filter the top entries and add a column for the original domain
    filtered_results_df = result_df.filter(pl.col(id_col).is_in(entries_to_keep.keys())).join(
        entries_to_keep_df.select([id_col, 'Domain', 'Counter', 'first_short_name']).rename({'Domain': 'Entry_Domain_of_Origin'}),
        on=id_col
    )
    filtered_results_df = filtered_results_df.drop("Mean_Log_Odds_Ratio")

    # Initialize new_df with filtered_results_df
    new_df = filtered_results_df.cast(schema)
    new_df = new_df.select(list(schema.keys()))

    # Create a list to store new rows
    new_rows = []

    # Get unique phyla from the dataframe
    unique_phyla = filtered_results_df['Phylum'].unique()

    # For each phylum
    for phylum in unique_phyla:
        # Get existing entries for this phylum
        phylum_ids = filtered_results_df.filter(pl.col("Phylum") == phylum)[id_col].to_list()

        # Find missing entries
        missing_ids = set(list(entries_to_keep.keys())) - set(phylum_ids)

        if missing_ids:
            # For each missing entry
            for entry_id in missing_ids:
                # Get the entry's original information using named columns
                entry_info = entries_to_keep_df.filter(pl.col(id_col) == entry_id).row(0, named=True)
                entry_coverage = entry_phylum_coverage.filter(pl.col(id_col) == entry_id).row(0, named=True)

                # Create base new row with named column access
                new_row = {
                    'Entry': entry_coverage['first_entry'],
                    'Interpro_ID': 'IPR000000' if id_type == 'go' else entry_id,
                    'Species_Count': 0,
                    'Species_Percentage': 0.0,
                    'Combined_Log_Odds_Ratio': 0.0,
                    'Combined_SE': 0.0,
                    'P_Value': 1.0,
                    'Adjusted_Meta_P_Value': 1.0,
                    'Log_Odds_Ratio_List': [0.0],
                    'SE_Log_Odds_Ratio_List': [0.0],
                    'Phylum': phylum,
                    'Short_Name': entry_coverage['first_short_name'], 
                    'Domain': entry_coverage['Domain'],
                    'Entry_Domain_of_Origin': entry_info['Domain'],
                    'Phyla_with_entry': entry_coverage['Phyla_with_entry'],
                    'Total_number_of_Phyla': entry_coverage['Total_number_of_Phyla'],
                    'Phylum_coverage': entry_coverage['Phylum_coverage'],
                    'Counter': entries_to_keep[entry_id]
                }

                # Add GO-specific fields if needed
                if id_type == 'go':
                    new_row.update({
                        'GO_Term_ID': entry_id,
                        'Class': default_class
                    })

                new_rows.append(new_row)

    # If there are new rows, concatenate them with new_df
    if new_rows:
        new_df = pl.concat([new_df, pl.DataFrame(new_rows, schema=schema)])

    phylum_order = {}
    order_index = 0

    for domain in domain_to_kingdom_to_phylum:
        for kingdom in domain_to_kingdom_to_phylum[domain]:
            for phylum in domain_to_kingdom_to_phylum[domain][kingdom]:
                phylum_order[phylum] = order_index
                order_index += 1

    # Create a temporary column for phylum ordering using a map expression
    organized_df = (
        new_df.with_columns([
                (pl.col("Phylum")
                .map_elements(lambda x: phylum_order.get(x, float('inf')), return_dtype=pl.Int64)
                .alias("phylum_order")),
                (pl.col('Combined_Log_Odds_Ratio').exp().alias('Combined_Odds_Ratio'))
            ])
            .sort(["phylum_order", "Counter"])
            .drop("phylum_order")
    )
    
    return result_df, organized_df

def process_all_dataframes(data_dict: Dict[str, pl.DataFrame]) -> Tuple[Dict[str, pl.DataFrame]]:
    """
    Process all dataframes in the dictionary and handle GO terms separately by class.
    """
    
    results = {}
    filtered_results = {}
    
    # Process domains and families dataframes
    results['domains'], filtered_results['domains'] = calculate_phylum_coverage(data_dict['domains'], id_type='interpro')
    results['families'], filtered_results['families'] = calculate_phylum_coverage(data_dict['families'], id_type='interpro')
    
    # Split and process GO terms by class
    go_terms_df = data_dict['go_terms']
    go_terms_df = go_terms_df.with_columns(Short_Name = pl.lit('None'))
    
    # Process biological processes
    bio_process_df = go_terms_df.filter(pl.col('Class') == 'biological_process')
    bio_process_df, filtered_bio_process_df = calculate_phylum_coverage(bio_process_df, id_type='go', default_class='biological_process')
    
    # Process molecular functions
    mol_function_df = go_terms_df.filter(pl.col('Class') == 'molecular_function')
    mol_function_df, filtered_mol_function_df = calculate_phylum_coverage(mol_function_df, id_type='go', default_class='molecular_function')
    
    # Process cellular components
    cell_component_df = go_terms_df.filter(pl.col('Class') == 'cellular_component')
    cell_component_df, filtered_cell_component_df = calculate_phylum_coverage(cell_component_df, id_type='go', default_class='cellular_component')
    
    # Concatanate into a final DataFrame and store in the dictionary
    results['go_terms'] = pl.concat([bio_process_df, mol_function_df, cell_component_df])
    filtered_results['go_terms'] = pl.concat([filtered_bio_process_df, filtered_mol_function_df, filtered_cell_component_df])

    return results, filtered_results

enrichment_dfs_with_phylum_coverage, filtered_enrichment_dfs = process_all_dataframes(enrichment_dfs)

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

def create_colorbar_legend(enrichment_df: pl.DataFrame, norm: Normalize, cmap: ListedColormap, colorbar_file: str) -> None:
    """
    Creates a horizontal colorbar legend matching the main plot's color scale, with consistent dimensions to match R output.
    """
    fig, ax = plt.subplots(figsize=(3, 1))
    plt.subplots_adjust(left=0.16, right=0.84, bottom=0.3, top=0.7)
    
    # Use the passed colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.outline.set_visible(False)
    
    min_val = enrichment_df['Combined_Log_Odds_Ratio'].min()
    max_val = enrichment_df['Combined_Log_Odds_Ratio'].max()
    
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_title('ln(Enrichment)', fontsize=12, pad=10, fontweight='bold')
    
    tick_locations = np.linspace(min_val, max_val, num=5)
    cbar.set_ticks(tick_locations)
    cbar.set_ticklabels([f'{val:.1f}' for val in tick_locations])
    cbar.ax.tick_params(labelsize=10)
    
    plt.savefig(colorbar_file, bbox_inches='tight', dpi=300)
    plt.close()

def create_size_legend(min_size: float, max_size: float, size_legend_file: str) -> None:
    """
    Creates a horizontal legend plot for the species percentage matching the main plot's size scale.
    """

    fig, ax = plt.subplots(figsize=(3, 1))
    
    # Use the actual percentage values that correspond to the plot's size range
    sizes = [0, 25, 50, 75, 100]  # Percentage values
    x_positions = np.linspace(0.2, 0.8, len(sizes))  # Spread points horizontally
    y_positions = np.zeros(len(sizes))  # All points at same y-level
    
    # Use the same size scaling as the main plot
    scatter_sizes = np.interp(sizes, [0, 100], [min_size, max_size])
    scatter = ax.scatter(x_positions, y_positions, 
                        s=scatter_sizes, 
                        c='black')
    
    # Add percentage labels below points
    for size, x in zip(sizes, x_positions):
        ax.text(x, -0.2, f'{size}', 
                horizontalalignment='center',
                verticalalignment='top',
                fontsize=10)
    
    # Add title above
    ax.text(0.5, 0.5, 'Species Percentage (%)', 
            fontsize=12, 
            fontweight='bold',
            horizontalalignment='center',
            verticalalignment='bottom')
    
    # Set appropriate limits for horizontal layout
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.3, 0.6)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig(size_legend_file, bbox_inches='tight', dpi=300)
    plt.close()

def create_clustered_dot_plot(
    enrichment_df: pl.DataFrame, 
    entry_column: str, 
    plot_file_path: str, 
    cmap: ListedColormap, 
    colorbar_file: str, 
    size_legend_file: str, 
    size_legend: bool) -> None:
    
    """
    Creates a size- and color-encoded heatmap
    """
    
    # Create main figure with simpler layout
    fig = plt.figure(figsize=(16, 10), dpi=300) 
    gs = fig.add_gridspec(2, 20, height_ratios=[0.3, 10], hspace=0.05) 

    # Create domain color bar axes and main scatter plot axes
    ax_domains = fig.add_subplot(gs[0, :19])
    ax = fig.add_subplot(gs[1, :19])
    
    # Define size range 
    min_size = 10  
    max_size = 100  

    # Create linear normalization since values are already log-transformed
    norm = Normalize(vmin=enrichment_df['Combined_Log_Odds_Ratio'].min(), 
                    vmax=enrichment_df['Combined_Log_Odds_Ratio'].max())

    sns.scatterplot(data=enrichment_df.to_pandas(), 
                    x='Phylum',
                    y=entry_column,
                    size='Species_Percentage',
                    hue='Combined_Log_Odds_Ratio',
                    sizes=(min_size, max_size),
                    palette=cmap,
                    hue_norm=norm,
                    ax=ax,
                    legend=False)

    # Update x-axis tick labels
    x_ticks = range(len(ax.get_xticklabels()))
    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    
    # Update y-axis tick labels
    y_ticks = range(len(ax.get_yticklabels()))
    y_labels = [label.get_text() for label in ax.get_yticklabels()]
    y_labels = [go_term_mapping.get(label, label) for label in y_labels]
    
    # Set the tick positions and new labels with adjusted font sizes
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.set_yticklabels(y_labels, fontsize=13, va='center') 

    ax.set_xlim(left=-1, right=(enrichment_df['Phylum'].n_unique()))
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Hide domain axes ticks
    ax_domains.axis('off')

    # Get phyla in plot order and create position mapping
    phyla_in_plot = x_labels
    x_positions = {phylum: i for i, phylum in enumerate(phyla_in_plot)}

    # Create phylum to lineage mapping
    phylum_to_lineage = {}
    for domain in domain_to_kingdom_to_phylum:
        for kingdom in domain_to_kingdom_to_phylum[domain]:
            phyla = domain_to_kingdom_to_phylum[domain][kingdom]
            for phylum in phyla:
                phylum_to_lineage[phylum] = (domain, kingdom)

    # Define domain colors
    domain_colors = {
        'Archaea': '#0072b2',
        'Bacteria': '#e69f00',
        'Eukaryota': '#009e73',
        'Viruses': '#cc79a7'
    }

    # Create domain to x positions mapping
    domain_to_x_positions = {}
    for domain in domain_colors.keys():
        phyla_in_domain = [phylum for phylum in phyla_in_plot 
                          if phylum_to_lineage.get(phylum, (None, None))[0] == domain]
        x_positions_in_domain = [x_positions[phylum] for phylum in phyla_in_domain]
        domain_to_x_positions[domain] = x_positions_in_domain

    # Add colored rectangles for domains with adjusted text size
    for domain in domain_colors.keys():
        x_positions_in_domain = domain_to_x_positions.get(domain, [])
        if x_positions_in_domain:
            min_x_domain = min(x_positions_in_domain) - 0.5
            max_x_domain = max(x_positions_in_domain) + 0.5
            rect = plt.Rectangle((min_x_domain, 0), max_x_domain - min_x_domain, 1,
                               color=domain_colors[domain], alpha=0.8)
            ax_domains.add_patch(rect)

    # Set domain axes limits
    ax_domains.set_xlim(ax.get_xlim())
    ax_domains.set_ylim(0, 1)

    # Format x-axis labels
    x_labels = [label.replace("Candidatus ", "C. ").replace("Candidate ", "C. ") for label in x_labels]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=10)
    plt.subplots_adjust(left=0.3)  
    
    sns.despine()
    plt.tight_layout(pad=0.2) 
    plt.savefig(plot_file_path, bbox_inches='tight', pad_inches=0.1, dpi=300)  
    plt.show()
    
    # Create separate legends using the same normalization and size range
    create_colorbar_legend(enrichment_df, norm, cmap, colorbar_file)
    if size_legend:
        create_size_legend(min_size, max_size, size_legend_file)
    else:
        pass 

create_clustered_dot_plot(
    filtered_enrichment_dfs['domains'], 
    "Short_Name",
    plots / "domains_dotplot.svg", 
    colormaps['domains'], 
    plots / "domains_colorbar.svg",  
    plots / "domains_size_legend.svg",
    size_legend = True)

create_clustered_dot_plot(
    filtered_enrichment_dfs['families'], 
    "Short_Name",
    plots / "families_dotplot.svg", 
    colormaps['families'], 
    plots / "families_colorbar.svg",  
    plots / "families_size_legend.svg",
    size_legend = False)

create_clustered_dot_plot(
    filtered_enrichment_dfs['go_terms'].filter(pl.col('Class') == 'biological_process'), 
    "Entry",
    plots / "biological_process.svg", 
    colormaps['biological_process'], 
    plots / "biological_process_colorbar.svg", 
    plots / "biological_process_size_legend.svg",
    size_legend = False)

create_clustered_dot_plot(
    filtered_enrichment_dfs['go_terms'].filter(pl.col('Class') == 'molecular_function'), 
    "Entry",
    plots / "molecular_function.svg", 
    colormaps['molecular_function'], 
    plots / "molecular_function_colorbar.svg", 
    plots / "molecular_function_size_legend.svg",
    size_legend = False)

create_clustered_dot_plot(
    filtered_enrichment_dfs['go_terms'].filter(pl.col('Class') == 'cellular_component'), 
    "Entry",
    plots / "cellular_component.svg", 
    colormaps['cellular_component'], 
    plots / "cellular_component_colorbar.svg", 
    plots / "cellular_component_size_legend.svg",
    size_legend = False)

# Define a list of representative phyla that are present in the clustered go terms to keep
representative_phyla = [
    "Candidatus Lokiarchaeota", "Candidatus Thorarchaeota", "Candidatus Bathyarchaeota",
    "Candidatus Woesearchaeota", "Candidatus Korarchaeota", "Chlorobiota",
    "Candidate division NC10", "Nitrospinota", "Candidatus Gracilibacteria",
    "Candidatus Poribacteria", "Cryptomycota","Blastocladiomycota",
    "Chordata", "Nematoda", "Mollusca",
    "Tardigrada", "Platyhelminthes", "Euglenozoa",
    "Ciliophora", "Bacillariophyta", "Myzozoa",
    "Foraminifera", "Streptophyta","Chlorophyta"
]

def extract_top_phylum_entries(enrichment_df: pl.DataFrame, percentage_threshold: float, go_analysis: bool, go_class: str = None) -> pl.DataFrame:
    """
    Extracts top 3 entries per Phylum based on either representative or combined log odds ratio.
    """
    if go_analysis:
        enrichment_df = (enrichment_df
                        .filter((pl.col("Class") == go_class) & (pl.col("Phylum").is_in(representative_phyla)))
                        .filter((pl.col("Representative_Percentage") >= percentage_threshold) & 
                               (pl.col("Adjusted_Meta_P_Value") <= 0.05))
                        .sort(["Cluster_Size", "Representative_LOR", "Phylum"], descending=[True, True, False])
                        .group_by("Phylum")
                        .head(3)
                        .drop(['Log_Odds_Ratio_List', 'SE_Log_Odds_Ratio_List', 'Species_Count', 
                              'P_Value', 'GO_Term_ID', 'Member_GOs']))
    else:
        enrichment_df = (enrichment_df
                        .filter(pl.col("Species_Percentage") >= percentage_threshold)
                        .filter(pl.col("Adjusted_Meta_P_Value") <= 0.05)
                        .sort(["Combined_Log_Odds_Ratio", "Phylum"], descending=[True, False])
                        .group_by("Phylum")
                        .head(3)
                        .drop(['Log_Odds_Ratio_List', 'SE_Log_Odds_Ratio_List', 'Species_Count', 'P_Value']))

    df_columns = enrichment_df.columns
    phylum_info = (enrichment_df.group_by(['Phylum', 'Domain'])
                   .agg(pl.len().alias('len'))
                   .sort('Phylum'))

    new_rows = []
    for row in phylum_info.iter_rows(named=True):
        if row['len'] < 3:
            base_row = {col: None for col in df_columns}
            base_row.update({
                'Phylum': row['Phylum'],
                'Domain': row['Domain'],
                'Adjusted_Meta_P_Value': 1.0,
            })
            
            if go_analysis:
                base_row.update({
                    'Representative_Description': " ",
                    'Representative_LOR': 0.0,
                    'Representative_SE': 0.0,
                    'Representative_Percentage': 0.0,
                    'Mean_Similarity': 0.0,
                    'Cluster_Size': 0,
                    'Interpro_ID': "IPR000000",
                    'Short_Name': " ",
                    'Class': go_class
                })
            else:
                base_row.update({
                    'Entry': " ",
                    'Combined_Log_Odds_Ratio': 0.0,
                    'Combined_SE': 0.0,
                    'Species_Percentage': 0.0,
                    'Interpro_ID': "IPR000000",
                    'Short_Name': " "
                })
            
            new_rows.extend([base_row.copy() for _ in range(3 - row['len'])])

    if new_rows:
        enrichment_df = pl.concat([enrichment_df, pl.DataFrame(new_rows)])

    lor_col = "Representative_LOR" if go_analysis else "Combined_Log_Odds_Ratio"
    se_col = "Representative_SE" if go_analysis else "Combined_SE"
    
    enrichment_df = enrichment_df.with_columns([
        (1.96 * pl.col(se_col)).alias("MOE"),
        (pl.col(lor_col) - (1.96 * pl.col(se_col))).alias("Lower_CI"),
        (pl.col(lor_col) + (1.96 * pl.col(se_col))).alias("Upper_CI")
    ])

    return enrichment_df.sort('Phylum')

def process_and_save(df, filename, go_analysis=False, go_class=None):
    # Apply the extract, replace, and save logic in one function
    result = extract_top_phylum_entries(
        enrichment_df=df,
        percentage_threshold=5.0,
        go_analysis=go_analysis,
        go_class=go_class
    )
    result.with_columns(
        pl.col("Phylum")
        .str.replace("Candidatus ", "C. ")
        .str.replace("Candidate ", "C. ")
    ).write_csv(filename)
    return result

# Process top domains and families
top_domains = process_and_save(enrichment_dfs['domains'], "domains.csv")
top_families = process_and_save(enrichment_dfs['families'], "families.csv")

# Process and concatenate top GO terms per GO class
selected_go_terms = [
    process_and_save(clustered_go_terms, None, go_analysis=True, go_class=go_class)
    for go_class in classes
]
top_clustered_go_terms = pl.concat(selected_go_terms)
top_clustered_go_terms.write_csv("go_terms.csv")

def create_lollipop_plot(top_clustered_go_terms: pl.DataFrame, go_term_mapping: Dict[str, str], go_class: str) -> None:
    """
    Create a lollipop plot with customized elements, sized for 180x57mm figure space.
    """
    
    top_clustered_go_terms = top_clustered_go_terms.filter((pl.col('Representative_LOR') != 0) & (pl.col('Class') == go_class)).to_pandas()
    
    # Sort data and create labels
    df_sorted = top_clustered_go_terms.copy()
    df_sorted['Component_Label'] = df_sorted['Representative_Description'] + ' (' + df_sorted['Phylum'] + ')'

    # Sort by Phylum alphabetically and then by LOR within each Phylum (descending)
    df_sorted = df_sorted.sort_values(['Phylum', 'Representative_LOR'], ascending=[True, False])

    # Get cluster size distribution
    cluster_sizes = sorted(df_sorted['Cluster_Size'].unique())
    legend_sizes = [cluster_sizes[0], cluster_sizes[len(cluster_sizes)//4], cluster_sizes[len(cluster_sizes)//2], cluster_sizes[-1]]

    # Adjust dot sizes for the smaller figure
    min_size = 15 
    max_size = 80 
    sizes = np.log10(df_sorted['Cluster_Size'] + 1) * (max_size - min_size) + min_size

    # Create color map based on negative log10 p-value
    p_values = df_sorted['Adjusted_Meta_P_Value']
    colors = -np.log10(p_values)
    cmap = plt.colormaps.get_cmap('flare')
    norm = mcolors.LogNorm(vmin=colors.min(), vmax=colors.max())
    color_values = cmap(norm(colors))

    plt.figure(figsize=(2.24, 18.49), dpi=300)  
    
    unique_phyla = df_sorted['Phylum'].unique()
    for i, phylum in enumerate(unique_phyla):
        phylum_indices = df_sorted[df_sorted['Phylum'] == phylum].index
        if i % 2 == 0:
            plt.axhspan(phylum_indices.min() - 0.5, phylum_indices.max() + 0.5, facecolor='grey', alpha=0.1, zorder=0)

    # Create lollipop stems, heads, and error bars with adjusted line widths
    for idx in range(len(df_sorted)):
        row = df_sorted.iloc[idx]
        plt.hlines(y=idx, xmin=0, xmax=row['Representative_LOR'], color=color_values[idx], linewidth=1) 
        plt.hlines(y=idx, xmin=max(0, row['Lower_CI']), xmax=row['Upper_CI'], color='black', linewidth=0.5) 
        plt.vlines(x=max(0, row['Lower_CI']), ymin=idx - 0.1, ymax=idx + 0.1, color='black', linewidth=1) 
        plt.vlines(x=row['Upper_CI'], ymin=idx - 0.1, ymax=idx + 0.1, color='black', linewidth=1)

    # Scatter plot for lollipop heads
    plt.scatter(df_sorted['Representative_LOR'], range(len(df_sorted)), color=color_values, s=sizes, zorder=3)

    # Customize plot with adjusted font sizes
    cleaned_labels = [re.sub(r"\s\(.*?\)", "", label) for label in df_sorted['Component_Label']]
    final_labels = [go_term_mapping.get(label, label) for label in cleaned_labels]
    plt.yticks(range(len(df_sorted)), final_labels, fontsize=12)  
    plt.xticks(ticks=np.arange(0, max(df_sorted['Upper_CI']) + 1, 2), fontsize=10) 
    plt.xlabel('ln(Enrichment)', fontsize=12) 
    plt.ylabel('')
    plt.grid(axis='x', linestyle='-', alpha=0.7, zorder=0)
    plt.xlim(0, max(df_sorted['Upper_CI']) + 1)
    plt.ylim(-0.5, len(df_sorted['Representative_Description']))

    # Adding Phylum labels to the right side with adjusted font size
    phylum_positions = df_sorted.groupby('Phylum').apply(lambda x: (x.index.min() + x.index.max()) / 2.0)
    for phylum, position in phylum_positions.items():
        plt.text(x=max(df_sorted['Upper_CI']) + 0.6, y=position, 
                s=phylum.replace("Candidatus ", "C. ").replace("Candidate ", "C. "),
                fontsize=10, fontweight='bold', va='center', color='black') 

    sns.despine()
    plt.tight_layout(pad=0.2) 
    plt.savefig(plots / f"{go_class}_lollipop.svg", bbox_inches='tight', pad_inches=0.1) 
    plt.close()

    # Create separate legend plot with adjusted size
    fig_legend = plt.figure(figsize=(1.5, 1.5)) 
    legend_elements = [plt.scatter([], [], s=np.log10(size + 1) * (max_size - min_size) + min_size, 
                                 color='black', alpha=0.7,
                                 label=f'Cluster Size: {size}') for size in legend_sizes]
    leg = fig_legend.legend(handles=legend_elements, title='Cluster Sizes', 
                          loc='center', fontsize=6, title_fontsize=7)
    plt.axis('off')
    fig_legend.savefig(plots / f'{go_class}_cluster_size_legend.svg', 
                      bbox_inches='tight', pad_inches=0.05)
    plt.close()

for go_class in classes:
    create_lollipop_plot(top_clustered_go_terms, go_term_mapping, go_class)
    
filtered_clustered_go_terms = clustered_go_terms.filter((pl.col('Representative_LOR')!=0) & (pl.col("Adjusted_Meta_P_Value")<0.05)).to_pandas()

# Group by 'Phylum' and 'Class', then sum the counts
grouped = filtered_clustered_go_terms.groupby(['Phylum', 'Class'])['Species_Count'].sum().unstack(fill_value=0)

# Reset index 
grouped_data = grouped.reset_index()

# Set Phylum as index 
grouped_data = grouped_data.set_index('Phylum')

# Convert any string numbers to float in the dataframe
for col in grouped_data.columns:
    if col != 'Phylum':  
        grouped_data[col] = pd.to_numeric(grouped_data[col], errors='coerce')

# Define domain colors
domain_colors = {
    'Archaea': '#0072b2',
    'Bacteria': '#e69f00',
    'Eukaryota': '#009e73',
    'Viruses': '#cc79a7'
}

# Define colors for GO term classes
go_colors = {
    'biological_process': '#D193AE',
    'molecular_function': '#CDB497',
    'cellular_component': '#83A7C4'
}

# Create phylum to lineage mapping
phylum_to_lineage = {}
for domain in domain_to_kingdom_to_phylum:
    for kingdom in domain_to_kingdom_to_phylum[domain]:
        phyla = domain_to_kingdom_to_phylum[domain][kingdom]
        for phylum in phyla:
            phylum_to_lineage[phylum] = (domain, kingdom)

# Get all unique phyla 
all_phyla = grouped_data.index.unique()

# Create a dictionary to store domain-wise phyla
domain_wise_phyla = {domain: [] for domain in domain_colors.keys()}

# Calculate total counts for each phylum (sum across all GO term classes)
total_counts = grouped_data.sum(axis=1) 

# Sort phyla into their respective domains with counts
for phylum in all_phyla:
    domain = phylum_to_lineage.get(phylum, (None, None))[0]
    if domain in domain_wise_phyla:
        domain_wise_phyla[domain].append((phylum, total_counts[phylum]))

# Sort phyla within each domain by total counts (descending)
for domain in domain_wise_phyla:
    domain_wise_phyla[domain].sort(key=lambda x: x[1], reverse=True)
    domain_wise_phyla[domain] = [x[0] for x in domain_wise_phyla[domain]]

# Create ordered list of phyla
ordered_phyla = []
for domain in domain_colors.keys():
    ordered_phyla.extend(domain_wise_phyla[domain])

# Reindex grouped data with the new order
grouped_data = grouped_data.reindex(ordered_phyla)

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 20, height_ratios=[0.5, 10], hspace=0.05)

ax_domains = fig.add_subplot(gs[0, :19])
ax = fig.add_subplot(gs[1, :19])

grouped_data.plot(kind='bar', stacked=True, ax=ax, 
                 color=[go_colors[col] for col in grouped_data.columns],
                 legend=False,
                 width=0.9)

plt.xticks(rotation=45, ha='right', fontsize=8)
plt.xlabel('', fontsize=12)
plt.ylabel('Number of enriched GO Terms', fontsize=18)
plt.title('')
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.yscale('log')

ax_domains.axis('off')

current_x = -0.5
for domain in domain_colors.keys():
    domain_phyla = domain_wise_phyla[domain]
    if domain_phyla:
        width = len(domain_phyla)
        rect = plt.Rectangle((current_x, 0), width, 1,
                           color=domain_colors[domain], alpha=0.8)
        ax_domains.add_patch(rect)        
        current_x += width

ax_domains.set_xlim(-0.5, len(all_phyla) - 0.5)
ax_domains.set_ylim(0, 1)

x_labels = [label.replace("Candidatus ", "C. ").replace("Candidate ", "C. ") 
            for label in grouped_data.index]
ax.set_xticklabels(x_labels, rotation=90, ha='center', fontsize=12)

sns.despine()

plt.savefig('stacked_bar_plot.svg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

# Create separate legend figure
figsize_legend = (6, 2)
fig_legend = plt.figure(figsize=figsize_legend)
ax_legend = fig_legend.add_subplot(111)

handles, labels = ax.get_legend_handles_labels()

label_mapping = {
    'biological_process': 'Biological Process',
    'molecular_function': 'Molecular Function',
    'cellular_component': 'Cellular Component'
}

labels = [label_mapping[label] for label in labels]

ax_legend.legend(handles, labels,
               title='GO Term Class',
               title_fontsize=12,
               fontsize=10,
               loc='center',
               ncol=3)

ax_legend.axis('off')

plt.savefig('legend.svg', dpi=300, bbox_inches='tight', pad_inches=0)

plt.close('all')
