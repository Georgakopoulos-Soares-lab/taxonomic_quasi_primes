#!/bin/bash

# Check if a directory path is provided
if [ $# -eq 0 ]; then
    echo "Error: No directory path provided"
    echo "Usage: $0 <path_to_pdb_files_directory>"
    exit 1
fi

# Get the input directory path
pdb_directory="$1"

# Check if the directory exists
if [ ! -d "$pdb_directory" ]; then
    echo "Error: Directory '$pdb_directory' not found"
    exit 1
fi

# Create output file with header
echo -e "PDB_ID\tSequence\tSecondary_structure\tAlpha_percent\t3_10_helix_percent\tPi_helix_percent\tExtended_percent\tIsolated_bridge_percent\tTurn_percent\tCoil_percent\tPredominant_structure" > "secondary_structures.txt"

# Function to process a single PDB file
process_pdb() {
    pdb_file="$1"
    pdb_id=$(basename "$pdb_file" | sed -n 's/.*_of_\(.*\)\.pdb/\1/p')
    
    stride "$pdb_file" | awk -v pdb="$pdb_id" '
    /^SEQ/ {
        for (i=3; i<=NF; i++) {
            if ($i ~ /^[A-Z]+$/) {
                seq = $i
                break
            }
        }
    }
    /^ASG/ {
        ss = ss $6
        if ($6 == "H") alpha++
        else if ($6 == "G") helix3_10++
        else if ($6 == "I") pi_helix++
        else if ($6 == "E") extended++
        else if ($6 == "B" || $6 == "b") isolated_bridge++
        else if ($6 == "T") turn++
        else if ($6 == "C") coil++
        total++
    }
    END {
        if (seq != "" && ss != "") {
            alpha_pct = alpha / total * 100
            helix3_10_pct = helix3_10 / total * 100
            pi_helix_pct = pi_helix / total * 100
            extended_pct = extended / total * 100
            isolated_bridge_pct = isolated_bridge / total * 100
            turn_pct = turn / total * 100
            coil_pct = coil / total * 100
            max_pct = alpha_pct
            predominant = "Alpha helix"
            if (helix3_10_pct > max_pct) { max_pct = helix3_10_pct; predominant = "3-10 helix" }
            if (pi_helix_pct > max_pct) { max_pct = pi_helix_pct; predominant = "Pi helix" }
            if (extended_pct > max_pct) { max_pct = extended_pct; predominant = "Extended" }
            if (isolated_bridge_pct > max_pct) { max_pct = isolated_bridge_pct; predominant = "Isolated bridge" }
            if (turn_pct > max_pct) { max_pct = turn_pct; predominant = "Turn" }
            if (coil_pct > max_pct) { max_pct = coil_pct; predominant = "Coil" }
            printf "%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s\n",
            pdb, seq, ss, alpha_pct, helix3_10_pct, pi_helix_pct, extended_pct, isolated_bridge_pct, turn_pct, coil_pct, predominant
        } else {
            print pdb, (seq != "" ? seq : "N/A"), (ss != "" ? ss : "N/A"), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
        }
    }
    '
}

export -f process_pdb

# Use GNU Parallel to process files in parallel
find "$pdb_directory" -name "*.pdb" | parallel  process_pdb {} >> "secondary_structures.txt"

echo "Output saved to secondary_structures.txt"