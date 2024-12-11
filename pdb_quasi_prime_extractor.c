#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_PATH 260
#define MAX_RECORDS 6000000
#define MAX_LINE_LENGTH 256

struct UniProtRecord {
    char record[30];
    int start;
    int end;
};

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <mappings_file>\n", argv[0]);
        return 1;
    }

    struct UniProtRecord *upr_list = malloc(MAX_RECORDS * sizeof(struct UniProtRecord));
    if (upr_list == NULL) {
        fprintf(stderr, "Memory allocation failed for upr_list\n");
        return 1;
    }

    FILE *qp_peptide_loc = fopen(argv[1], "r");
    char line[MAX_LINE_LENGTH];
    int record_count = 0;

    if (qp_peptide_loc != NULL) {
        // Read all mappings into upr_list
        while (fgets(line, sizeof(line), qp_peptide_loc)) {
            line[strcspn(line, "\n")] = 0; // Remove newline character

            char *token = strtok(line, "\t");
            if (token != NULL) {
                strncpy(upr_list[record_count].record, token, sizeof(upr_list[record_count].record) - 1);
                upr_list[record_count].record[sizeof(upr_list[record_count].record) - 1] = '\0';

                token = strtok(NULL, "\t");
                if (token != NULL) {
                    upr_list[record_count].start = atoi(token);

                    token = strtok(NULL, "\t");
                    if (token != NULL) {
                        upr_list[record_count].end = atoi(token);
                        record_count++;
                        if (record_count >= MAX_RECORDS) {
                            fprintf(stderr, "Exceeded maximum number of records (%d)\n", MAX_RECORDS);
                            break;
                        }
                    }
                }
            }
        }
        fclose(qp_peptide_loc);
    } else {
        fprintf(stderr, "Unable to open %s!\n", argv[1]);
        free(upr_list);
        return 1;
    }

    // Parallel processing using OpenMP
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < record_count; i++) {
        char filepath[MAX_PATH];
        char output_filepath[MAX_PATH];
        FILE *record_file = NULL;
        FILE *output_file = NULL;

        // Create input filepath
        snprintf(filepath, sizeof(filepath), "/storage/group/izg5139/default/lefteris/multi_species_structural_analysis_files/model_and_health_alphafold_pdbs/AF-%s-F1-model_v4.pdb", upr_list[i].record);
        // Create output filepath with unique name for each range
        snprintf(output_filepath, sizeof(output_filepath), "/storage/group/izg5139/default/lefteris/multi_species_structural_analysis_files/model_and_health_qp_peptide_extractions/qp_peptide_of_%s_%d_%d.pdb", 
                 upr_list[i].record, upr_list[i].start, upr_list[i].end);

        // Open the input file
        record_file = fopen(filepath, "r");
        if (record_file != NULL) {
            // Open output file in write mode (creates a new file each time)
            output_file = fopen(output_filepath, "w");
            if (output_file != NULL) {
                // Write a header for this range
                fprintf(output_file, "REMARK 220 PEPTIDE RANGE %d-%d\n", upr_list[i].start, upr_list[i].end);

                char record_line[MAX_LINE_LENGTH];
                while (fgets(record_line, sizeof(record_line), record_file)) {
                    // Check if the line starts with "ATOM"
                    if (strncmp(record_line, "ATOM  ", 6) == 0) {
                        // Extract residue number from fixed columns (23-26)
                        char resSeqStr[5]; // 4 characters + null terminator
                        strncpy(resSeqStr, &record_line[22], 4);
                        resSeqStr[4] = '\0';
                        int residue_number = atoi(resSeqStr);

                        // Check if the residue number is within the range
                        if (residue_number >= upr_list[i].start && residue_number <= upr_list[i].end) {
                            fprintf(output_file, "%s", record_line);
                        }
                    }
                }

                // Write an end marker
                fprintf(output_file, "END\n");

                fclose(output_file);
            }
            fclose(record_file);
        }
    }

    free(upr_list);
    return 0;
}