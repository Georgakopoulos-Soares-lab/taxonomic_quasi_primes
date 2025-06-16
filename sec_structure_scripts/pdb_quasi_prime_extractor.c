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
    // Expect 4 arguments: program name, mappings file, input directory, output directory
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <mappings_file> <input_pdb_directory> <output_pdb_directory>\n", argv[0]);
        return 1;
    }
    
    // Assign command-line arguments to variables for clarity
    char *mappings_file_path = argv[1];
    char *input_dir = argv[2];
    char *output_dir = argv[3];

    // Allocate memory to store all the records from the mappings file
    struct UniProtRecord *upr_list = malloc(MAX_RECORDS * sizeof(struct UniProtRecord));
    if (upr_list == NULL) {
        fprintf(stderr, "Memory allocation failed for upr_list\n");
        return 1;
    }

    // Open the mappings file provided as the first argument
    FILE *qp_peptide_loc = fopen(mappings_file_path, "r");
    char line[MAX_LINE_LENGTH];
    int record_count = 0;

    if (qp_peptide_loc != NULL) {
        // Read all mappings into the upr_list array
        while (fgets(line, sizeof(line), qp_peptide_loc)) {
            line[strcspn(line, "\n")] = 0; // Remove newline character

            // Tokenize the line by tab delimiter
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
        fprintf(stderr, "Unable to open %s!\n", mappings_file_path);
        free(upr_list);
        return 1;
    }

    // Use OpenMP to process the records in parallel for speed
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < record_count; i++) {
        char filepath[MAX_PATH];
        char output_filepath[MAX_PATH];
        FILE *record_file = NULL;
        FILE *output_file = NULL;

        // Create the full input filepath using the input directory argument
        snprintf(filepath, sizeof(filepath), "%s/AF-%s-F1-model_v4.pdb", input_dir, upr_list[i].record);
        
        // Create a unique output filepath using the output directory argument
        snprintf(output_filepath, sizeof(output_filepath), "%s/qp_peptide_of_%s_%d_%d.pdb", 
                 output_dir, upr_list[i].record, upr_list[i].start, upr_list[i].end);

        // Open the source PDB file
        record_file = fopen(filepath, "r");
        if (record_file != NULL) {
            // Open the destination file for writing the extracted peptide
            output_file = fopen(output_filepath, "w");
            if (output_file != NULL) {
                // Write a header indicating the extracted range
                fprintf(output_file, "REMARK 220 PEPTIDE RANGE %d-%d\n", upr_list[i].start, upr_list[i].end);

                char record_line[MAX_LINE_LENGTH];
                while (fgets(record_line, sizeof(record_line), record_file)) {
                    // Check if the line is an ATOM record
                    if (strncmp(record_line, "ATOM  ", 6) == 0) {
                        // Extract residue number from fixed columns (23-26)
                        char resSeqStr[5]; // 4 characters + null terminator
                        strncpy(resSeqStr, &record_line[22], 4);
                        resSeqStr[4] = '\0';
                        int residue_number = atoi(resSeqStr);

                        // Check if the residue number is within the specified range
                        if (residue_number >= upr_list[i].start && residue_number <= upr_list[i].end) {
                            fprintf(output_file, "%s", record_line);
                        }
                    }
                }

                // Write an end marker and close the output file
                fprintf(output_file, "END\n");
                fclose(output_file);
            }
            // Close the input file
            fclose(record_file);
        }
    }

    // Free the allocated memory for the record list
    free(upr_list);
    return 0;
}