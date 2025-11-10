#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE 10000
#define MAX_SEQ_LEN 2000

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.fasta> <output.fasta>\n", argv[0]);
        return 1;
    }

    FILE *input = fopen(argv[1], "r");
    if (!input) {
        perror("Error opening input file");
        return 1;
    }

    FILE *output = fopen(argv[2], "w");
    if (!output) {
        perror("Error opening output file");
        fclose(input);
        return 1;
    }

    char line[MAX_LINE];
    int seq_count = 0;
    int reading_seq = 1;  // Are we still reading this sequence?

    while (fgets(line, sizeof(line), input)) {
        // Header line (starts with '>')
        if (line[0] == '>') {
            // If we were in a sequence, add newline before new header
            if (seq_count > 0) {
                fputc('\n', output);
            }
            seq_count = 0;
            reading_seq = 1;
            fputs(line, output);  // Write header as-is
        }
        // Sequence line
        else if (reading_seq) {
            int line_len = strlen(line);
            // Remove newline if present
            if (line_len > 0 && line[line_len - 1] == '\n') {
                line_len--;
            }

            // Calculate how many characters to write
            int chars_to_write = line_len;
            if (seq_count + line_len > MAX_SEQ_LEN) {
                chars_to_write = MAX_SEQ_LEN - seq_count;
            }

            // Write the appropriate number of characters
            if (chars_to_write > 0) {
                fwrite(line, 1, chars_to_write, output);
                seq_count += chars_to_write;
            }

            // Stop reading this sequence if we've hit the limit
            if (seq_count >= MAX_SEQ_LEN) {
                reading_seq = 0;
            }
        }
        // If reading_seq is 0, we skip remaining lines until next header
    }

    // Add final newline if needed
    if (seq_count > 0) {
        fputc('\n', output);
    }

    fclose(input);
    fclose(output);

    printf("Successfully processed FASTA file.\n");
    printf("Output written to: %s\n", argv[2]);

    return 0;
}