#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "embh_types.h"

/*
 * Matrix Operations (4x4 row-major)
 * All matrices are stored as double[16] in row-major order
 * Index [i][j] = src[i*4 + j]
 */

void matrix_transpose_4x4(const double* src, double* dst) {
    /* Transpose: dst[j][i] = src[i][j] */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            dst[j * 4 + i] = src[i * 4 + j];
        }
    }
}

void matrix_multiply_4x4(const double* A, const double* B, double* R) {
    /* R = A * B (standard matrix multiplication) */
    /* R[i][j] = sum_k A[i][k] * B[k][j] */
    double temp[16];  /* Temporary storage in case R aliases A or B */

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double sum = 0.0;
            for (int k = 0; k < 4; k++) {
                sum += A[i * 4 + k] * B[k * 4 + j];
            }
            temp[i * 4 + j] = sum;
        }
    }

    /* Copy result */
    memcpy(R, temp, 16 * sizeof(double));
}

void matrix_identity_4x4(double* M) {
    /* Set M to identity matrix */
    memset(M, 0, 16 * sizeof(double));
    M[0] = 1.0;   /* M[0][0] */
    M[5] = 1.0;   /* M[1][1] */
    M[10] = 1.0;  /* M[2][2] */
    M[15] = 1.0;  /* M[3][3] */
}

void matrix_copy_4x4(const double* src, double* dst) {
    memcpy(dst, src, 16 * sizeof(double));
}

/*
 * String Operations
 */

bool string_starts_with(const char* str, const char* prefix) {
    if (!str || !prefix) return false;
    size_t prefix_len = strlen(prefix);
    if (strlen(str) < prefix_len) return false;
    return strncmp(str, prefix, prefix_len) == 0;
}

char** split_whitespace(const char* str, int* num_tokens) {
    if (!str || !num_tokens) {
        if (num_tokens) *num_tokens = 0;
        return NULL;
    }

    /* First pass: count tokens */
    const char* p = str;
    int count = 0;
    bool in_token = false;

    while (*p) {
        if (isspace((unsigned char)*p)) {
            in_token = false;
        } else {
            if (!in_token) {
                count++;
                in_token = true;
            }
        }
        p++;
    }

    if (count == 0) {
        *num_tokens = 0;
        return NULL;
    }

    /* Allocate array of string pointers */
    char** tokens = (char**)malloc(count * sizeof(char*));
    if (!tokens) {
        *num_tokens = 0;
        return NULL;
    }

    /* Second pass: extract tokens */
    p = str;
    int token_idx = 0;
    const char* token_start = NULL;
    in_token = false;

    while (*p) {
        if (isspace((unsigned char)*p)) {
            if (in_token) {
                /* End of token */
                size_t token_len = p - token_start;
                tokens[token_idx] = (char*)malloc(token_len + 1);
                if (!tokens[token_idx]) {
                    /* Cleanup on failure */
                    for (int i = 0; i < token_idx; i++) {
                        free(tokens[i]);
                    }
                    free(tokens);
                    *num_tokens = 0;
                    return NULL;
                }
                strncpy(tokens[token_idx], token_start, token_len);
                tokens[token_idx][token_len] = '\0';
                token_idx++;
                in_token = false;
            }
        } else {
            if (!in_token) {
                token_start = p;
                in_token = true;
            }
        }
        p++;
    }

    /* Handle last token if not followed by whitespace */
    if (in_token) {
        size_t token_len = p - token_start;
        tokens[token_idx] = (char*)malloc(token_len + 1);
        if (!tokens[token_idx]) {
            for (int i = 0; i < token_idx; i++) {
                free(tokens[i]);
            }
            free(tokens);
            *num_tokens = 0;
            return NULL;
        }
        strncpy(tokens[token_idx], token_start, token_len);
        tokens[token_idx][token_len] = '\0';
    }

    *num_tokens = count;
    return tokens;
}

void free_string_array(char** arr, int num_tokens) {
    if (!arr) return;
    for (int i = 0; i < num_tokens; i++) {
        free(arr[i]);
    }
    free(arr);
}

/*
 * DNA Conversion Functions
 */

int convert_dna_to_index(char dna) {
    switch (dna) {
        case 'A': case 'a': return DNA_A;
        case 'C': case 'c': return DNA_C;
        case 'G': case 'g': return DNA_G;
        case 'T': case 't': return DNA_T;
        case '-': return DNA_GAP;
        default: return DNA_GAP;  /* Unknown treated as gap */
    }
}

char convert_index_to_dna(int index) {
    switch (index) {
        case DNA_A: return 'A';
        case DNA_C: return 'C';
        case DNA_G: return 'G';
        case DNA_T: return 'T';
        case DNA_GAP: return '-';
        default: return '?';
    }
}

double gap_proportion_in_pattern(const int* pattern, int length) {
    if (!pattern || length <= 0) return 0.0;

    int gap_count = 0;
    for (int i = 0; i < length; i++) {
        if (pattern[i] == DNA_GAP) {
            gap_count++;
        }
    }

    return (double)gap_count / (double)length;
}

int unique_non_gap_count_in_pattern(const int* pattern, int length) {
    if (!pattern || length <= 0) return 0;

    bool seen[NUM_BASES] = {false, false, false, false};
    int count = 0;

    for (int i = 0; i < length; i++) {
        int base = pattern[i];
        if (base >= 0 && base < NUM_BASES && !seen[base]) {
            seen[base] = true;
            count++;
        }
    }

    return count;
}
