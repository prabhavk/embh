#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "embh_types.h"

int test_matrix_identity(void) {
    printf("Test: matrix_identity_4x4\n");

    double M[16];
    matrix_identity_4x4(M);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (fabs(M[i * 4 + j] - expected) > 1e-15) {
                printf("  FAIL: M[%d][%d] = %f, expected %f\n", i, j, M[i * 4 + j], expected);
                return 1;
            }
        }
    }

    printf("  PASS\n");
    return 0;
}

int test_matrix_copy(void) {
    printf("Test: matrix_copy_4x4\n");

    double src[16];
    for (int i = 0; i < 16; i++) {
        src[i] = (double)(i + 1);
    }

    double dst[16];
    matrix_copy_4x4(src, dst);

    for (int i = 0; i < 16; i++) {
        if (fabs(dst[i] - src[i]) > 1e-15) {
            printf("  FAIL: dst[%d] = %f, expected %f\n", i, dst[i], src[i]);
            return 1;
        }
    }

    printf("  PASS\n");
    return 0;
}

int test_matrix_transpose(void) {
    printf("Test: matrix_transpose_4x4\n");

    /* Create a known matrix:
     * A = [1  2  3  4 ]
     *     [5  6  7  8 ]
     *     [9  10 11 12]
     *     [13 14 15 16]
     */
    double A[16];
    for (int i = 0; i < 16; i++) {
        A[i] = (double)(i + 1);
    }

    double AT[16];
    matrix_transpose_4x4(A, AT);

    /* Check that AT[j][i] = A[i][j] */
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            double expected = A[i * 4 + j];
            double actual = AT[j * 4 + i];
            if (fabs(actual - expected) > 1e-15) {
                printf("  FAIL: AT[%d][%d] = %f, expected A[%d][%d] = %f\n",
                       j, i, actual, i, j, expected);
                return 1;
            }
        }
    }

    printf("  PASS\n");
    return 0;
}

int test_matrix_multiply(void) {
    printf("Test: matrix_multiply_4x4\n");

    /* Test: I * A = A */
    double I[16];
    matrix_identity_4x4(I);

    double A[16] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };

    double R[16];
    matrix_multiply_4x4(I, A, R);

    for (int i = 0; i < 16; i++) {
        if (fabs(R[i] - A[i]) > 1e-15) {
            printf("  FAIL: I*A != A at index %d\n", i);
            return 1;
        }
    }

    /* Test: specific multiplication */
    /* B = [2 0 0 0]    C = [1 1 1 1]
     *     [0 3 0 0]        [1 1 1 1]
     *     [0 0 4 0]        [1 1 1 1]
     *     [0 0 0 5]        [1 1 1 1]
     *
     * B*C should give:
     *     [2 2 2 2]
     *     [3 3 3 3]
     *     [4 4 4 4]
     *     [5 5 5 5]
     */
    double B[16] = {
        2.0, 0.0, 0.0, 0.0,
        0.0, 3.0, 0.0, 0.0,
        0.0, 0.0, 4.0, 0.0,
        0.0, 0.0, 0.0, 5.0
    };

    double C[16] = {
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0
    };

    matrix_multiply_4x4(B, C, R);

    double expected[16] = {
        2.0, 2.0, 2.0, 2.0,
        3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0,
        5.0, 5.0, 5.0, 5.0
    };

    for (int i = 0; i < 16; i++) {
        if (fabs(R[i] - expected[i]) > 1e-15) {
            printf("  FAIL: B*C incorrect at index %d: got %f, expected %f\n",
                   i, R[i], expected[i]);
            return 1;
        }
    }

    printf("  PASS\n");
    return 0;
}

int test_string_starts_with(void) {
    printf("Test: string_starts_with\n");

    if (!string_starts_with("hello world", "hello")) {
        printf("  FAIL: 'hello world' should start with 'hello'\n");
        return 1;
    }

    if (!string_starts_with("ABC", "ABC")) {
        printf("  FAIL: 'ABC' should start with 'ABC'\n");
        return 1;
    }

    if (string_starts_with("hello", "hello world")) {
        printf("  FAIL: 'hello' should not start with 'hello world'\n");
        return 1;
    }

    if (string_starts_with("xyz", "abc")) {
        printf("  FAIL: 'xyz' should not start with 'abc'\n");
        return 1;
    }

    if (!string_starts_with("", "")) {
        printf("  FAIL: empty string should start with empty prefix\n");
        return 1;
    }

    if (string_starts_with(NULL, "test")) {
        printf("  FAIL: NULL should return false\n");
        return 1;
    }

    printf("  PASS\n");
    return 0;
}

int test_split_whitespace(void) {
    printf("Test: split_whitespace\n");

    int num_tokens;
    char** tokens;

    /* Test basic splitting */
    tokens = split_whitespace("hello world test", &num_tokens);
    if (num_tokens != 3) {
        printf("  FAIL: Expected 3 tokens, got %d\n", num_tokens);
        free_string_array(tokens, num_tokens);
        return 1;
    }

    if (strcmp(tokens[0], "hello") != 0 || strcmp(tokens[1], "world") != 0 ||
        strcmp(tokens[2], "test") != 0) {
        printf("  FAIL: Token content mismatch\n");
        free_string_array(tokens, num_tokens);
        return 1;
    }
    free_string_array(tokens, num_tokens);

    /* Test multiple spaces */
    tokens = split_whitespace("  one   two  ", &num_tokens);
    if (num_tokens != 2) {
        printf("  FAIL: Expected 2 tokens with multiple spaces, got %d\n", num_tokens);
        free_string_array(tokens, num_tokens);
        return 1;
    }

    if (strcmp(tokens[0], "one") != 0 || strcmp(tokens[1], "two") != 0) {
        printf("  FAIL: Token content mismatch with multiple spaces\n");
        free_string_array(tokens, num_tokens);
        return 1;
    }
    free_string_array(tokens, num_tokens);

    /* Test tabs and newlines */
    tokens = split_whitespace("A\tB\nC", &num_tokens);
    if (num_tokens != 3) {
        printf("  FAIL: Expected 3 tokens with tabs/newlines, got %d\n", num_tokens);
        free_string_array(tokens, num_tokens);
        return 1;
    }
    free_string_array(tokens, num_tokens);

    /* Test empty string */
    tokens = split_whitespace("", &num_tokens);
    if (num_tokens != 0 || tokens != NULL) {
        printf("  FAIL: Empty string should give 0 tokens\n");
        free_string_array(tokens, num_tokens);
        return 1;
    }

    /* Test only whitespace */
    tokens = split_whitespace("   \t\n  ", &num_tokens);
    if (num_tokens != 0) {
        printf("  FAIL: Whitespace-only string should give 0 tokens\n");
        free_string_array(tokens, num_tokens);
        return 1;
    }

    printf("  PASS\n");
    return 0;
}

int test_dna_conversion(void) {
    printf("Test: convert_dna_to_index and convert_index_to_dna\n");

    /* Test DNA to index */
    if (convert_dna_to_index('A') != DNA_A) {
        printf("  FAIL: 'A' should map to DNA_A\n");
        return 1;
    }
    if (convert_dna_to_index('a') != DNA_A) {
        printf("  FAIL: 'a' should map to DNA_A\n");
        return 1;
    }
    if (convert_dna_to_index('C') != DNA_C) {
        printf("  FAIL: 'C' should map to DNA_C\n");
        return 1;
    }
    if (convert_dna_to_index('G') != DNA_G) {
        printf("  FAIL: 'G' should map to DNA_G\n");
        return 1;
    }
    if (convert_dna_to_index('T') != DNA_T) {
        printf("  FAIL: 'T' should map to DNA_T\n");
        return 1;
    }
    if (convert_dna_to_index('-') != DNA_GAP) {
        printf("  FAIL: '-' should map to DNA_GAP\n");
        return 1;
    }

    /* Test index to DNA */
    if (convert_index_to_dna(DNA_A) != 'A') {
        printf("  FAIL: DNA_A should map to 'A'\n");
        return 1;
    }
    if (convert_index_to_dna(DNA_C) != 'C') {
        printf("  FAIL: DNA_C should map to 'C'\n");
        return 1;
    }
    if (convert_index_to_dna(DNA_G) != 'G') {
        printf("  FAIL: DNA_G should map to 'G'\n");
        return 1;
    }
    if (convert_index_to_dna(DNA_T) != 'T') {
        printf("  FAIL: DNA_T should map to 'T'\n");
        return 1;
    }
    if (convert_index_to_dna(DNA_GAP) != '-') {
        printf("  FAIL: DNA_GAP should map to '-'\n");
        return 1;
    }

    printf("  PASS\n");
    return 0;
}

int test_gap_proportion(void) {
    printf("Test: gap_proportion_in_pattern\n");

    int pattern1[] = {DNA_A, DNA_C, DNA_GAP, DNA_T};
    double prop1 = gap_proportion_in_pattern(pattern1, 4);
    if (fabs(prop1 - 0.25) > 1e-15) {
        printf("  FAIL: Expected 0.25, got %f\n", prop1);
        return 1;
    }

    int pattern2[] = {DNA_GAP, DNA_GAP, DNA_GAP, DNA_GAP};
    double prop2 = gap_proportion_in_pattern(pattern2, 4);
    if (fabs(prop2 - 1.0) > 1e-15) {
        printf("  FAIL: Expected 1.0, got %f\n", prop2);
        return 1;
    }

    int pattern3[] = {DNA_A, DNA_C, DNA_G, DNA_T};
    double prop3 = gap_proportion_in_pattern(pattern3, 4);
    if (fabs(prop3 - 0.0) > 1e-15) {
        printf("  FAIL: Expected 0.0, got %f\n", prop3);
        return 1;
    }

    printf("  PASS\n");
    return 0;
}

int test_unique_non_gap_count(void) {
    printf("Test: unique_non_gap_count_in_pattern\n");

    int pattern1[] = {DNA_A, DNA_A, DNA_A, DNA_A};
    int count1 = unique_non_gap_count_in_pattern(pattern1, 4);
    if (count1 != 1) {
        printf("  FAIL: Expected 1 unique base, got %d\n", count1);
        return 1;
    }

    int pattern2[] = {DNA_A, DNA_C, DNA_G, DNA_T};
    int count2 = unique_non_gap_count_in_pattern(pattern2, 4);
    if (count2 != 4) {
        printf("  FAIL: Expected 4 unique bases, got %d\n", count2);
        return 1;
    }

    int pattern3[] = {DNA_A, DNA_GAP, DNA_A, DNA_GAP};
    int count3 = unique_non_gap_count_in_pattern(pattern3, 4);
    if (count3 != 1) {
        printf("  FAIL: Expected 1 unique base (gaps excluded), got %d\n", count3);
        return 1;
    }

    int pattern4[] = {DNA_GAP, DNA_GAP, DNA_GAP, DNA_GAP};
    int count4 = unique_non_gap_count_in_pattern(pattern4, 4);
    if (count4 != 0) {
        printf("  FAIL: Expected 0 unique bases (all gaps), got %d\n", count4);
        return 1;
    }

    int pattern5[] = {DNA_A, DNA_C, DNA_GAP, DNA_T};
    int count5 = unique_non_gap_count_in_pattern(pattern5, 4);
    if (count5 != 3) {
        printf("  FAIL: Expected 3 unique bases, got %d\n", count5);
        return 1;
    }

    printf("  PASS\n");
    return 0;
}

int main(void) {
    printf("=== Utility Function Unit Tests ===\n\n");

    int failures = 0;

    failures += test_matrix_identity();
    failures += test_matrix_copy();
    failures += test_matrix_transpose();
    failures += test_matrix_multiply();
    failures += test_string_starts_with();
    failures += test_split_whitespace();
    failures += test_dna_conversion();
    failures += test_gap_proportion();
    failures += test_unique_non_gap_count();

    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    return failures;
}
