#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "embh_types.h"

int test_pattern_create(void) {
    printf("Test: pattern_create and pattern_destroy\n");

    uint8_t chars[] = {DNA_A, DNA_C, DNA_G, DNA_T, DNA_GAP};
    Pattern* p = pattern_create(10, chars, 5);

    if (!p) {
        printf("  FAIL: pattern_create returned NULL\n");
        return 1;
    }

    if (p->weight != 10) {
        printf("  FAIL: weight is %d, expected 10\n", p->weight);
        pattern_destroy(p);
        return 1;
    }

    if (p->num_taxa != 5) {
        printf("  FAIL: num_taxa is %d, expected 5\n", p->num_taxa);
        pattern_destroy(p);
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        if (p->characters[i] != chars[i]) {
            printf("  FAIL: character[%d] is %d, expected %d\n", i, p->characters[i], chars[i]);
            pattern_destroy(p);
            return 1;
        }
    }

    pattern_destroy(p);
    printf("  PASS\n");
    return 0;
}

int test_packed_storage_basic(void) {
    printf("Test: packed_storage_create and destroy\n");

    PackedPatternStorage* storage = packed_storage_create(10, 5);
    if (!storage) {
        printf("  FAIL: packed_storage_create returned NULL\n");
        return 1;
    }

    if (storage->num_patterns != 10) {
        printf("  FAIL: num_patterns is %d, expected 10\n", storage->num_patterns);
        packed_storage_destroy(storage);
        return 1;
    }

    if (storage->num_taxa != 5) {
        printf("  FAIL: num_taxa is %d, expected 5\n", storage->num_taxa);
        packed_storage_destroy(storage);
        return 1;
    }

    /* Memory should be ceil(10 * 5 * 3 / 8) = ceil(150/8) = 19 bytes */
    size_t expected_bytes = (10 * 5 * 3 + 7) / 8;  /* 19 bytes */
    size_t actual_bytes = packed_storage_get_memory_bytes(storage);

    if (actual_bytes != expected_bytes) {
        printf("  FAIL: memory bytes is %zu, expected %zu\n", actual_bytes, expected_bytes);
        packed_storage_destroy(storage);
        return 1;
    }

    printf("  Memory usage: %zu bytes (expected %zu)\n", actual_bytes, expected_bytes);

    packed_storage_destroy(storage);
    printf("  PASS\n");
    return 0;
}

int test_packed_storage_set_get(void) {
    printf("Test: packed_storage_set_base and get_base\n");

    PackedPatternStorage* storage = packed_storage_create(10, 5);
    if (!storage) {
        printf("  FAIL: packed_storage_create returned NULL\n");
        return 1;
    }

    /* Test all base values */
    uint8_t test_bases[] = {DNA_A, DNA_C, DNA_G, DNA_T, DNA_GAP};

    /* Set pattern 0 */
    for (int i = 0; i < 5; i++) {
        packed_storage_set_base(storage, 0, i, test_bases[i]);
    }

    /* Verify pattern 0 */
    for (int i = 0; i < 5; i++) {
        uint8_t got = packed_storage_get_base(storage, 0, i);
        if (got != test_bases[i]) {
            printf("  FAIL: pattern 0, taxon %d: got %d, expected %d\n", i, got, test_bases[i]);
            packed_storage_destroy(storage);
            return 1;
        }
    }

    /* Test pattern at different positions to check bit boundary handling */
    for (int p = 0; p < 10; p++) {
        for (int t = 0; t < 5; t++) {
            uint8_t val = (p + t) % 5;  /* 0-4 */
            packed_storage_set_base(storage, p, t, val);
        }
    }

    /* Verify all patterns */
    int errors = 0;
    for (int p = 0; p < 10; p++) {
        for (int t = 0; t < 5; t++) {
            uint8_t expected = (p + t) % 5;
            uint8_t got = packed_storage_get_base(storage, p, t);
            if (got != expected) {
                printf("  FAIL: pattern %d, taxon %d: got %d, expected %d\n", p, t, got, expected);
                errors++;
            }
        }
    }

    packed_storage_destroy(storage);

    if (errors > 0) {
        printf("  FAIL: %d errors\n", errors);
        return 1;
    }

    printf("  PASS\n");
    return 0;
}

int test_packed_storage_patterns(void) {
    printf("Test: packed_storage_store_pattern and get_pattern\n");

    PackedPatternStorage* storage = packed_storage_create(3, 8);
    if (!storage) {
        printf("  FAIL: packed_storage_create returned NULL\n");
        return 1;
    }

    /* Create test patterns */
    uint8_t pattern0[] = {0, 1, 2, 3, 4, 0, 1, 2};
    uint8_t pattern1[] = {3, 2, 1, 0, 4, 3, 2, 1};
    uint8_t pattern2[] = {4, 4, 4, 4, 0, 1, 2, 3};

    packed_storage_store_pattern(storage, 0, pattern0, 8);
    packed_storage_store_pattern(storage, 1, pattern1, 8);
    packed_storage_store_pattern(storage, 2, pattern2, 8);

    /* Retrieve and verify */
    uint8_t retrieved[8];

    packed_storage_get_pattern(storage, 0, retrieved);
    if (memcmp(retrieved, pattern0, 8) != 0) {
        printf("  FAIL: pattern 0 mismatch\n");
        packed_storage_destroy(storage);
        return 1;
    }

    packed_storage_get_pattern(storage, 1, retrieved);
    if (memcmp(retrieved, pattern1, 8) != 0) {
        printf("  FAIL: pattern 1 mismatch\n");
        packed_storage_destroy(storage);
        return 1;
    }

    packed_storage_get_pattern(storage, 2, retrieved);
    if (memcmp(retrieved, pattern2, 8) != 0) {
        printf("  FAIL: pattern 2 mismatch\n");
        packed_storage_destroy(storage);
        return 1;
    }

    packed_storage_destroy(storage);
    printf("  PASS\n");
    return 0;
}

int test_memory_efficiency(void) {
    printf("Test: Memory efficiency (3-bit vs int storage)\n");

    /* Simulate 648 patterns, 38 taxa (from baseline) */
    int num_patterns = 648;
    int num_taxa = 38;

    PackedPatternStorage* storage = packed_storage_create(num_patterns, num_taxa);
    if (!storage) {
        printf("  FAIL: packed_storage_create returned NULL\n");
        return 1;
    }

    size_t packed_bytes = packed_storage_get_memory_bytes(storage);
    size_t int_bytes = num_patterns * num_taxa * sizeof(int);
    double savings = 100.0 * (1.0 - (double)packed_bytes / int_bytes);

    printf("  Patterns: %d, Taxa: %d\n", num_patterns, num_taxa);
    printf("  Packed storage: %zu bytes\n", packed_bytes);
    printf("  Int storage: %zu bytes\n", int_bytes);
    printf("  Memory savings: %.2f%%\n", savings);

    /* Expected: ceil(648 * 38 * 3 / 8) = ceil(73872/8) = 9234 bytes */
    size_t expected_bytes = (num_patterns * num_taxa * 3 + 7) / 8;
    if (packed_bytes != expected_bytes) {
        printf("  FAIL: expected %zu bytes, got %zu\n", expected_bytes, packed_bytes);
        packed_storage_destroy(storage);
        return 1;
    }

    /* Should match baseline: 9234 bytes, ~90.625% savings */
    if (savings < 90.0) {
        printf("  FAIL: memory savings too low (expected >90%%)\n");
        packed_storage_destroy(storage);
        return 1;
    }

    packed_storage_destroy(storage);
    printf("  PASS\n");
    return 0;
}

int main(void) {
    printf("=== PackedPatternStorage Unit Tests ===\n\n");

    int failures = 0;

    failures += test_pattern_create();
    failures += test_packed_storage_basic();
    failures += test_packed_storage_set_get();
    failures += test_packed_storage_patterns();
    failures += test_memory_efficiency();

    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests PASSED\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    return failures;
}
