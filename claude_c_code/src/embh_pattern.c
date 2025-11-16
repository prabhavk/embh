#include "embh_types.h"
#include <stdlib.h>
#include <string.h>

/* Create a pattern with weight and character array */
Pattern* pattern_create(int weight, const uint8_t* characters, int num_taxa) {
    Pattern* p = (Pattern*)malloc(sizeof(Pattern));
    if (!p) return NULL;

    p->weight = weight;
    p->num_taxa = num_taxa;
    p->characters = (uint8_t*)malloc(num_taxa * sizeof(uint8_t));
    if (!p->characters) {
        free(p);
        return NULL;
    }

    if (characters) {
        memcpy(p->characters, characters, num_taxa * sizeof(uint8_t));
    } else {
        memset(p->characters, DNA_GAP, num_taxa * sizeof(uint8_t));
    }

    return p;
}

/* Free pattern memory */
void pattern_destroy(Pattern* p) {
    if (!p) return;
    free(p->characters);
    free(p);
}

/* Create packed storage for patterns - 3-bit encoding */
PackedPatternStorage* packed_storage_create(int num_patterns, int num_taxa) {
    PackedPatternStorage* storage = (PackedPatternStorage*)malloc(sizeof(PackedPatternStorage));
    if (!storage) return NULL;

    storage->num_patterns = num_patterns;
    storage->num_taxa = num_taxa;

    /* Calculate bytes needed: 3 bits per base */
    int total_bits = num_patterns * num_taxa * 3;
    storage->data_size = (total_bits + 7) / 8;

    storage->packed_data = (uint8_t*)calloc(storage->data_size, sizeof(uint8_t));
    if (!storage->packed_data) {
        free(storage);
        return NULL;
    }

    return storage;
}

/* Free packed storage */
void packed_storage_destroy(PackedPatternStorage* storage) {
    if (!storage) return;
    free(storage->packed_data);
    free(storage);
}

/* Get 3-bit base at specific position */
uint8_t packed_storage_get_base(const PackedPatternStorage* storage, int pattern_idx, int taxon_idx) {
    if (!storage || !storage->packed_data) return DNA_GAP;

    int bit_position = (pattern_idx * storage->num_taxa + taxon_idx) * 3;
    int byte_idx = bit_position / 8;
    int bit_offset = bit_position % 8;

    if ((size_t)byte_idx >= storage->data_size) return DNA_GAP;

    if (bit_offset <= 5) {
        /* All 3 bits in same byte */
        return (storage->packed_data[byte_idx] >> bit_offset) & 0x07;
    } else {
        /* Spans two bytes */
        int bits_in_first = 8 - bit_offset;
        uint8_t low_bits = (storage->packed_data[byte_idx] >> bit_offset);

        if ((size_t)(byte_idx + 1) < storage->data_size) {
            int bits_in_second = 3 - bits_in_first;
            uint8_t high_bits = (storage->packed_data[byte_idx + 1] & ((1 << bits_in_second) - 1));
            return low_bits | (high_bits << bits_in_first);
        }
        return low_bits & 0x07;
    }
}

/* Set 3-bit base at specific position */
void packed_storage_set_base(PackedPatternStorage* storage, int pattern_idx, int taxon_idx, uint8_t base) {
    if (!storage || !storage->packed_data) return;

    int bit_position = (pattern_idx * storage->num_taxa + taxon_idx) * 3;
    int byte_idx = bit_position / 8;
    int bit_offset = bit_position % 8;

    if ((size_t)byte_idx >= storage->data_size) return;

    /* Ensure base is only 3 bits */
    base &= 0x07;

    if (bit_offset <= 5) {
        /* All 3 bits in same byte */
        uint8_t mask = ~(0x07 << bit_offset);
        storage->packed_data[byte_idx] = (storage->packed_data[byte_idx] & mask) | (base << bit_offset);
    } else {
        /* Spans two bytes */
        int bits_in_first = 8 - bit_offset;
        int bits_in_second = 3 - bits_in_first;

        uint8_t mask1 = ~(((1 << bits_in_first) - 1) << bit_offset);
        storage->packed_data[byte_idx] = (storage->packed_data[byte_idx] & mask1) | (base << bit_offset);

        if ((size_t)(byte_idx + 1) < storage->data_size) {
            uint8_t mask2 = ~((1 << bits_in_second) - 1);
            storage->packed_data[byte_idx + 1] = (storage->packed_data[byte_idx + 1] & mask2) | (base >> bits_in_first);
        }
    }
}

/* Store entire pattern from array */
void packed_storage_store_pattern(PackedPatternStorage* storage, int pattern_idx, const uint8_t* pattern, int pattern_size) {
    if (!storage || !pattern) return;

    int n = (pattern_size < storage->num_taxa) ? pattern_size : storage->num_taxa;
    for (int i = 0; i < n; i++) {
        packed_storage_set_base(storage, pattern_idx, i, pattern[i]);
    }
}

/* Get entire pattern into array */
void packed_storage_get_pattern(const PackedPatternStorage* storage, int pattern_idx, uint8_t* pattern_out) {
    if (!storage || !pattern_out) return;

    for (int i = 0; i < storage->num_taxa; i++) {
        pattern_out[i] = packed_storage_get_base(storage, pattern_idx, i);
    }
}

/* Get memory usage in bytes */
size_t packed_storage_get_memory_bytes(const PackedPatternStorage* storage) {
    if (!storage) return 0;
    return storage->data_size;
}

/* Get number of patterns */
int packed_storage_get_num_patterns(const PackedPatternStorage* storage) {
    if (!storage) return 0;
    return storage->num_patterns;
}

/* Get number of taxa */
int packed_storage_get_num_taxa(const PackedPatternStorage* storage) {
    if (!storage) return 0;
    return storage->num_taxa;
}
