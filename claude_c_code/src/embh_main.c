#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "embh_manager.h"

static void print_usage(const char* program_name) {
    fprintf(stderr, "Usage: %s [options]\n", program_name);
    fprintf(stderr, "Required options:\n");
    fprintf(stderr, "  -e <file>  Edge list file (tree topology)\n");
    fprintf(stderr, "  -p <file>  Pattern file\n");
    fprintf(stderr, "  -x <file>  Taxon order file\n");
    fprintf(stderr, "  -b <file>  Base composition file\n");
    fprintf(stderr, "  -o <name>  Root vertex name for optimization\n");
    fprintf(stderr, "  -c <name>  Root vertex name for check/comparison\n");
    fprintf(stderr, "\nOptional:\n");
    fprintf(stderr, "  -s <file>  Cache specification file (for selective subtree caching)\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s -e data/tree.edgelist -p data/patterns.pat \\\n", program_name);
    fprintf(stderr, "     -x data/taxon_order.txt -b data/basecomp.txt -o h_0 -c h_5\n");
    fprintf(stderr, "\nWith pre-computed cache:\n");
    fprintf(stderr, "  %s -e data/tree.edgelist -p data/patterns.pat \\\n", program_name);
    fprintf(stderr, "     -x data/taxon_order.txt -b data/basecomp.txt -o h_0 -c h_5 \\\n");
    fprintf(stderr, "     -s data/cache_spec.cache_spec\n");
}

int main(int argc, char* argv[]) {
    char* edge_list_file = NULL;
    char* pattern_file = NULL;
    char* taxon_order_file = NULL;
    char* base_comp_file = NULL;
    char* root_optimize = NULL;
    char* root_check = NULL;
    char* cache_spec_file = NULL;

    /* Parse command line arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            edge_list_file = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            pattern_file = argv[++i];
        } else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            taxon_order_file = argv[++i];
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            base_comp_file = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            root_optimize = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            root_check = argv[++i];
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            cache_spec_file = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    /* Validate required arguments */
    if (!edge_list_file || !pattern_file || !taxon_order_file ||
        !base_comp_file || !root_optimize || !root_check) {
        fprintf(stderr, "Error: Missing required arguments\n\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Print configuration */
    printf("EMBH C Implementation\n");
    printf("=====================\n");
    printf("Edge list file: %s\n", edge_list_file);
    printf("Pattern file: %s\n", pattern_file);
    printf("Taxon order file: %s\n", taxon_order_file);
    printf("Base composition file: %s\n", base_comp_file);
    printf("Root optimize: %s\n", root_optimize);
    printf("Root check: %s\n", root_check);
    if (cache_spec_file) {
        printf("Cache spec file: %s\n", cache_spec_file);
    }

    /* Create and run manager */
    Manager* mgr = manager_create(
        edge_list_file,
        pattern_file,
        taxon_order_file,
        base_comp_file,
        root_optimize,
        root_check,
        cache_spec_file
    );

    if (!mgr) {
        fprintf(stderr, "Error: Failed to create manager\n");
        return 1;
    }

    /* Print final results */
    printf("\n=== Final Results ===\n");
    printf("Maximum log-likelihood at %s: %.11f\n", root_optimize, mgr->max_log_likelihood);
    printf("Log-likelihood at %s: %.11f\n", root_check, mgr->max_log_likelihood_hss);

    /* Cleanup */
    manager_destroy(mgr);

    return 0;
}
