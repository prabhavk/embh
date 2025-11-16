/* single cpp file containing functions for embh cpp*/

/*
Input
1. fasta file // replace with pattern file
2. edge list of phylogenetic tree
3. two root locations (h_opt, h_com)
4. precomputed models M_com and M_opt for comparing log-likelihood scores

h_opt is used as root to define the BH model M_opt.
embh optimizes parameters of M_opt.

probability vector pi of M_opt is intialized with base composition of input patterns.
probability matrices P are initialized with F81 model.

h_com is used as root define BH model M_com.
parameters of M_com are derived from M_opt using Bayes rule as described in SSH paper.

Output
1. Parameters of M_opt and M_com
2. Pattern log-likelihoods for BH and SSH models
3. Log-likelihoods for BH and SSH models

*/

/* headers */

#include <iostream>
#include <cstring>
#include "embh_core.hpp"

int main(int argc, char* argv[]) {
    char* edge_list_file_name = nullptr;
    char* pattern_file_name = nullptr;
    char* taxon_order_file_name = nullptr;
    char* base_comp_file_name = nullptr;
    char* root_optimize_name = nullptr;
    char* root_check_name = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) edge_list_file_name = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) pattern_file_name = argv[++i];
        else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) taxon_order_file_name = argv[++i];
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) base_comp_file_name = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) root_optimize_name = argv[++i];
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) root_check_name = argv[++i];
    }

    if (!edge_list_file_name || !pattern_file_name || !taxon_order_file_name || !base_comp_file_name || !root_optimize_name || !root_check_name) {
        std::cerr << "Usage: " << argv[0] << " -e edge_list -p pattern_file -x taxon_order_file -b base_comp_file -o root_optimize_name -c root_check_name\n";
        return 1;
    }

    printf("edge list file is: %s\n", edge_list_file_name);
    printf("pattern file name is: %s\n", pattern_file_name);
    printf("taxon order file name is: %s\n", taxon_order_file_name);
    printf("base composition file name is: %s\n", base_comp_file_name);
    printf("root optimize name is: %s\n", root_optimize_name);
    printf("root check name is: %s\n", root_check_name);

    manager * M = new manager(
        string(edge_list_file_name),
        string(pattern_file_name),
        string(taxon_order_file_name),
        string(base_comp_file_name),
        string(root_optimize_name),
        string(root_check_name)
    );
    delete M;

    return 0;
}
