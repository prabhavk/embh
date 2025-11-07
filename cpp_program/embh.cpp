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

int main(int argc, char* argv[]) {
    char* edge_list = nullptr;
    char* fasta_file = nullptr;
    char* root_estimate = nullptr;
    char* root_test = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) edge_list = argv[++i];
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) fasta_file = argv[++i];
        else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) root_estimate = argv[++i];
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) root_test = argv[++i];
    }

    if (!edge_list || !fasta_file || !root_estimate || !root_test) {
        std::cerr << "Usage: " << argv[0] << " -e edge_list -f fasta_file -r root_estimate -t root_test\n";
        return 1;
    }

    printf("Edge list file is: %s\n", edge_list);
    printf("Fasta file is: %s\n", fasta_file);
    printf("Root estimate is: %s\n", root_estimate);
    printf("Root test is: %s\n", root_test);

    return 0;
}