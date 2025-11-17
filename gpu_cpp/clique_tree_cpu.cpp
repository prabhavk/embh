/**
 * CPU-only clique tree belief propagation implementation
 *
 * This is a direct translation of the C++ clique tree algorithm from embh_core.cpp.
 * The goal is to verify that all cliques give identical log-likelihood values after calibration.
 *
 * Key data structures (matching embh_core.cpp):
 * - initialPotential: 4x4 matrix (transition matrix × leaf indicator)
 * - messagesFromNeighbors: map from neighbor clique ID to 4-element message
 * - logScalingFactorForMessages: map from neighbor clique ID to its log scaling factor
 * - logScalingFactorForClique: accumulated during ComputeBelief()
 * - belief: 4x4 matrix (final calibrated belief)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>
#include <array>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <string>
#include <algorithm>
#include <utility>
#include <functional>
#include <tuple>

using namespace std;

// Simple 4x4 matrix type
using Matrix4x4 = array<array<double, 4>, 4>;

// Forward declaration
struct Clique;

struct Clique {
    int id;
    int x_node;  // parent variable (separator)
    int y_node;  // child variable

    // Tree structure
    Clique* parent;
    vector<Clique*> children;

    // Belief propagation data
    Matrix4x4 initialPotential;
    Matrix4x4 belief;
    map<Clique*, array<double, 4>> messagesFromNeighbors;
    map<Clique*, double> logScalingFactorForMessages;
    double logScalingFactorForClique;

    Clique(int id_, int x_, int y_) : id(id_), x_node(x_), y_node(y_), parent(nullptr) {
        // Initialize matrices to zero
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                initialPotential[i][j] = 0.0;
                belief[i][j] = 0.0;
            }
        }
        logScalingFactorForClique = 0.0;
    }

    void ComputeBelief();
    array<double, 4> MarginalizeOverVariable(int var);
};

struct CliqueTree {
    vector<Clique*> cliques;
    Clique* root_clique;
    int root_node;  // The root variable in the phylogenetic tree

    vector<pair<Clique*, Clique*>> edgesForPostOrderTreeTraversal;  // leaf to root
    vector<pair<Clique*, Clique*>> edgesForPreOrderTreeTraversal;   // root to leaf

    ~CliqueTree() {
        for (auto* c : cliques) delete c;
    }

    void SendMessage(Clique* C_from, Clique* C_to);
    void CalibrateTree();
    void BuildTraversalOrders();
};

// Global tree structure and model parameters
int num_nodes;
int num_leaves;
vector<int> parent_map;  // parent_map[i] = parent of node i (-1 for root)
vector<double> branch_lengths;
array<double, 4> root_prior;
double f81_mu = 1.0;  // F81 scaling factor (mu = 1 / (1 - sum_i pi_i^2))

// Site data
vector<int> leaf_states;  // leaf_states[leaf_idx] = nucleotide state (0-3, or 4 for unknown)

/**
 * Compute F81 mu (rate multiplier)
 * mu = 1 / (1 - sum_i pi_i^2)
 */
void compute_f81_mu() {
    double sum_pi_sq = 0.0;
    for (int i = 0; i < 4; i++) {
        sum_pi_sq += root_prior[i] * root_prior[i];
    }
    f81_mu = 1.0 / (1.0 - sum_pi_sq);
    cout << "F81 mu is " << f81_mu << endl;
}

/**
 * Compute F81 transition probability matrix
 */
Matrix4x4 compute_transition_matrix(double branch_length) {
    Matrix4x4 P;
    double t = branch_length * f81_mu;  // Scale by F81 mu
    double exp_term = exp(-t);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                P[i][j] = exp_term + root_prior[j] * (1.0 - exp_term);
            } else {
                P[i][j] = root_prior[j] * (1.0 - exp_term);
            }
        }
    }
    return P;
}

/**
 * SendMessage: Sends message from C_from to C_to
 *
 * This exactly matches embh_core.cpp SendMessage():
 * A) Multiply initial potential by messages from all neighbors except C_to
 * B) Marginalize over the variable that is in C_from but not in C_to
 * C) Store message and scaling factor in C_to
 */
void CliqueTree::SendMessage(Clique* C_from, Clique* C_to) {
    double logScalingFactor = 0.0;
    double largestElement;
    array<double, 4> messageFromNeighbor;
    array<double, 4> messageToNeighbor;

    // Select neighbors (excluding C_to)
    vector<Clique*> neighbors;
    if (C_from->parent != nullptr && C_from->parent != C_to) {
        neighbors.push_back(C_from->parent);
    }
    for (Clique* C_child : C_from->children) {
        if (C_child != C_to) {
            neighbors.push_back(C_child);
        }
    }

    // Start with initial potential
    Matrix4x4 factor = C_from->initialPotential;

    // A. PRODUCT: Multiply messages from neighbors that are not C_to
    for (Clique* C_neighbor : neighbors) {
        messageFromNeighbor = C_from->messagesFromNeighbors[C_neighbor];

        // Determine which variable is shared
        if (C_from->y_node == C_neighbor->x_node || C_from->y_node == C_neighbor->y_node) {
            // Message applies to Y dimension (rows)
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    factor[dna_x][dna_y] *= messageFromNeighbor[dna_y];
                }
            }
        } else if (C_from->x_node == C_neighbor->x_node || C_from->x_node == C_neighbor->y_node) {
            // Message applies to X dimension (columns)
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    factor[dna_x][dna_y] *= messageFromNeighbor[dna_x];
                }
            }
        } else {
            cerr << "Error in SendMessage: no shared variable" << endl;
            exit(1);
        }

        // Rescale factor to prevent underflow
        largestElement = 0.0;
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                if (largestElement < factor[dna_x][dna_y]) {
                    largestElement = factor[dna_x][dna_y];
                }
            }
        }
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                factor[dna_x][dna_y] /= largestElement;
            }
        }
        logScalingFactor += log(largestElement);
        logScalingFactor += C_from->logScalingFactorForMessages[C_neighbor];
    }

    // B. SUM: Marginalize over the variable not shared with C_to
    largestElement = 0.0;
    if (C_from->y_node == C_to->x_node || C_from->y_node == C_to->y_node) {
        // Shared variable is Y, sum over X
        for (int dna_y = 0; dna_y < 4; dna_y++) {
            messageToNeighbor[dna_y] = 0.0;
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                messageToNeighbor[dna_y] += factor[dna_x][dna_y];
            }
        }
    } else if (C_from->x_node == C_to->x_node || C_from->x_node == C_to->y_node) {
        // Shared variable is X, sum over Y
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            messageToNeighbor[dna_x] = 0.0;
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                messageToNeighbor[dna_x] += factor[dna_x][dna_y];
            }
        }
    } else {
        cerr << "Error in SendMessage: no shared variable for sum step" << endl;
        exit(1);
    }

    // Rescale message
    largestElement = 0.0;
    for (int dna_x = 0; dna_x < 4; dna_x++) {
        if (largestElement < messageToNeighbor[dna_x]) {
            largestElement = messageToNeighbor[dna_x];
        }
    }
    if (largestElement == 0.0) {
        cerr << "Error: Division by zero in SendMessage" << endl;
        exit(1);
    }
    for (int dna_x = 0; dna_x < 4; dna_x++) {
        messageToNeighbor[dna_x] /= largestElement;
    }
    logScalingFactor += log(largestElement);

    // C. TRANSMIT: Store message in recipient
    C_to->logScalingFactorForMessages[C_from] = logScalingFactor;
    C_to->messagesFromNeighbors[C_from] = messageToNeighbor;
}

/**
 * ComputeBelief: Compute calibrated belief for this clique
 *
 * This exactly matches embh_core.cpp ComputeBelief():
 * - Multiply initial potential by ALL messages from neighbors
 * - Accumulate scaling factors
 * - Normalize
 */
void Clique::ComputeBelief() {
    Matrix4x4 factor = initialPotential;

    // Get all neighbors
    vector<Clique*> neighbors = children;
    if (parent != nullptr) {
        neighbors.push_back(parent);
    }

    // Reset log scaling factor for this clique
    logScalingFactorForClique = 0.0;

    // Multiply by messages from all neighbors
    for (Clique* C_neighbor : neighbors) {
        logScalingFactorForClique += logScalingFactorForMessages[C_neighbor];
        array<double, 4> messageFromNeighbor = messagesFromNeighbors[C_neighbor];

        if (y_node == C_neighbor->x_node || y_node == C_neighbor->y_node) {
            // Message applies to Y dimension (rows)
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                for (int dna_y = 0; dna_y < 4; dna_y++) {
                    factor[dna_x][dna_y] *= messageFromNeighbor[dna_y];
                }
            }
        } else if (x_node == C_neighbor->x_node || x_node == C_neighbor->y_node) {
            // Message applies to X dimension (columns)
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                for (int dna_x = 0; dna_x < 4; dna_x++) {
                    factor[dna_x][dna_y] *= messageFromNeighbor[dna_x];
                }
            }
        } else {
            cerr << "Error in ComputeBelief: no shared variable" << endl;
            exit(1);
        }

        // Scale factor after each multiplication
        double scalingFactor = 0.0;
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                scalingFactor += factor[dna_x][dna_y];
            }
        }
        assert(scalingFactor > 0);
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                factor[dna_x][dna_y] /= scalingFactor;
            }
        }
        logScalingFactorForClique += log(scalingFactor);
    }

    // Final normalization
    double scalingFactor = 0.0;
    for (int dna_x = 0; dna_x < 4; dna_x++) {
        for (int dna_y = 0; dna_y < 4; dna_y++) {
            scalingFactor += factor[dna_x][dna_y];
        }
    }
    assert(scalingFactor > 0);
    for (int dna_x = 0; dna_x < 4; dna_x++) {
        for (int dna_y = 0; dna_y < 4; dna_y++) {
            factor[dna_x][dna_y] /= scalingFactor;
        }
    }
    logScalingFactorForClique += log(scalingFactor);

    // Copy to belief
    belief = factor;
}

/**
 * MarginalizeOverVariable: Marginalize belief over given variable
 */
array<double, 4> Clique::MarginalizeOverVariable(int var) {
    array<double, 4> marginal = {0.0, 0.0, 0.0, 0.0};

    if (var == y_node) {
        // Sum over Y, get marginal over X
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                marginal[dna_x] += belief[dna_x][dna_y];
            }
        }
    } else if (var == x_node) {
        // Sum over X, get marginal over Y
        for (int dna_y = 0; dna_y < 4; dna_y++) {
            for (int dna_x = 0; dna_x < 4; dna_x++) {
                marginal[dna_y] += belief[dna_x][dna_y];
            }
        }
    } else {
        cerr << "Error: variable " << var << " not in clique" << endl;
        exit(1);
    }

    return marginal;
}

/**
 * CalibrateTree: Perform two-pass message passing
 */
void CliqueTree::CalibrateTree() {
    // Send messages from leaves to root (upward pass)
    for (auto& edge : edgesForPostOrderTreeTraversal) {
        SendMessage(edge.second, edge.first);  // child to parent
    }

    // Send messages from root to leaves (downward pass)
    for (auto& edge : edgesForPreOrderTreeTraversal) {
        SendMessage(edge.first, edge.second);  // parent to child
    }

    // Compute beliefs for all cliques
    for (Clique* C : cliques) {
        C->ComputeBelief();
    }
}

/**
 * BuildTraversalOrders: Build post-order and pre-order traversal edges
 */
void CliqueTree::BuildTraversalOrders() {
    edgesForPostOrderTreeTraversal.clear();
    edgesForPreOrderTreeTraversal.clear();

    // Post-order: leaves to root (children before parent)
    vector<bool> visited(cliques.size(), false);
    vector<Clique*> order;

    function<void(Clique*)> postorder = [&](Clique* c) {
        visited[c->id] = true;
        for (Clique* child : c->children) {
            if (!visited[child->id]) {
                postorder(child);
                edgesForPostOrderTreeTraversal.push_back({c, child});
            }
        }
        order.push_back(c);
    };

    postorder(root_clique);

    // Pre-order: root to leaves (parent before children)
    for (auto& edge : edgesForPostOrderTreeTraversal) {
        edgesForPreOrderTreeTraversal.push_back({edge.first, edge.second});
    }
    reverse(edgesForPreOrderTreeTraversal.begin(), edgesForPreOrderTreeTraversal.end());
}

/**
 * Build clique tree from phylogenetic tree
 * Each edge (parent -> child) becomes a clique (parent_node, child_node)
 */
CliqueTree* build_clique_tree() {
    CliqueTree* tree = new CliqueTree();
    tree->root_node = -1;

    // Find root node
    for (int i = 0; i < num_nodes; i++) {
        if (parent_map[i] == -1) {
            tree->root_node = i;
            break;
        }
    }

    // Create cliques for each edge
    map<int, vector<Clique*>> cliques_by_child;  // cliques where child_node = key
    map<int, vector<Clique*>> cliques_by_parent; // cliques where parent_node = key

    int clique_id = 0;
    for (int child = 0; child < num_nodes; child++) {
        int parent = parent_map[child];
        if (parent != -1) {  // Not root
            Clique* c = new Clique(clique_id++, parent, child);
            tree->cliques.push_back(c);
            cliques_by_child[child].push_back(c);
            cliques_by_parent[parent].push_back(c);
        }
    }

    // Build tree structure: Connect cliques that share a variable
    // A clique (A, B) is parent of clique (B, C) - they share variable B
    for (Clique* c : tree->cliques) {
        // Find parent clique: must share c->x_node
        for (Clique* potential_parent : cliques_by_child[c->x_node]) {
            // potential_parent->y_node = c->x_node, so they share x_node
            c->parent = potential_parent;
            potential_parent->children.push_back(c);
            break;
        }
    }

    // Identify root cliques (those with no parent in the clique tree)
    vector<Clique*> root_cliques;
    for (Clique* c : tree->cliques) {
        if (c->parent == nullptr) {
            root_cliques.push_back(c);
        }
    }

    // Designate the first root clique as THE root
    if (!root_cliques.empty()) {
        tree->root_clique = root_cliques[0];
        // Make other root cliques children of the designated root
        for (size_t i = 1; i < root_cliques.size(); i++) {
            root_cliques[i]->parent = tree->root_clique;
            tree->root_clique->children.push_back(root_cliques[i]);
        }
    }

    // Build traversal orders
    tree->BuildTraversalOrders();

    return tree;
}

/**
 * Initialize clique potentials for a given site pattern
 */
void initialize_clique_potentials(CliqueTree* tree) {
    for (Clique* c : tree->cliques) {
        // Compute transition matrix P(Y | X)
        Matrix4x4 trans_matrix = compute_transition_matrix(branch_lengths[c->y_node]);

        // Check if Y is a leaf
        bool y_is_leaf = (c->y_node < num_leaves);

        // Initialize potential: P(Y | X) × indicator(Y = observed state)
        for (int dna_x = 0; dna_x < 4; dna_x++) {
            for (int dna_y = 0; dna_y < 4; dna_y++) {
                if (y_is_leaf) {
                    int state = leaf_states[c->y_node];
                    if (state < 4) {
                        // Observed state
                        c->initialPotential[dna_x][dna_y] = (dna_y == state) ? trans_matrix[dna_x][dna_y] : 0.0;
                    } else {
                        // Unknown state (N), all states possible
                        c->initialPotential[dna_x][dna_y] = trans_matrix[dna_x][dna_y];
                    }
                } else {
                    // Internal node
                    c->initialPotential[dna_x][dna_y] = trans_matrix[dna_x][dna_y];
                }
            }
        }

        // Clear messages and scaling factors
        c->messagesFromNeighbors.clear();
        c->logScalingFactorForMessages.clear();
        c->logScalingFactorForClique = 0.0;
    }
}

/**
 * Compute log-likelihood at a specific clique
 *
 * Key insight from embh_core.cpp:
 * After calibration, each clique's logScalingFactorForClique contains the log of the
 * partition function (unnormalized likelihood). The beliefs are normalized to sum to 1.
 *
 * For cliques containing the root variable:
 *   LL = logScalingFactorForClique + log(sum_root P(root) * P(root | data))
 *
 * For non-root cliques, we need to incorporate the root prior by finding
 * the marginal over the root variable from a clique that contains it.
 *
 * The crucial property we're testing: After proper calibration with the junction tree
 * algorithm, all cliques should give the same log-likelihood value, because the
 * partition function (evidence) is the same regardless of where you compute it.
 */
double compute_ll_at_clique(Clique* target_clique, CliqueTree* tree, const array<double, 4>& root_prior) {
    double siteLikelihood = 0.0;
    int root_node = tree->root_node;

    // Check if target clique contains the root variable
    if (target_clique->x_node == root_node) {
        // X is root, marginalize over Y
        array<double, 4> marginalX = target_clique->MarginalizeOverVariable(target_clique->y_node);
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += root_prior[dna] * marginalX[dna];
        }
    } else if (target_clique->y_node == root_node) {
        // Y is root, marginalize over X
        array<double, 4> marginalY = target_clique->MarginalizeOverVariable(target_clique->x_node);
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += root_prior[dna] * marginalY[dna];
        }
    } else {
        // Non-root clique: we need to find a clique containing the root
        // and get the marginal P(root | data) from there
        Clique* root_clique = nullptr;
        for (Clique* c : tree->cliques) {
            if (c->x_node == root_node || c->y_node == root_node) {
                root_clique = c;
                break;
            }
        }

        if (root_clique == nullptr) {
            cerr << "Error: No clique contains root node" << endl;
            exit(1);
        }

        // Get marginal over root from the root-containing clique
        array<double, 4> marginalRoot;
        if (root_clique->x_node == root_node) {
            marginalRoot = root_clique->MarginalizeOverVariable(root_clique->y_node);
        } else {
            marginalRoot = root_clique->MarginalizeOverVariable(root_clique->x_node);
        }

        // Compute P(data) = sum_r P(r) * P(r | data)
        // This should give the same result as root clique computation
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += root_prior[dna] * marginalRoot[dna];
        }
    }

    return target_clique->logScalingFactorForClique + log(siteLikelihood);
}

/**
 * Load tree structure from edge file using taxon order
 * Format: parent_name child_name branch_length
 */
void load_tree_from_edges(const string& edge_file, const string& taxon_order_file = "") {
    ifstream fin(edge_file);
    if (!fin) {
        cerr << "Error opening tree edge file: " << edge_file << endl;
        exit(1);
    }

    vector<tuple<string, string, double>> edges;
    set<string> all_nodes;
    set<string> has_parent;

    string parent_name, child_name;
    double branch_len;

    while (fin >> parent_name >> child_name >> branch_len) {
        edges.push_back({parent_name, child_name, branch_len});
        all_nodes.insert(parent_name);
        all_nodes.insert(child_name);
        has_parent.insert(child_name);
    }
    fin.close();

    // Identify leaves and internal nodes
    vector<string> leaf_names;
    vector<string> internal_names;

    for (const auto& name : all_nodes) {
        if (name.substr(0, 2) == "h_") {
            internal_names.push_back(name);
        } else {
            leaf_names.push_back(name);
        }
    }

    num_leaves = leaf_names.size();
    num_nodes = all_nodes.size();

    cout << "Tree has " << num_leaves << " leaves and " << num_nodes << " total nodes" << endl;

    // Load taxon order if provided
    map<string, int> name_to_id;
    if (!taxon_order_file.empty()) {
        ifstream tof(taxon_order_file);
        if (tof) {
            string line;
            getline(tof, line);  // Skip header
            while (getline(tof, line)) {
                size_t comma = line.find(',');
                if (comma != string::npos) {
                    string taxon = line.substr(0, comma);
                    int pos = stoi(line.substr(comma + 1));
                    if (name_to_id.find(taxon) == name_to_id.end()) {
                        name_to_id[taxon] = pos;
                    }
                }
            }
            tof.close();
            cout << "Loaded taxon order from " << taxon_order_file << endl;

            // Reorder leaf_names according to taxon order
            leaf_names.clear();
            leaf_names.resize(num_leaves);
            for (const auto& p : name_to_id) {
                if (p.second < num_leaves) {
                    leaf_names[p.second] = p.first;
                }
            }
        } else {
            cout << "Warning: Could not load taxon order file, using alphabetical order" << endl;
            sort(leaf_names.begin(), leaf_names.end());
            for (int i = 0; i < (int)leaf_names.size(); i++) {
                name_to_id[leaf_names[i]] = i;
            }
        }
    } else {
        sort(leaf_names.begin(), leaf_names.end());
        for (int i = 0; i < (int)leaf_names.size(); i++) {
            name_to_id[leaf_names[i]] = i;
        }
    }

    // Sort internal nodes and assign IDs after leaves
    sort(internal_names.begin(), internal_names.end());
    int next_id = num_leaves;
    for (const auto& name : internal_names) {
        name_to_id[name] = next_id++;
    }

    // Find root (node with no parent)
    string root_name;
    for (const auto& name : all_nodes) {
        if (has_parent.find(name) == has_parent.end()) {
            root_name = name;
            break;
        }
    }
    cout << "Root node: " << root_name << " (ID: " << name_to_id[root_name] << ")" << endl;

    // Build parent map and branch lengths
    parent_map.resize(num_nodes, -1);
    branch_lengths.resize(num_nodes, 0.0);

    for (const auto& edge : edges) {
        string p_name = get<0>(edge);
        string c_name = get<1>(edge);
        double b_len = get<2>(edge);

        int p_id = name_to_id[p_name];
        int c_id = name_to_id[c_name];

        parent_map[c_id] = p_id;
        branch_lengths[c_id] = b_len;
    }

    // Set root prior (uniform, will be overwritten if basecomp loaded)
    root_prior = {0.25, 0.25, 0.25, 0.25};

    // Print leaf order
    cout << "Leaf order:" << endl;
    for (int i = 0; i < min(num_leaves, 10); i++) {
        cout << "  " << i << ": " << leaf_names[i] << endl;
    }
    if (num_leaves > 10) cout << "  ..." << endl;
}

/**
 * Load site patterns from file
 * Format: weight state_1 state_2 ... state_n
 * Returns: vector of (pattern, weight) pairs
 */
vector<pair<vector<int>, int>> load_patterns(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening pattern file: " << filename << endl;
        exit(1);
    }

    vector<pair<vector<int>, int>> patterns;
    string line;

    while (getline(fin, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int weight;
        iss >> weight;

        vector<int> pattern(num_leaves);
        for (int i = 0; i < num_leaves; i++) {
            iss >> pattern[i];
        }
        patterns.push_back({pattern, weight});
    }

    fin.close();
    return patterns;
}

/**
 * Load base composition for F81 model
 * Format: state freq (count)
 */
void load_base_composition(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening base composition file: " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(fin, line)) {
        if (line.empty() || line[0] == 'T' || line[0] == 'N' || line[0] == 'G') continue;
        istringstream iss(line);
        int state;
        double freq;
        if (iss >> state >> freq) {
            if (state >= 0 && state < 4) {
                root_prior[state] = freq;
            }
        }
    }

    fin.close();

    // Normalize just in case
    double sum = 0.0;
    for (int i = 0; i < 4; i++) sum += root_prior[i];
    for (int i = 0; i < 4; i++) root_prior[i] /= sum;

    cout << "Base composition (A, C, G, T): " << root_prior[0] << ", "
         << root_prior[1] << ", " << root_prior[2] << ", " << root_prior[3] << endl;
}

/**
 * Load tree structure from file (legacy format)
 */
void load_tree(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening tree file: " << filename << endl;
        exit(1);
    }

    fin >> num_nodes >> num_leaves;
    parent_map.resize(num_nodes);
    branch_lengths.resize(num_nodes);

    for (int i = 0; i < num_nodes; i++) {
        fin >> parent_map[i] >> branch_lengths[i];
    }

    // Read root prior
    for (int i = 0; i < 4; i++) {
        fin >> root_prior[i];
    }

    fin.close();
}

/**
 * Load site pattern from file (legacy format)
 */
void load_site_pattern(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening pattern file: " << filename << endl;
        exit(1);
    }

    leaf_states.resize(num_leaves);
    for (int i = 0; i < num_leaves; i++) {
        fin >> leaf_states[i];
    }

    fin.close();
}

/**
 * Generate a random tree for testing
 *
 * Creates a balanced binary tree where:
 * - Nodes 0 to num_leaves-1 are leaves
 * - Nodes num_leaves to num_nodes-2 are internal nodes
 * - Node num_nodes-1 is the root
 *
 * Structure for n_leaves=5 (9 nodes total):
 * Leaves: 0, 1, 2, 3, 4
 * Internal: 5, 6, 7
 * Root: 8
 *
 * Tree structure:
 *           8 (root)
 *          / \
 *         7   6
 *        / \  / \
 *       5  4 3  2
 *      / \
 *     0   1
 */
void generate_test_tree(int n_leaves) {
    num_leaves = n_leaves;
    num_nodes = 2 * num_leaves - 1;  // Binary tree
    parent_map.resize(num_nodes, -1);
    branch_lengths.resize(num_nodes, 0.0);

    int root = num_nodes - 1;
    parent_map[root] = -1;
    branch_lengths[root] = 0.0;

    // Use a simple recursive approach to build a balanced tree
    // Each internal node will have exactly 2 children

    // Collect all nodes that need to be assigned as children
    vector<int> nodes_to_assign;
    for (int i = 0; i < num_nodes - 1; i++) {  // All except root
        nodes_to_assign.push_back(i);
    }

    // Queue of parents that need children
    vector<int> parents_needing_children;
    parents_needing_children.push_back(root);

    // Assign children to parents, prioritizing internal nodes
    // Internal nodes are num_leaves..num_nodes-2
    // We want internal nodes to be parents of other internal nodes or leaves

    // Sort nodes_to_assign: internal nodes first (descending), then leaves (descending)
    sort(nodes_to_assign.begin(), nodes_to_assign.end(), [](int a, int b) {
        return a > b;  // Higher index first (internal nodes have higher indices)
    });

    while (!nodes_to_assign.empty() && !parents_needing_children.empty()) {
        int parent = parents_needing_children.front();
        parents_needing_children.erase(parents_needing_children.begin());

        // Assign up to 2 children
        for (int c = 0; c < 2 && !nodes_to_assign.empty(); c++) {
            int child = nodes_to_assign.front();
            nodes_to_assign.erase(nodes_to_assign.begin());
            parent_map[child] = parent;
            branch_lengths[child] = 0.1 + 0.05 * ((child * 7 + c * 3) % 10);

            // If child is internal node, it needs children too
            if (child >= num_leaves) {
                parents_needing_children.push_back(child);
            }
        }
    }

    // Verify tree is complete
    for (int i = 0; i < num_nodes; i++) {
        if (i != root && parent_map[i] == -1) {
            cerr << "Error: Node " << i << " has no parent" << endl;
            exit(1);
        }
    }

    // Debug: Print tree structure
    cout << "Tree structure (node -> parent):" << endl;
    for (int i = 0; i < num_nodes; i++) {
        cout << "  Node " << i << " -> " << parent_map[i];
        if (i < num_leaves) cout << " (leaf)";
        else if (i == root) cout << " (root)";
        else cout << " (internal)";
        cout << ", branch_length=" << branch_lengths[i] << endl;
    }

    // Root prior (F81 model)
    root_prior = {0.25, 0.25, 0.25, 0.25};

    // Generate leaf states (deterministic for reproducibility)
    leaf_states.resize(num_leaves);
    for (int i = 0; i < num_leaves; i++) {
        leaf_states[i] = (i * 3) % 4;  // 0, 3, 2, 1, 0, 3, ...
    }
    cout << "Leaf states: ";
    for (int i = 0; i < num_leaves; i++) {
        cout << leaf_states[i] << " ";
    }
    cout << endl;
}

int main(int argc, char** argv) {
    cout << "=== CPU Clique Tree Log-Likelihood Consistency Test ===" << endl;

    string tree_file = "data/tree_edges.txt";
    string pattern_file = "data/patterns_1000.pat";
    string basecomp_file = "data/patterns_1000.basecomp";
    string taxon_order_file = "data/patterns_1000.taxon_order";
    bool use_real_data = false;
    int n_leaves = 10;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--tree" && i + 1 < argc) {
            tree_file = argv[++i];
            use_real_data = true;
        } else if (arg == "--patterns" && i + 1 < argc) {
            pattern_file = argv[++i];
            use_real_data = true;
        } else if (arg == "--basecomp" && i + 1 < argc) {
            basecomp_file = argv[++i];
            use_real_data = true;
        } else if (arg == "--taxon_order" && i + 1 < argc) {
            taxon_order_file = argv[++i];
            use_real_data = true;
        } else if (arg == "--real") {
            use_real_data = true;
        } else if (isdigit(arg[0])) {
            n_leaves = atoi(argv[i]);
        }
    }

    if (use_real_data) {
        cout << "Loading real data from files..." << endl;
        cout << "Tree file: " << tree_file << endl;
        cout << "Pattern file: " << pattern_file << endl;
        cout << "Base comp file: " << basecomp_file << endl;
        cout << "Taxon order file: " << taxon_order_file << endl;

        // Load tree structure with taxon order
        load_tree_from_edges(tree_file, taxon_order_file);

        // Load base composition for F81 model
        load_base_composition(basecomp_file);
        compute_f81_mu();

        // Build clique tree
        CliqueTree* tree = build_clique_tree();
        cout << "\nBuilt clique tree with " << tree->cliques.size() << " cliques" << endl;
        cout << "Root clique: (" << tree->root_clique->x_node << ", " << tree->root_clique->y_node << ")" << endl;

        // Load patterns
        auto patterns = load_patterns(pattern_file);
        cout << "Loaded " << patterns.size() << " patterns" << endl;

        // Compute total LL over all patterns, checking consistency at each clique
        cout << "\nComputing log-likelihood at each clique over all patterns..." << endl;

        vector<double> total_lls(tree->cliques.size(), 0.0);
        int num_patterns = patterns.size();  // Process all patterns

        for (int p = 0; p < num_patterns; p++) {
            const auto& pattern = patterns[p].first;
            int weight = patterns[p].second;

            // Set leaf states for this pattern
            leaf_states = pattern;

            // Initialize and calibrate
            initialize_clique_potentials(tree);
            tree->CalibrateTree();

            // Compute LL at each clique
            for (size_t c = 0; c < tree->cliques.size(); c++) {
                double ll = compute_ll_at_clique(tree->cliques[c], tree, root_prior);
                total_lls[c] += ll * weight;
            }
        }

        // Print results
        cout << "\n=== Total Log-Likelihood at Each Clique ===" << endl;
        cout << setw(10) << "CliqueID" << setw(10) << "X_node" << setw(10) << "Y_node"
             << setw(20) << "Total_LL" << endl;

        for (size_t c = 0; c < tree->cliques.size(); c++) {
            cout << setw(10) << tree->cliques[c]->id
                 << setw(10) << tree->cliques[c]->x_node
                 << setw(10) << tree->cliques[c]->y_node
                 << setw(20) << setprecision(10) << total_lls[c] << endl;
        }

        // Compute statistics
        double min_ll = *min_element(total_lls.begin(), total_lls.end());
        double max_ll = *max_element(total_lls.begin(), total_lls.end());
        double mean_ll = 0.0;
        for (double ll : total_lls) mean_ll += ll;
        mean_ll /= total_lls.size();

        cout << "\n=== Summary ===" << endl;
        cout << "Patterns processed: " << num_patterns << endl;
        cout << "Min LL: " << setprecision(10) << min_ll << endl;
        cout << "Max LL: " << setprecision(10) << max_ll << endl;
        cout << "Mean LL: " << setprecision(10) << mean_ll << endl;
        cout << "Max difference: " << setprecision(10) << (max_ll - min_ll) << endl;

        if (max_ll - min_ll < 1e-6) {
            cout << "\nSUCCESS: All cliques give consistent log-likelihood values!" << endl;
        } else {
            cout << "\nWARNING: Log-likelihood values differ across cliques" << endl;
        }

        delete tree;
    } else {
        cout << "Testing with " << n_leaves << " synthetic leaves" << endl;

        // Generate test tree
        generate_test_tree(n_leaves);

        cout << "Generated tree with " << num_nodes << " nodes" << endl;
        cout << "Root node: ";
        for (int i = 0; i < num_nodes; i++) {
            if (parent_map[i] == -1) {
                cout << i << endl;
                break;
            }
        }

        // Build clique tree
        CliqueTree* tree = build_clique_tree();
        cout << "Built clique tree with " << tree->cliques.size() << " cliques" << endl;
        cout << "Root clique: (" << tree->root_clique->x_node << ", " << tree->root_clique->y_node << ")" << endl;

        // Initialize potentials
        initialize_clique_potentials(tree);

        // Calibrate tree
        cout << "\nCalibrating tree..." << endl;
        tree->CalibrateTree();

        // Compute log-likelihood at each clique
        cout << "\n=== Log-Likelihood at Each Clique ===" << endl;
        cout << setw(10) << "CliqueID" << setw(10) << "X_node" << setw(10) << "Y_node"
             << setw(20) << "LogScalingFactor" << setw(20) << "LL" << endl;

        vector<double> lls;
        for (Clique* c : tree->cliques) {
            double ll = compute_ll_at_clique(c, tree, root_prior);
            lls.push_back(ll);
            cout << setw(10) << c->id << setw(10) << c->x_node << setw(10) << c->y_node
                 << setw(20) << setprecision(10) << c->logScalingFactorForClique
                 << setw(20) << setprecision(10) << ll << endl;
        }

        // Compute statistics
        double min_ll = *min_element(lls.begin(), lls.end());
        double max_ll = *max_element(lls.begin(), lls.end());
        double mean_ll = 0.0;
        for (double ll : lls) mean_ll += ll;
        mean_ll /= lls.size();

        cout << "\n=== Summary ===" << endl;
        cout << "Min LL: " << setprecision(10) << min_ll << endl;
        cout << "Max LL: " << setprecision(10) << max_ll << endl;
        cout << "Mean LL: " << setprecision(10) << mean_ll << endl;
        cout << "Max difference: " << setprecision(10) << (max_ll - min_ll) << endl;

        if (max_ll - min_ll < 1e-8) {
            cout << "\nSUCCESS: All cliques give consistent log-likelihood values!" << endl;
        } else {
            cout << "\nWARNING: Log-likelihood values differ across cliques" << endl;
        }

        delete tree;
    }

    return 0;
}
