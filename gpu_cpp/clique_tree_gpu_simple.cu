/**
 * GPU-accelerated clique tree belief propagation
 *
 * Simple version: Each CUDA thread handles one pattern, performing full
 * belief propagation sequentially. This parallelizes over patterns.
 *
 * Uses the same data files as cpp_program (tree_edges.txt, patterns_1000.pat)
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
#include <tuple>
#include <functional>
#include <cuda_runtime.h>

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
} while(0)

// Maximum sizes (adjust as needed)
// Note: 73 cliques for the actual tree, but we need to be conservative with GPU memory
const int MAX_CLIQUES = 80;
const int MAX_CHILDREN = 2;

// Tree structure on GPU
struct GPUTree {
    int num_cliques;
    int num_nodes;
    int num_leaves;
    int root_node;
    int root_clique;

    // Clique info [num_cliques]
    int clique_x[MAX_CLIQUES];
    int clique_y[MAX_CLIQUES];
    int clique_parent[MAX_CLIQUES];  // -1 for root
    int clique_num_children[MAX_CLIQUES];
    int clique_children[MAX_CLIQUES * MAX_CHILDREN];

    // Node info [num_nodes]
    double branch_lengths[MAX_CLIQUES * 2];  // at most 2*num_cliques nodes

    // Traversal order [num_cliques-1 edges, stored as pairs]
    int postorder_from[MAX_CLIQUES];  // child clique
    int postorder_to[MAX_CLIQUES];    // parent clique
    int preorder_from[MAX_CLIQUES];   // parent clique
    int preorder_to[MAX_CLIQUES];     // child clique
    int num_edges;

    // Root prior
    double root_prior[4];
};

// Host-side clique representation
struct HostClique {
    int id, x_node, y_node, parent;
    vector<int> children;
};

// Global host data
int h_num_nodes, h_num_leaves;
vector<int> h_parent_map;
vector<double> h_branch_lengths;
array<double, 4> h_root_prior;
vector<pair<vector<int>, int>> h_patterns;  // (pattern, weight)

/**
 * Load tree from edges file
 */
void load_tree_from_edges(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening " << filename << endl;
        exit(1);
    }

    map<string, int> name_to_id;
    vector<tuple<string, string, double>> edges;
    set<string> all_nodes, has_parent;

    string p, c; double bl;
    while (fin >> p >> c >> bl) {
        edges.push_back({p, c, bl});
        all_nodes.insert(p);
        all_nodes.insert(c);
        has_parent.insert(c);
    }
    fin.close();

    // Separate leaves and internal nodes
    vector<string> leaves, internals;
    for (const auto& n : all_nodes) {
        if (n.substr(0, 2) == "h_") internals.push_back(n);
        else leaves.push_back(n);
    }
    sort(leaves.begin(), leaves.end());
    sort(internals.begin(), internals.end());

    h_num_leaves = leaves.size();
    h_num_nodes = all_nodes.size();

    // Assign IDs: leaves 0..num_leaves-1, internals num_leaves..
    int id = 0;
    for (const auto& n : leaves) name_to_id[n] = id++;
    for (const auto& n : internals) name_to_id[n] = id++;

    // Find root
    string root_name;
    for (const auto& n : all_nodes) {
        if (has_parent.find(n) == has_parent.end()) {
            root_name = n;
            break;
        }
    }

    cout << "Tree: " << h_num_leaves << " leaves, " << h_num_nodes << " nodes" << endl;
    cout << "Root: " << root_name << " (ID " << name_to_id[root_name] << ")" << endl;

    // Build parent map
    h_parent_map.resize(h_num_nodes, -1);
    h_branch_lengths.resize(h_num_nodes, 0.0);

    for (const auto& e : edges) {
        int pid = name_to_id[get<0>(e)];
        int cid = name_to_id[get<1>(e)];
        h_parent_map[cid] = pid;
        h_branch_lengths[cid] = get<2>(e);
    }

    h_root_prior = {0.25, 0.25, 0.25, 0.25};
}

/**
 * Load patterns from file
 */
void load_patterns(const string& filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        int weight;
        iss >> weight;
        vector<int> pat(h_num_leaves);
        for (int i = 0; i < h_num_leaves; i++) iss >> pat[i];
        h_patterns.push_back({pat, weight});
    }
    fin.close();
    cout << "Loaded " << h_patterns.size() << " patterns" << endl;
}

/**
 * Build clique tree on host
 */
vector<HostClique> build_host_clique_tree(int& root_clique_id) {
    vector<HostClique> cliques;
    map<int, vector<int>> cliques_by_child;

    int cid = 0;
    for (int child = 0; child < h_num_nodes; child++) {
        int parent = h_parent_map[child];
        if (parent != -1) {
            HostClique c;
            c.id = cid++;
            c.x_node = parent;
            c.y_node = child;
            c.parent = -1;
            cliques.push_back(c);
            cliques_by_child[child].push_back(c.id);
        }
    }

    // Link cliques
    for (auto& c : cliques) {
        for (int pid : cliques_by_child[c.x_node]) {
            c.parent = pid;
            cliques[pid].children.push_back(c.id);
            break;
        }
    }

    // Handle root cliques
    vector<int> roots;
    for (const auto& c : cliques) {
        if (c.parent == -1) roots.push_back(c.id);
    }
    root_clique_id = roots[0];
    for (size_t i = 1; i < roots.size(); i++) {
        cliques[roots[i]].parent = root_clique_id;
        cliques[root_clique_id].children.push_back(roots[i]);
    }

    return cliques;
}

/**
 * Build traversal orders
 */
void build_traversals(const vector<HostClique>& cliques, int root_clique_id,
                      vector<pair<int,int>>& postorder, vector<pair<int,int>>& preorder) {
    vector<bool> visited(cliques.size(), false);
    function<void(int)> dfs = [&](int cid) {
        visited[cid] = true;
        for (int child : cliques[cid].children) {
            if (!visited[child]) {
                dfs(child);
                postorder.push_back({cid, child});
            }
        }
    };
    dfs(root_clique_id);
    preorder = postorder;
    reverse(preorder.begin(), preorder.end());
}

/**
 * GPU Kernel: Compute log-likelihood for each pattern at each clique
 *
 * Each thread handles one pattern. Performs full belief propagation.
 *
 * MEMORY OPTIMIZED: Only store messages along edges (parent-child), not all pairs.
 * Each clique stores messages from its neighbors:
 * - messages_from_parent[clique] = message received from parent
 * - messages_from_child[clique][child_idx] = message received from child
 */
__global__ void compute_ll_kernel(
    const GPUTree* tree_ptr,  // Pass by pointer to avoid parameter size limit
    const int* patterns,  // [num_patterns * num_leaves]
    int num_patterns,
    double* pattern_lls   // [num_patterns * num_cliques] output
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_patterns) return;

    // Copy tree to local reference for convenience
    const GPUTree& tree = *tree_ptr;

    const int* pat = &patterns[pid * tree.num_leaves];

    // Local storage for this pattern - OPTIMIZED to only store neighbor messages
    double init_pot[MAX_CLIQUES][16];
    double msg_from_parent[MAX_CLIQUES][4];      // Message each clique receives from parent
    double msg_from_child[MAX_CLIQUES][MAX_CHILDREN][4];  // Messages from children
    double scale_from_parent[MAX_CLIQUES];       // Log scale factors
    double scale_from_child[MAX_CLIQUES][MAX_CHILDREN];
    double clique_scales[MAX_CLIQUES];

    // Initialize all to zero
    for (int c = 0; c < tree.num_cliques; c++) {
        clique_scales[c] = 0.0;
        scale_from_parent[c] = 0.0;
        for (int i = 0; i < 16; i++) {
            init_pot[c][i] = 0.0;
        }
        for (int i = 0; i < 4; i++) {
            msg_from_parent[c][i] = 1.0;  // Default to 1 (no message)
        }
        for (int ci = 0; ci < MAX_CHILDREN; ci++) {
            scale_from_child[c][ci] = 0.0;
            for (int i = 0; i < 4; i++) {
                msg_from_child[c][ci][i] = 1.0;
            }
        }
    }

    // Initialize potentials for each clique
    for (int c = 0; c < tree.num_cliques; c++) {
        int y = tree.clique_y[c];
        double t = tree.branch_lengths[y];
        double exp_t = exp(-t);

        for (int x = 0; x < 4; x++) {
            for (int yy = 0; yy < 4; yy++) {
                double P = (x == yy) ? exp_t + tree.root_prior[yy] * (1.0 - exp_t)
                                     : tree.root_prior[yy] * (1.0 - exp_t);
                if (y < tree.num_leaves) {
                    int state = pat[y];
                    init_pot[c][x*4+yy] = (state < 4 && yy == state) ? P : (state >= 4 ? P : 0.0);
                } else {
                    init_pot[c][x*4+yy] = P;
                }
            }
        }
    }

    // Upward pass: send messages from children to parents
    for (int e = 0; e < tree.num_edges; e++) {
        int from = tree.postorder_from[e];  // child
        int to = tree.postorder_to[e];      // parent

        int from_y = tree.clique_y[from];
        int to_x = tree.clique_x[to];
        int to_y = tree.clique_y[to];

        // Start with initial potential
        double factor[16];
        for (int i = 0; i < 16; i++) factor[i] = init_pot[from][i];

        double logScale = 0.0;

        // Multiply by messages from children of 'from' (already received)
        for (int ci = 0; ci < tree.clique_num_children[from]; ci++) {
            int child = tree.clique_children[from * MAX_CHILDREN + ci];
            int child_y = tree.clique_y[child];

            // Child shares its y_node with 'from' (child->y is separator)
            // Message applies to the dimension matching child->y
            if (from_y == child_y) {
                // from->y == child->y, message applies to Y dimension
                for (int x = 0; x < 4; x++)
                    for (int yy = 0; yy < 4; yy++)
                        factor[x*4+yy] *= msg_from_child[from][ci][yy];
            } else {
                // from->x == child->y, message applies to X dimension
                for (int yy = 0; yy < 4; yy++)
                    for (int x = 0; x < 4; x++)
                        factor[x*4+yy] *= msg_from_child[from][ci][x];
            }

            // Rescale
            double maxv = 0.0;
            for (int i = 0; i < 16; i++) if (factor[i] > maxv) maxv = factor[i];
            if (maxv > 0) {
                for (int i = 0; i < 16; i++) factor[i] /= maxv;
                logScale += log(maxv);
            }
            logScale += scale_from_child[from][ci];
        }

        // Marginalize over the variable not shared with 'to'
        double msg[4] = {0, 0, 0, 0};
        if (from_y == to_x || from_y == to_y) {
            // Shared is from_y, sum over from_x (rows)
            for (int yy = 0; yy < 4; yy++)
                for (int x = 0; x < 4; x++)
                    msg[yy] += factor[x*4+yy];
        } else {
            // Shared is from_x, sum over from_y (columns)
            for (int x = 0; x < 4; x++)
                for (int yy = 0; yy < 4; yy++)
                    msg[x] += factor[x*4+yy];
        }

        // Rescale message
        double maxv = 0.0;
        for (int i = 0; i < 4; i++) if (msg[i] > maxv) maxv = msg[i];
        if (maxv > 0) {
            for (int i = 0; i < 4; i++) msg[i] /= maxv;
            logScale += log(maxv);
        }

        // Store: 'to' (parent) receives from 'from' (child)
        // Find which child index 'from' is for 'to'
        for (int ci = 0; ci < tree.clique_num_children[to]; ci++) {
            if (tree.clique_children[to * MAX_CHILDREN + ci] == from) {
                for (int i = 0; i < 4; i++) msg_from_child[to][ci][i] = msg[i];
                scale_from_child[to][ci] = logScale;
                break;
            }
        }
    }

    // Downward pass: send messages from parents to children
    for (int e = 0; e < tree.num_edges; e++) {
        int from = tree.preorder_from[e];  // parent
        int to = tree.preorder_to[e];      // child

        int from_y = tree.clique_y[from];
        int to_x = tree.clique_x[to];
        int to_y = tree.clique_y[to];

        double factor[16];
        for (int i = 0; i < 16; i++) factor[i] = init_pot[from][i];

        double logScale = 0.0;

        // Multiply by parent's message (if exists)
        if (tree.clique_parent[from] >= 0) {
            int parent_y = tree.clique_y[tree.clique_parent[from]];

            if (from_y == parent_y) {
                // from->y matches parent->y
                for (int x = 0; x < 4; x++)
                    for (int yy = 0; yy < 4; yy++)
                        factor[x*4+yy] *= msg_from_parent[from][yy];
            } else {
                // from->x matches parent->y
                for (int yy = 0; yy < 4; yy++)
                    for (int x = 0; x < 4; x++)
                        factor[x*4+yy] *= msg_from_parent[from][x];
            }

            double maxv = 0.0;
            for (int i = 0; i < 16; i++) if (factor[i] > maxv) maxv = factor[i];
            if (maxv > 0) {
                for (int i = 0; i < 16; i++) factor[i] /= maxv;
                logScale += log(maxv);
            }
            logScale += scale_from_parent[from];
        }

        // Multiply by children's messages (except recipient 'to')
        for (int ci = 0; ci < tree.clique_num_children[from]; ci++) {
            int child = tree.clique_children[from * MAX_CHILDREN + ci];
            if (child == to) continue;

            int child_y = tree.clique_y[child];

            if (from_y == child_y) {
                for (int x = 0; x < 4; x++)
                    for (int yy = 0; yy < 4; yy++)
                        factor[x*4+yy] *= msg_from_child[from][ci][yy];
            } else {
                for (int yy = 0; yy < 4; yy++)
                    for (int x = 0; x < 4; x++)
                        factor[x*4+yy] *= msg_from_child[from][ci][x];
            }

            double maxv = 0.0;
            for (int i = 0; i < 16; i++) if (factor[i] > maxv) maxv = factor[i];
            if (maxv > 0) {
                for (int i = 0; i < 16; i++) factor[i] /= maxv;
                logScale += log(maxv);
            }
            logScale += scale_from_child[from][ci];
        }

        // Marginalize
        double msg[4] = {0, 0, 0, 0};
        if (from_y == to_x || from_y == to_y) {
            for (int yy = 0; yy < 4; yy++)
                for (int x = 0; x < 4; x++)
                    msg[yy] += factor[x*4+yy];
        } else {
            for (int x = 0; x < 4; x++)
                for (int yy = 0; yy < 4; yy++)
                    msg[x] += factor[x*4+yy];
        }

        double maxv = 0.0;
        for (int i = 0; i < 4; i++) if (msg[i] > maxv) maxv = msg[i];
        if (maxv > 0) {
            for (int i = 0; i < 4; i++) msg[i] /= maxv;
            logScale += log(maxv);
        }

        // Store: 'to' (child) receives from parent
        for (int i = 0; i < 4; i++) msg_from_parent[to][i] = msg[i];
        scale_from_parent[to] = logScale;
    }

    // Compute beliefs and log scaling factors
    double beliefs[MAX_CLIQUES][16];
    for (int c = 0; c < tree.num_cliques; c++) {
        double factor[16];
        for (int i = 0; i < 16; i++) factor[i] = init_pot[c][i];

        int y_node = tree.clique_y[c];
        double logScale = 0.0;

        // Multiply by parent message
        if (tree.clique_parent[c] >= 0) {
            int parent_y = tree.clique_y[tree.clique_parent[c]];

            if (y_node == parent_y) {
                for (int x = 0; x < 4; x++)
                    for (int yy = 0; yy < 4; yy++)
                        factor[x*4+yy] *= msg_from_parent[c][yy];
            } else {
                for (int yy = 0; yy < 4; yy++)
                    for (int x = 0; x < 4; x++)
                        factor[x*4+yy] *= msg_from_parent[c][x];
            }

            double sum = 0.0;
            for (int i = 0; i < 16; i++) sum += factor[i];
            for (int i = 0; i < 16; i++) factor[i] /= sum;
            logScale += log(sum) + scale_from_parent[c];
        }

        // Multiply by children messages
        for (int ci = 0; ci < tree.clique_num_children[c]; ci++) {
            int child_y = tree.clique_y[tree.clique_children[c * MAX_CHILDREN + ci]];

            if (y_node == child_y) {
                for (int x = 0; x < 4; x++)
                    for (int yy = 0; yy < 4; yy++)
                        factor[x*4+yy] *= msg_from_child[c][ci][yy];
            } else {
                for (int yy = 0; yy < 4; yy++)
                    for (int x = 0; x < 4; x++)
                        factor[x*4+yy] *= msg_from_child[c][ci][x];
            }

            double sum = 0.0;
            for (int i = 0; i < 16; i++) sum += factor[i];
            for (int i = 0; i < 16; i++) factor[i] /= sum;
            logScale += log(sum) + scale_from_child[c][ci];
        }

        // Final normalization
        double sum = 0.0;
        for (int i = 0; i < 16; i++) sum += factor[i];
        for (int i = 0; i < 16; i++) factor[i] /= sum;
        logScale += log(sum);

        for (int i = 0; i < 16; i++) beliefs[c][i] = factor[i];
        clique_scales[c] = logScale;
    }

    // Compute LL at each clique
    // Find a clique containing root node
    int root_clique_for_prior = -1;
    for (int c = 0; c < tree.num_cliques; c++) {
        if (tree.clique_x[c] == tree.root_node || tree.clique_y[c] == tree.root_node) {
            root_clique_for_prior = c;
            break;
        }
    }

    // Get root marginal from root-containing clique
    double root_marginal[4] = {0, 0, 0, 0};
    if (tree.clique_x[root_clique_for_prior] == tree.root_node) {
        for (int x = 0; x < 4; x++)
            for (int yy = 0; yy < 4; yy++)
                root_marginal[x] += beliefs[root_clique_for_prior][x*4+yy];
    } else {
        for (int yy = 0; yy < 4; yy++)
            for (int x = 0; x < 4; x++)
                root_marginal[yy] += beliefs[root_clique_for_prior][x*4+yy];
    }

    double site_ll_contribution = 0.0;
    for (int i = 0; i < 4; i++) {
        site_ll_contribution += tree.root_prior[i] * root_marginal[i];
    }
    double log_site_ll = log(site_ll_contribution);

    // All cliques should have same LL
    for (int c = 0; c < tree.num_cliques; c++) {
        pattern_lls[pid * tree.num_cliques + c] = clique_scales[c] + log_site_ll;
    }
}

int main(int argc, char** argv) {
    cout << "=== GPU Clique Tree Log-Likelihood Consistency Test ===" << endl;

    string tree_file = "data/tree_edges.txt";
    string pattern_file = "data/patterns_1000.pat";
    int max_patterns = 100;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--tree" && i+1 < argc) tree_file = argv[++i];
        else if (arg == "--patterns" && i+1 < argc) pattern_file = argv[++i];
        else if (arg == "--max" && i+1 < argc) max_patterns = atoi(argv[++i]);
    }

    // Load data
    load_tree_from_edges(tree_file);
    load_patterns(pattern_file);

    // Build clique tree
    int root_clique_id;
    auto cliques = build_host_clique_tree(root_clique_id);
    cout << "Built " << cliques.size() << " cliques, root clique: " << root_clique_id << endl;

    if (cliques.size() > MAX_CLIQUES) {
        cerr << "Too many cliques (" << cliques.size() << "). Increase MAX_CLIQUES." << endl;
        return 1;
    }

    // Build traversal orders
    vector<pair<int,int>> postorder, preorder;
    build_traversals(cliques, root_clique_id, postorder, preorder);

    // Find root node
    int root_node = -1;
    for (int i = 0; i < h_num_nodes; i++) {
        if (h_parent_map[i] == -1) {
            root_node = i;
            break;
        }
    }

    // Prepare GPU tree structure
    GPUTree gpu_tree;
    gpu_tree.num_cliques = cliques.size();
    gpu_tree.num_nodes = h_num_nodes;
    gpu_tree.num_leaves = h_num_leaves;
    gpu_tree.root_node = root_node;
    gpu_tree.root_clique = root_clique_id;
    gpu_tree.num_edges = postorder.size();

    for (size_t c = 0; c < cliques.size(); c++) {
        gpu_tree.clique_x[c] = cliques[c].x_node;
        gpu_tree.clique_y[c] = cliques[c].y_node;
        gpu_tree.clique_parent[c] = cliques[c].parent;
        gpu_tree.clique_num_children[c] = cliques[c].children.size();
        for (int ci = 0; ci < (int)cliques[c].children.size() && ci < MAX_CHILDREN; ci++) {
            gpu_tree.clique_children[c * MAX_CHILDREN + ci] = cliques[c].children[ci];
        }
    }

    for (int i = 0; i < h_num_nodes; i++) {
        gpu_tree.branch_lengths[i] = h_branch_lengths[i];
    }

    for (size_t e = 0; e < postorder.size(); e++) {
        gpu_tree.postorder_from[e] = postorder[e].second;  // child
        gpu_tree.postorder_to[e] = postorder[e].first;     // parent
        gpu_tree.preorder_from[e] = preorder[e].first;     // parent
        gpu_tree.preorder_to[e] = preorder[e].second;      // child
    }

    for (int i = 0; i < 4; i++) {
        gpu_tree.root_prior[i] = h_root_prior[i];
    }

    // Prepare pattern data
    int num_patterns = min((int)h_patterns.size(), max_patterns);
    vector<int> h_pattern_data(num_patterns * h_num_leaves);
    vector<int> h_weights(num_patterns);

    for (int p = 0; p < num_patterns; p++) {
        h_weights[p] = h_patterns[p].second;
        for (int l = 0; l < h_num_leaves; l++) {
            h_pattern_data[p * h_num_leaves + l] = h_patterns[p].first[l];
        }
    }

    cout << "Processing " << num_patterns << " patterns..." << endl;

    // Allocate GPU memory
    int* d_patterns;
    double* d_pattern_lls;
    GPUTree* d_tree;

    CUDA_CHECK(cudaMalloc(&d_patterns, num_patterns * h_num_leaves * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern_lls, num_patterns * gpu_tree.num_cliques * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tree, sizeof(GPUTree)));

    CUDA_CHECK(cudaMemcpy(d_patterns, h_pattern_data.data(),
                          num_patterns * h_num_leaves * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tree, &gpu_tree, sizeof(GPUTree), cudaMemcpyHostToDevice));

    // Launch kernel
    int block_size = 32;  // Small due to high register usage
    int num_blocks = (num_patterns + block_size - 1) / block_size;

    cout << "Launching kernel with " << num_blocks << " blocks x " << block_size << " threads" << endl;

    compute_ll_kernel<<<num_blocks, block_size>>>(d_tree, d_patterns, num_patterns, d_pattern_lls);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    vector<double> h_pattern_lls(num_patterns * gpu_tree.num_cliques);
    CUDA_CHECK(cudaMemcpy(h_pattern_lls.data(), d_pattern_lls,
                          num_patterns * gpu_tree.num_cliques * sizeof(double), cudaMemcpyDeviceToHost));

    // Compute weighted total LL for each clique
    vector<double> total_lls(gpu_tree.num_cliques, 0.0);
    for (int p = 0; p < num_patterns; p++) {
        for (int c = 0; c < gpu_tree.num_cliques; c++) {
            total_lls[c] += h_pattern_lls[p * gpu_tree.num_cliques + c] * h_weights[p];
        }
    }

    // Print results (first 10 cliques)
    cout << "\n=== Total Log-Likelihood at Each Clique ===" << endl;
    for (int c = 0; c < min(10, gpu_tree.num_cliques); c++) {
        cout << "Clique " << c << " (" << gpu_tree.clique_x[c] << ", "
             << gpu_tree.clique_y[c] << "): " << setprecision(10) << total_lls[c] << endl;
    }
    if (gpu_tree.num_cliques > 10) cout << "..." << endl;

    // Summary
    double min_ll = *min_element(total_lls.begin(), total_lls.end());
    double max_ll = *max_element(total_lls.begin(), total_lls.end());
    double mean_ll = 0;
    for (double ll : total_lls) mean_ll += ll;
    mean_ll /= total_lls.size();

    cout << "\n=== Summary ===" << endl;
    cout << "Patterns: " << num_patterns << ", Cliques: " << gpu_tree.num_cliques << endl;
    cout << "Min LL: " << setprecision(10) << min_ll << endl;
    cout << "Max LL: " << setprecision(10) << max_ll << endl;
    cout << "Mean LL: " << setprecision(10) << mean_ll << endl;
    cout << "Max diff: " << setprecision(10) << (max_ll - min_ll) << endl;

    if (max_ll - min_ll < 1e-6) {
        cout << "\nSUCCESS: All cliques give consistent LL values!" << endl;
    } else {
        cout << "\nWARNING: LL values differ across cliques!" << endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_patterns));
    CUDA_CHECK(cudaFree(d_pattern_lls));
    CUDA_CHECK(cudaFree(d_tree));

    return 0;
}
