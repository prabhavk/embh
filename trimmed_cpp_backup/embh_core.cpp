#include "embh_core.hpp"

#include <map>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <tuple>
#include <chrono>
#include <set>

using namespace std;

namespace emtr {  
    
    // ---------- 4x4 helpers ----------
    using Md = std::array<std::array<double,4>,4>;

    inline std::mt19937_64& rng() {
        static thread_local std::mt19937_64 g{0xC0FFEEULL};
        return g;
    }

    inline bool starts_with(const std::string& str, const std::string& prefix) {
        return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
    }

    inline std::pair<int,int> ord_pair(int a, int b) {
        return (a < b) ? std::make_pair(a,b) : std::make_pair(b,a);
    }

    inline std::vector<std::string> split_ws(const std::string& s) {
        std::istringstream iss(s);
        std::vector<std::string> out;
        for (std::string tok; iss >> tok;) out.push_back(tok);
        return out;
    }

    inline Md MT(const Md& P) {
        Md Pt{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                Pt[j][i] = P[i][j];
        return Pt;
    }

    inline Md MM(const Md& A, const Md& B) {
        Md R{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < 4; ++k)
                    R[i][j] += A[i][k] * B[k][j];
        return R;
    }  
}

using namespace emtr;

int ll_precision = 14;


class pattern {
	public:
 		int weight;
		vector <char> characters;
	pattern (int weightToSet, vector <char> charactersToSet) {
		this->weight = weightToSet;
		this->characters = charactersToSet;
	}
};

class PackedPatternStorage {
private:
    vector <uint8_t> packed_data;  // Packed 3-bit values
    int num_patterns;
    int num_taxa;
    
    // Pack 3-bit value at specific position
    void set_base(int pattern_idx, int taxon_idx, uint8_t base) {
        int bit_position = (pattern_idx * num_taxa + taxon_idx) * 3;
        int byte_idx = bit_position / 8;
        int bit_offset = bit_position % 8;
        
        // Ensure we have enough space
        int required_bytes = (bit_position + 3 + 7) / 8;
        if (packed_data.size() < required_bytes) {
            packed_data.resize(required_bytes, 0);
        }
        
        // Clear the 3 bits we're about to write
        if (bit_offset <= 5) {
            // All 3 bits in same byte
            uint8_t mask = ~(0x07 << bit_offset);
            packed_data[byte_idx] = (packed_data[byte_idx] & mask) | (base << bit_offset);
        } else {
            // Spans two bytes
            int bits_in_first = 8 - bit_offset;
            int bits_in_second = 3 - bits_in_first;
            
            uint8_t mask1 = ~(((1 << bits_in_first) - 1) << bit_offset);
            packed_data[byte_idx] = (packed_data[byte_idx] & mask1) | (base << bit_offset);
            
            if (byte_idx + 1 < packed_data.size()) {
                uint8_t mask2 = ~((1 << bits_in_second) - 1);
                packed_data[byte_idx + 1] = (packed_data[byte_idx + 1] & mask2) | (base >> bits_in_first);
            }
        }
    }
    
public:
    PackedPatternStorage(int num_patterns, int num_taxa) 
        : num_patterns(num_patterns), num_taxa(num_taxa) {
        int total_bits = num_patterns * num_taxa * 3;
        int required_bytes = (total_bits + 7) / 8;
        packed_data.resize(required_bytes, 0);
    }
    
    // Get base at specific position (unpacks 3 bits)
    uint8_t get_base(int pattern_idx, int taxon_idx) const {
        int bit_position = (pattern_idx * num_taxa + taxon_idx) * 3;
        int byte_idx = bit_position / 8;
        int bit_offset = bit_position % 8;
        
        if (byte_idx >= packed_data.size()) return 4; // gap
        
        if (bit_offset <= 5) {
            // All 3 bits in same byte
            return (packed_data[byte_idx] >> bit_offset) & 0x07;
        } else {
            // Spans two bytes
            int bits_in_first = 8 - bit_offset;
            uint8_t low_bits = (packed_data[byte_idx] >> bit_offset);
            
            if (byte_idx + 1 < packed_data.size()) {
                int bits_in_second = 3 - bits_in_first;
                uint8_t high_bits = (packed_data[byte_idx + 1] & ((1 << bits_in_second) - 1));
                return low_bits | (high_bits << bits_in_first);
            }
            return low_bits & 0x07;
        }
    }
    
    // Store pattern from vector
    void store_pattern(int pattern_idx, const vector<uint8_t>& pattern) {
        for (int i = 0; i < min((int)pattern.size(), num_taxa); i++) {
            set_base(pattern_idx, i, pattern[i]);
        }
    }
    
    // Get entire pattern as vector
    vector<uint8_t> get_pattern(int pattern_idx) const {
        vector<uint8_t> pattern(num_taxa);
        for (int i = 0; i < num_taxa; i++) {
            pattern[i] = get_base(pattern_idx, i);
        }
        return pattern;
    }
    
    // Get memory usage stats
    size_t get_memory_bytes() const {
        return packed_data.size();
    }
    
    int get_num_patterns() const { return num_patterns; }
    int get_num_taxa() const { return num_taxa; }
};

class SEM_vertex {
public:
	int degree = 0;
	int pattern_index;
	int timesVisited = 0;
	bool observed = 0;
	string completeSequence;
    vector <int> DNArecoded;
	vector <int> DNAcompressed;
	vector <string> dupl_seq_names;
	int id = -40;
	int global_id = -40;
	string newickLabel = "";
	string name = "";
	double logScalingFactors = 0;
	double vertexLogLikelihood = 0;
	double sumOfEdgeLogLikelihoods = 0;
	int rateCategory = 0;
	int GCContent = 0;	
	bool DNA_cond_like_initialized = 0;
	vector <SEM_vertex *> neighbors;
	vector <SEM_vertex *> children;
	array <double, 4> root_prob_hss;
	SEM_vertex * parent = this;
	void AddNeighbor(SEM_vertex * v_ptr);
	void RemoveNeighbor(SEM_vertex * v_ptr);
	void AddParent(SEM_vertex * v_ptr);
	void RemoveParent();
	void AddChild(SEM_vertex * v_ptr);
	void RemoveChild(SEM_vertex * v_ptr);
	void SetVertexLogLikelihood(double vertexLogLikelihoodToSet);
	int inDegree = 0;
	int outDegree = 0;
	emtr::Md transitionMatrix;	
	emtr::Md transitionMatrix_stored;	
	array <double, 4> rootProbability;
	array <double, 4> posteriorProbability;
	
	SEM_vertex (int idToAdd, vector <int> compressedSequenceToAdd) {
		this->id = idToAdd;
		this->DNArecoded = compressedSequenceToAdd;
		this->transitionMatrix = emtr::Md{};
		this->transitionMatrix_stored = emtr::Md{};
		for (int dna = 0; dna < 4; dna ++) {
			this->transitionMatrix[dna][dna] = 1.0;
			this->transitionMatrix_stored[dna][dna] = 1.0;
		}
		for (int i = 0; i < 4; i ++) {
			this->rootProbability[i] = 0;
			this->posteriorProbability[i] = 0;
			this->root_prob_hss[i] = 0;
		}
	}	
	~SEM_vertex () {
		this->neighbors.clear();
	}
};

void SEM_vertex::SetVertexLogLikelihood(double vertexLogLikelihoodToSet) {
	this->vertexLogLikelihood = vertexLogLikelihoodToSet;
}

void SEM_vertex::AddParent(SEM_vertex * v) {
	this->parent = v;
	this->inDegree += 1;
}

void SEM_vertex::RemoveParent() {
	this->parent = this;
	this->inDegree -=1;
}

void SEM_vertex::AddChild(SEM_vertex * v) {
	this->children.push_back(v);
	this->outDegree += 1;
}

void SEM_vertex::RemoveChild(SEM_vertex * v) {
	int ind = find(this->children.begin(),this->children.end(),v) - this->children.begin();
	this->children.erase(this->children.begin()+ind);
	this->outDegree -=1;
}

void SEM_vertex::AddNeighbor(SEM_vertex * v) {
	this->degree += 1;
	this->neighbors.push_back(v);
}

void SEM_vertex::RemoveNeighbor(SEM_vertex * v) {
	this->degree -= 1;
	int ind = find(this->neighbors.begin(),this->neighbors.end(),v) - this->neighbors.begin();
	this->neighbors.erase(this->neighbors.begin()+ind);
}

class clique {
	public:
	map <clique *, double> logScalingFactorForMessages;
	double logScalingFactorForClique;
	map <clique *, std::array <double, 4>> messagesFromNeighbors;
    vector <int> compressedSequence;
	string name;
	int id;
	int site;
	int inDegree = 0;
	int outDegree = 0;
	int timesVisited = 0;
	clique * parent = this;
	SEM_vertex * root_variable;
	vector <clique *> children;
	void AddParent(clique * C);
	void AddChild(clique * C);
	void ComputeBelief();
	void ScaleFactor();
	SEM_vertex * x;
	SEM_vertex * y;
	std::array <double, 4> MarginalizeOverVariable(SEM_vertex * v);
    // emtr::Md DivideBeliefByMessageMarginalizedOverVariable(SEM_vertex * v);
	// clique is defined over the vertex pair (X,Y)
	// no of variables is always 2 for bifurcating tree-structured DAGs
	double scalingFactor;
	emtr::Md factor;
	emtr::Md initialPotential;
	emtr::Md basePotential;  // Store base potential (without site-specific evidence)
	emtr::Md belief;
	// P(X,Y)

	// Memoization for message passing
	vector<SEM_vertex*> subtreeLeaves;  // Observed variables in subtree (for upward messages)
	vector<SEM_vertex*> complementLeaves;  // Observed variables NOT in subtree (for downward messages)
	map<vector<int>, pair<std::array<double,4>, double>> messageCache;  // Cache: pattern signature -> (message, logScalingFactor)
	map<vector<int>, pair<std::array<double,4>, double>> downwardMessageCache;  // Cache for downward messages
	bool isLeafClique = false;  // True if this clique contains an observed variable

	// Per-edge statistics for cache analysis
	int upwardCacheHits = 0;
	int upwardCacheMisses = 0;
	int downwardCacheHits = 0;
	int downwardCacheMisses = 0;

	void SetInitialPotentialAndBelief(int site);
	void SetBasePotential();  // Set base potential once (transition matrix, root prior)
	void ApplyEvidenceForSite(int site);  // Apply site-specific evidence to leaf cliques
	void ResetForNewSite();  // Reset messages and scaling factors for new site
	void ComputeSubtreeLeaves();  // Compute which observed variables are in subtree
	vector<int> GetSubtreeSignature(int site);  // Get pattern signature for subtree at given site
	vector<int> GetComplementSignature(int site);  // Get pattern signature for complement (non-subtree) leaves
	void ClearMessageCache();  // Clear the message cache
	void ClearCacheStatistics();  // Clear per-edge statistics

	// If the clique contains an observed variable then initializing
	// the potential is the same as restricting the corresponding
	// CPD to row corresponding to observed variable
	void AddNeighbor(clique * C);
	clique (SEM_vertex * x, SEM_vertex * y) {
		this->x = x;
		this->y = y;
		this->name = to_string(x->id) + "-" + to_string(y->id);
		this->logScalingFactorForClique = 0;
		this->isLeafClique = false;
	}

	~clique () {

	}
};



std::array <double, 4> clique::MarginalizeOverVariable(SEM_vertex * v) {
	std::array <double, 4> message;	
	if (this->x == v) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			message[dna_y] = 0;
			for (int dna_x = 0; dna_x < 4; dna_x ++) {
				message[dna_y] += this->belief[dna_x][dna_y];
			}
		}
	} else if (this->y == v) {
		for (int dna_x = 0; dna_x < 4; dna_x ++) {
			message[dna_x] = 0;
			for (int dna_y = 0; dna_y < 4; dna_y ++) {
				message[dna_x] += this->belief[dna_x][dna_y];
			}
		}
	} else {
		cout << "Check marginalization over variable" << endl;
	}
	return (message);
}

void clique::ComputeBelief() {
	this->factor = this->initialPotential;
	vector <clique *> neighbors = this->children;
	std::array <double, 4> messageFromNeighbor;
	bool debug = 0;		
	// if (this->y->name == "Mouse" && this->root_variable->name == "h_4" && this->site == 257) {debug = 1; cout << "..........................................................................." << endl;}
	if (this->parent != this) {
		neighbors.push_back(this->parent);
	}
	// cout << "6a" << endl;
	for (clique * C_neighbor : neighbors) {		
		this->logScalingFactorForClique += this->logScalingFactorForMessages[C_neighbor];
		messageFromNeighbor = this->messagesFromNeighbors[C_neighbor];
		for (int i = 0; i < 4; i ++) if (isnan(messageFromNeighbor[i])) throw mt_error("message contain nan");

		//
		if (this->y == C_neighbor->x or this->y == C_neighbor->y) {
		    //	factor_row_i = factor_row_i (dot) message
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) for (int dna_y = 0 ; dna_y < 4; dna_y ++) this->factor[dna_x][dna_y] *= messageFromNeighbor[dna_y];
			// scale factor
			this->scalingFactor = 0;
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) for (int dna_y = 0 ; dna_y < 4; dna_y ++) this->scalingFactor += this->factor[dna_x][dna_y];			
			assert(scalingFactor > 0);
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) for (int dna_y = 0 ; dna_y < 4; dna_y ++) this->factor[dna_x][dna_y] /= this->scalingFactor;
			this->logScalingFactorForClique += log(scalingFactor);
			if (debug) cout << "Performing row-wise multiplication" << endl;
		} else if (this->x == C_neighbor->x or this->x == C_neighbor->y) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) for (int dna_x = 0 ; dna_x < 4; dna_x ++) this->factor[dna_x][dna_y] *= messageFromNeighbor[dna_x];
			// scale factor
			this->scalingFactor = 0;
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) for (int dna_y = 0 ; dna_y < 4; dna_y ++) this->scalingFactor += this->factor[dna_x][dna_y];			
			assert(scalingFactor > 0);
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) for (int dna_y = 0 ; dna_y < 4; dna_y ++) this->factor[dna_x][dna_y] /= this->scalingFactor;
			this->logScalingFactorForClique += log(scalingFactor);
			if (debug) cout << "Performing column-wise multiplication" << endl;
		} else {
			cout << "Check product step" << endl;
            throw mt_error("check product step");
		}
	}
	// cout << "6b" << endl;	
	this->scalingFactor = 0;
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) for (int dna_y = 0 ; dna_y < 4; dna_y ++) this->scalingFactor += this->factor[dna_x][dna_y];
	assert(scalingFactor > 0);
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) for (int dna_y = 0 ; dna_y < 4; dna_y ++) this->factor[dna_x][dna_y] /= this->scalingFactor;
	this->logScalingFactorForClique += log(this->scalingFactor);
	// cout << "6c" << endl;		
	// cout << "6d" << endl;
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) for (int dna_y = 0 ; dna_y < 4; dna_y ++) this->belief[dna_x][dna_y] = this->factor[dna_x][dna_y];
	
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {			
			if (isnan(this->belief[dna_x][dna_y])){
				throw mt_error("belief has nan");
			}
		}
	}
	// cout << "6e" << endl;
}

void clique::AddParent(clique * C) {
	this->parent = C;
	this->inDegree += 1; 
}

void clique::AddChild(clique * C) {
	this->children.push_back(C);
	this->outDegree += 1;
}

void clique::SetInitialPotentialAndBelief(int site) {
	this->site = site;
	// Initialize psi
	// V = (X,Y) X->Y (wlog), X is always an unobserved vertex
	int matchingCase = 0;
	(void)matchingCase; // suppress unused warning - used for debugging
	// Case 1. Y is an observed vertex
	// Product factor psi = P(Y|X) restricted to observed value xe of X
	// psi (y|x) is set to 0 if x != xe
	// of y->DNAcompressed[site] is 4 (gap) then transition matrix is not conditioned
	// if (y->observed && y->DNAcompressed[site] == 4) {
	// 	cout << "dealing with gap" << endl;
	// }

	if (y->observed) {
		this->initialPotential = y->transitionMatrix;
		int dna_y = y->DNAcompressed[site];
		if (dna_y == 4) {
			// cout << "case 1" << endl;
			matchingCase = 1; // gap
		} else {
			// cout << "case 2" << endl;
			matchingCase = 2;			
			for (int dna_p = 0; dna_p < 4; dna_p++) {
				for (int dna_c = 0; dna_c < 4; dna_c++) {
					if (dna_c != dna_y) {
						this->initialPotential[dna_p][dna_c] *= 0;
					} else {
						this->initialPotential[dna_p][dna_c] *= 1;
					}
				}
			}	
		}
	}
	
	// Case 2. X and Y are hidden and X is not the root
	// psi = P(Y|X)
	if (!y->observed) {
		// cout << "case 3" << endl;
		matchingCase = 3;
		this->initialPotential = y->transitionMatrix;		
	}	
	
	// Case 3. X and Y are hidden and X is the root and "this" is not the root clique
	// psi = P(Y|X)
	if (!y->observed and (x->parent == x) and (this->parent != this)) {
		// cout << "case 4" << endl;
		matchingCase = 4;
		this->initialPotential = y->transitionMatrix;
	}
	
	// Case 4. X and Y are hidden and X is the root and "this" is the root clique
	// psi = P(X) * P(Y|X) 
	if (!y->observed and (x->parent == x) and (this->parent == this)) {	
		// cout << "case 5" << endl;
		matchingCase = 5;
		this->initialPotential = y->transitionMatrix;
		for (int dna_p = 0; dna_p < 4; dna_p++) {
			for (int dna_c = 0; dna_c < 4; dna_c++) {
				this->initialPotential[dna_p][dna_c] *= x->rootProbability[dna_c];
			}
		}
	}

	bool all_zero_initial_potential = 1;
	bool all_zero_transition = 1;
	for(int i = 0; i < 4; i ++) {
		for (int j = 0; j < 4; j ++) {
			if(this->initialPotential[i][j] != 0) all_zero_initial_potential = 0;
			if(y->transitionMatrix[i][j] != 0) all_zero_transition = 0;
		}
	}

	if (all_zero_transition) {
		cout  << "all zero transition" << endl;
		exit(-1);
	}
	
	if (all_zero_initial_potential) {
		cout << y->name << endl;		
		int dna_y = y->DNAcompressed[site];		
		cout << dna_y << endl;
		cout  << "all_zero_initial_potential" << endl;
		for (int i = 0; i < 4; i ++) {
			for (int j = 0; j < 4; j ++) {
				cout << i << "\t" << j << "\t" << y->transitionMatrix[i][j] << endl;
			}
		}
		exit(-1);
	}

	// double maxValue = 0;
	for (int i = 0; i < 4; i ++) {
		for (int j = 0; j < 4; j ++) {			
			if(isnan(this->initialPotential[i][j])){
				cout << "initial potential has nan" << endl;
				throw mt_error("initial potential has nan");
			}
		}
	}
	
	this->belief = this->initialPotential;
	this->logScalingFactorForClique = 0;
	this->logScalingFactorForMessages.clear();
	this->messagesFromNeighbors.clear();
}

void clique::SetBasePotential() {
	// Set the base potential (transition matrix, root prior) - done once
	// This doesn't include site-specific evidence for observed variables

	if (!y->observed) {
		// Hidden child: base potential is just the transition matrix
		this->basePotential = y->transitionMatrix;

		// If X is root and this is the root clique, multiply by root prior
		if ((x->parent == x) && (this->parent == this)) {
			for (int dna_p = 0; dna_p < 4; dna_p++) {
				for (int dna_c = 0; dna_c < 4; dna_c++) {
					this->basePotential[dna_p][dna_c] *= x->rootProbability[dna_c];
				}
			}
		}
	} else {
		// Observed child: base potential is the transition matrix
		// Evidence will be applied per site
		this->basePotential = y->transitionMatrix;
	}
}

void clique::ApplyEvidenceForSite(int site) {
	// Apply site-specific evidence to the initial potential
	this->site = site;

	// Start with base potential
	this->initialPotential = this->basePotential;

	// If Y is observed, restrict to observed value at this site
	if (y->observed) {
		int dna_y = y->DNAcompressed[site];
		if (dna_y != 4) {  // Not a gap
			// Zero out columns that don't match observed value
			for (int dna_p = 0; dna_p < 4; dna_p++) {
				for (int dna_c = 0; dna_c < 4; dna_c++) {
					if (dna_c != dna_y) {
						this->initialPotential[dna_p][dna_c] = 0;
					}
				}
			}
		}
		// If gap (dna_y == 4), keep full transition matrix (no restriction)
	}
}

void clique::ResetForNewSite() {
	// Reset messages and scaling factors for new site computation
	this->logScalingFactorForClique = 0;
	this->logScalingFactorForMessages.clear();
	this->messagesFromNeighbors.clear();
}

void clique::ComputeSubtreeLeaves() {
	// Compute which observed (leaf) variables are in this clique's subtree
	// This is used for memoization of upward messages
	this->subtreeLeaves.clear();

	// Check if Y is observed (this makes it a leaf clique)
	if (this->y->observed) {
		this->isLeafClique = true;
		this->subtreeLeaves.push_back(this->y);
	}

	// Recursively collect leaves from children
	for (clique* child : this->children) {
		for (SEM_vertex* leaf : child->subtreeLeaves) {
			this->subtreeLeaves.push_back(leaf);
		}
	}

	// Sort by vertex id for consistent signature
	sort(this->subtreeLeaves.begin(), this->subtreeLeaves.end(),
		 [](SEM_vertex* a, SEM_vertex* b) { return a->id < b->id; });
}

vector<int> clique::GetSubtreeSignature(int site) {
	// Get the pattern of observed values in the subtree at given site
	vector<int> signature;
	signature.reserve(this->subtreeLeaves.size());
	for (SEM_vertex* leaf : this->subtreeLeaves) {
		signature.push_back(leaf->DNAcompressed[site]);
	}
	return signature;
}

vector<int> clique::GetComplementSignature(int site) {
	// Get the pattern of observed values NOT in the subtree (for downward messages)
	vector<int> signature;
	signature.reserve(this->complementLeaves.size());
	for (SEM_vertex* leaf : this->complementLeaves) {
		signature.push_back(leaf->DNAcompressed[site]);
	}
	return signature;
}

void clique::ClearMessageCache() {
	this->messageCache.clear();
	this->downwardMessageCache.clear();
}

void clique::ClearCacheStatistics() {
	this->upwardCacheHits = 0;
	this->upwardCacheMisses = 0;
	this->downwardCacheHits = 0;
	this->downwardCacheMisses = 0;
}

class cliqueTree {
public:
	vector < pair <clique *, clique *> > edgesForPreOrderTreeTraversal;
	vector < pair <clique *, clique *> > edgesForPostOrderTreeTraversal;
	vector < pair <clique *, clique *> > cliquePairsSortedWrtLengthOfShortestPath;
	map < pair <SEM_vertex *, SEM_vertex *>, emtr::Md> marginalizedProbabilitiesForVariablePair;
	map < pair <clique *, clique *>, pair <SEM_vertex *, SEM_vertex *>> cliquePairToVariablePair;
	int site;
	clique * root;
	bool rootSet;
	vector <clique *> leaves;
	vector <clique *> cliques;
	void CalibrateTree();
	void SetMarginalProbabilitesForEachEdgeAsBelief();
	void ComputeMarginalProbabilitesForEachVariablePair();	
	void ConstructSortedListOfAllCliquePairs();
	clique * GetLCA (clique * C_1, clique * C_2);
	int GetDistance(clique * C_1, clique * C_2);
	int GetDistanceToAncestor(clique * C_d, clique * C_a);
	void SetLeaves();
	void SetRoot();
	void AddEdge(clique * C_1, clique * C_2);
	void SendMessage(clique * C_1, clique * C_2);
	void AddClique(clique * C);
	void SetSite(int site);
	void InitializePotentialAndBeliefs();
	void InitializeBasePotentials();  // Set base potentials for all cliques once
	void ApplyEvidenceAndReset(int site);  // Apply evidence and reset for a specific site
	void SetEdgesForTreeTraversalOperations();
	void WriteCliqueTreeAndPathLengthForCliquePairs(string fileName);
	emtr::Md GetP_XZ(SEM_vertex * X, SEM_vertex * Y, SEM_vertex * Z);
	SEM_vertex * GetCommonVariable(clique * Ci, clique * Cj);
	tuple <SEM_vertex *,SEM_vertex *,SEM_vertex *> GetXYZ(clique * Ci, clique * Cj);

	// Memoization support
	void ComputeAllSubtreeLeaves();  // Compute subtree leaves for all cliques
	void ComputeAllComplementLeaves();  // Compute complement leaves for all cliques
	void ClearAllMessageCaches();  // Clear all message caches
	void ClearAllCacheStatistics();  // Clear per-edge statistics
	void SendMessageWithMemoization(clique * C_from, clique * C_to);  // Memoized version
	void CalibrateTreeWithMemoization();  // Calibrate using memoization
	void WriteCacheStatisticsToCSV(string filename);  // Write per-edge statistics to CSV
	vector<SEM_vertex*> allObservedLeaves;  // All observed variables in the tree
	int cacheHits = 0;  // Statistics for cache performance
	int cacheMisses = 0;
	int downwardCacheHits = 0;  // Statistics for downward message cache
	int downwardCacheMisses = 0;

	cliqueTree () {
		rootSet = 0;
		cacheHits = 0;
		cacheMisses = 0;
	}
	~cliqueTree () {
		for (clique * C: this->cliques) {
			delete C;
		}
		this->cliques.clear();
		this->leaves.clear();
	}
};

tuple <SEM_vertex *,SEM_vertex *,SEM_vertex *> cliqueTree::GetXYZ(clique * Ci, clique * Cj) {
	SEM_vertex * X; SEM_vertex * Y; SEM_vertex * Z;
	SEM_vertex * Y_temp;
	clique * Cl;
	pair <clique *, clique *> cliquePairToCheck;
	if (Ci->parent == Cj or Cj->parent == Ci) {
		// Case 1: Ci and Cj are neighbors
		Y = this->GetCommonVariable(Ci, Cj);
		if (Ci->y == Y){
		X = Ci->x;		
		} else {
			X = Ci->y;			
		}
		if (Cj->y == Y){			
			Z = Cj->x;
		} else {
			Z = Cj->y;
		}
		
	} else {
		// Case 2: Ci and Cj are not neighbors
		// Ci-...-Cl-Cj
		vector <clique *> neighbors;
		if (Cj->parent != Cj) {
			neighbors.push_back(Cj->parent);
		}
		for (clique * C: Cj->children) {
			neighbors.push_back(C);
		}
		
		Cl = Ci;
		
		for (clique * C: neighbors) {
			if (C->name < Ci->name) {
				cliquePairToCheck = pair <clique*, clique*>(C,Ci);
			} else {
				cliquePairToCheck = pair <clique*, clique*>(Ci,C);
			}
			if (this->cliquePairToVariablePair.find(cliquePairToCheck) != this->cliquePairToVariablePair.end()) {
				if (Ci == cliquePairToCheck.first) {
					Cl = cliquePairToCheck.second;
				} else {
					Cl = cliquePairToCheck.first;
				}
				break;
			}
		}		
		
		// Scope(Ci,Cl) = {X,Y}
		if (Ci->name < Cl->name) {
			tie(X,Y) = this->cliquePairToVariablePair[pair <clique*, clique*>(Ci,Cl)];
		} else {
			tie(Y,X) = this->cliquePairToVariablePair[pair <clique*, clique*>(Cl,Ci)];
		}			
				
				
		if (Cj->x == Y or Cj->y == Y){
			// Case 2a
			// Scope(Cj) = {Y,Z}
			if (Cj->x == Y) {
				Z = Cj->y;
			} else {
				Z = Cj->x;
			}
		} else {
			// Case 2b
			// Scope(Cl,Cj) = {Y,Z}
			if (Cl->name < Cj->name) {
				tie(Y_temp,Z) = this->cliquePairToVariablePair[pair <clique*, clique*>(Cl,Cj)];
			} else {
				tie(Z,Y_temp) = this->cliquePairToVariablePair[pair <clique*, clique*>(Cj,Cl)];
			}
			if (Y_temp != Y){
                throw mt_error("check Case 2b");
            }
		}
	}	
	
	return (tuple <SEM_vertex *,SEM_vertex *,SEM_vertex *>(X,Y,Z));
}

emtr::Md cliqueTree::GetP_XZ(SEM_vertex * X, SEM_vertex * Y, SEM_vertex * Z) {
	emtr::Md P_XY; emtr::Md P_YZ;
	emtr::Md P_ZGivenY; emtr::Md P_XZ;
	
    if (X->id < Y->id) {
		P_XY = this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(X,Y)];
	} else {		
		P_XY = emtr::MT(this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Y,X)]);
	}
	if (Y->id < Z->id) {
		P_YZ = this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Y,Z)];
	} else {		
		P_YZ = emtr::MT(this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Z,Y)]);
	}
//	cout << "P_XY is " << endl << P_XY << endl;
//	cout << "P_YZ is " << endl << P_YZ << endl;
	P_ZGivenY = emtr::Md{};
	double rowSum;
	for (int row = 0; row < 4; row ++) {		
		rowSum = 0;
		for (int col = 0; col < 4; col ++) {
			rowSum += P_YZ[row][col];
		}
		for (int col = 0; col < 4; col ++) {
			if (rowSum != 0){
				P_ZGivenY[row][col] = P_YZ[row][col]/rowSum;
			}			
		}
	}
	
//	cout << "P_ZGivenY is " << endl << P_ZGivenY << endl;
	
	for (int row = 0; row < 4; row ++) {		
		for (int col = 0; col < 4; col ++) {
			P_XZ[row][col] = 0;
		}
	}
	
	for (int dna_y = 0; dna_y < 4; dna_y ++) {		
		for (int dna_x = 0; dna_x < 4; dna_x ++) {
			for (int dna_z = 0; dna_z < 4; dna_z ++) {					
				// Sum over Y
				P_XZ[dna_x][dna_z] += P_XY[dna_x][dna_y] * P_ZGivenY[dna_x][dna_z];
			}
		}
	}
	
	return (P_XZ);
}

SEM_vertex * cliqueTree::GetCommonVariable(clique * Ci, clique * Cj) {
	SEM_vertex * commonVariable;
	if (Ci->x == Cj->x or Ci->x == Cj->y) {
		commonVariable = Ci->x;
	} else {
		commonVariable = Ci->y;
	}
	return (commonVariable);
}


void cliqueTree::ConstructSortedListOfAllCliquePairs() {
	this->cliquePairsSortedWrtLengthOfShortestPath.clear();
	vector < tuple <int, clique*, clique*>> sortedPathLengthAndCliquePair;
	int pathLength;
	for (clique * Ci : this->cliques) {
		for (clique * Cj : this->cliques) {
			if (Ci->name < Cj->name) {
				if (Ci->outDegree > 0 or Cj->outDegree > 0) {
					pathLength = this->GetDistance(Ci, Cj);
					sortedPathLengthAndCliquePair.push_back(make_tuple(pathLength,Ci,Cj));
				}				
			}
		}
	}
	sort(sortedPathLengthAndCliquePair.begin(),sortedPathLengthAndCliquePair.end());
	clique * Ci; clique * Cj;
	for (tuple <int, clique*, clique*> pathLengthCliquePair : sortedPathLengthAndCliquePair) {
		Ci = get<1>(pathLengthCliquePair);
		Cj = get<2>(pathLengthCliquePair);
		this->cliquePairsSortedWrtLengthOfShortestPath.push_back(pair <clique *, clique *> (Ci,Cj));
	}
}

int cliqueTree::GetDistance(clique * C_1, clique * C_2) {
	clique * lca = this->GetLCA(C_1, C_2);
	int d;
	d = this->GetDistanceToAncestor(C_1,lca) + this->GetDistanceToAncestor(C_2,lca);
	return (d);
}

int cliqueTree::GetDistanceToAncestor(clique * C_d, clique* C_a) {
	int d = 0;
	clique * C_p;
	C_p = C_d;
	while (C_p != C_a) {
		C_p = C_p->parent;
		d += 1;
	}
	return (d);
}

clique * cliqueTree::GetLCA(clique * C_1, clique * C_2) {
	vector <clique *> pathToRootForC1;
	vector <clique *> pathToRootForC2;
	clique * C1_p;
	clique * C2_p;
	C1_p = C_1;
	C2_p = C_2;
	
	clique * C_r = this->edgesForPreOrderTreeTraversal[0].first;
	
	while (C1_p->parent != C1_p) {
		pathToRootForC1.push_back(C1_p);
		C1_p = C1_p->parent;
	}
	pathToRootForC1.push_back(C1_p);
	if (C1_p != C_r) {
		cout << "Check get LCA for C1" << endl;
	}
	
	while (C2_p->parent != C2_p) {
		pathToRootForC2.push_back(C2_p);
		C2_p = C2_p->parent;
	}
	pathToRootForC2.push_back(C2_p);
	if (C2_p != C_r) {
		cout << "Check get LCA for C2" << endl;
	}
	
	clique * lca;
	lca = C_1;
	
	for (clique * C : pathToRootForC1) {
		if (find(pathToRootForC2.begin(),pathToRootForC2.end(),C)!=pathToRootForC2.end()) {
			lca = C;
			break;
		}
	}		
	return (lca);	
}

void cliqueTree::SetMarginalProbabilitesForEachEdgeAsBelief() {
	this->marginalizedProbabilitiesForVariablePair.clear();
	//	Store P(X,Y) for each clique
	for (clique * C: this->cliques) {
		if (C->x->id < C->y->id) {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, emtr::Md>(pair<SEM_vertex *, SEM_vertex *>(C->x,C->y),C->belief));
		} else {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, emtr::Md>(pair<SEM_vertex *, SEM_vertex *>(C->y,C->x),emtr::MT(C->belief)));
		}
	}
}

void cliqueTree::ComputeMarginalProbabilitesForEachVariablePair() {	
	this->marginalizedProbabilitiesForVariablePair.clear();
	this->cliquePairToVariablePair.clear();
	// For each clique pair store variable pair 
	// Iterate over clique pairs in order of increasing distance in clique tree	
	
	clique * Ci; clique * Cj;
	
	SEM_vertex * X; SEM_vertex * Z;
	SEM_vertex * Y;

	emtr::Md P_XZ;
		
	//	Store P(X,Y) for each clique
	for (clique * C: this->cliques) {
		if (C->x->id < C->y->id) {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, emtr::Md>(pair<SEM_vertex *, SEM_vertex *>(C->x,C->y),C->belief));
		} else {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, emtr::Md>(pair<SEM_vertex *, SEM_vertex *>(C->y,C->x),emtr::MT(C->belief)));
			
		}
	}
	
	for (pair <clique *, clique *> cliquePair : this->cliquePairsSortedWrtLengthOfShortestPath) {
		tie (Ci, Cj) = cliquePair;
		tie (X, Y, Z) = this->GetXYZ(Ci, Cj);		
		this->cliquePairToVariablePair.insert(pair <pair <clique *, clique *>,pair <SEM_vertex *, SEM_vertex *>>(pair <clique *, clique *>(Ci,Cj), pair <SEM_vertex *, SEM_vertex *>(X,Z)));
		P_XZ = this->GetP_XZ(X, Y, Z);
		if (X->id < Z->id) {
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, emtr::Md>(pair<SEM_vertex *, SEM_vertex *>(X,Z),P_XZ));			
		} else {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, emtr::Md>(pair<SEM_vertex *, SEM_vertex *>(Z,X),emtr::MT(P_XZ)));
		}
	}
}


void cliqueTree::SetRoot() {
	for (clique * C: this->cliques) {
		if (C->inDegree == 0) {
			this->root = C;
		}
	}
}

void cliqueTree::InitializePotentialAndBeliefs() {
	for (clique * C: this->cliques) {
		C->SetInitialPotentialAndBelief(this->site);
	}
}

void cliqueTree::InitializeBasePotentials() {
	// Set base potentials for all cliques (done once before iterating over sites)
	for (clique * C: this->cliques) {
		C->SetBasePotential();
	}
}

void cliqueTree::ApplyEvidenceAndReset(int site) {
	// Apply site-specific evidence and reset for new computation
	this->site = site;
	for (clique * C: this->cliques) {
		C->ApplyEvidenceForSite(site);
		C->ResetForNewSite();
	}
}

void cliqueTree::SetSite(int site) {
	this->site = site;
}

void cliqueTree::AddClique(clique * C) {
	this->cliques.push_back(C);
}

void cliqueTree::AddEdge(clique * C_1, clique * C_2) {
	C_1->AddChild(C_2);
	C_2->AddParent(C_1);
}

void cliqueTree::SetEdgesForTreeTraversalOperations() {
	for (clique * C : this->cliques) {
		C->timesVisited = 0;
	}
	this->edgesForPostOrderTreeTraversal.clear();
	this->edgesForPreOrderTreeTraversal.clear();
	vector <clique *> verticesToVisit;
	verticesToVisit = this->leaves;
	clique * C_child; clique * C_parent;
	int numberOfVerticesToVisit = verticesToVisit.size();
	
	while (numberOfVerticesToVisit > 0) {
		C_child = verticesToVisit[numberOfVerticesToVisit - 1];
		verticesToVisit.pop_back();
		numberOfVerticesToVisit -= 1;
		C_parent = C_child->parent;
		if (C_child != C_parent) {
			C_parent->timesVisited += 1;
			this->edgesForPostOrderTreeTraversal.push_back(make_pair(C_parent, C_child));
			if (C_parent->timesVisited == C_parent->outDegree) {				
				verticesToVisit.push_back(C_parent);
				numberOfVerticesToVisit += 1;				
			}
		}
	}
	
	for (int edgeInd = this->edgesForPostOrderTreeTraversal.size() -1; edgeInd > -1; edgeInd --) {
		this->edgesForPreOrderTreeTraversal.push_back(this->edgesForPostOrderTreeTraversal[edgeInd]);
	}
}

void cliqueTree::SetLeaves() {
	this->leaves.clear();
	for (clique * C: this->cliques) {
		if (C->outDegree == 0) {
			this->leaves.push_back(C);
		}		
	}
}


void cliqueTree::SendMessage(clique * C_from, clique * C_to) {	
	double logScalingFactor;
	double largestElement;	
	array <double, 4> messageFromNeighbor;
	array <double, 4> messageToNeighbor;
	bool verbose = 0;
	if (verbose) {
		cout << "Preparing message to send from " << C_from->x->name << "," << C_from->y->name << " to " ;
		cout << C_to->x->name << "," << C_to->y->name << " is " << endl;
	}
	
	// Perform the three following actions
	
	// A) Compute product: Multiply the initial potential of C_from
	// with messages from all neighbors of C_from except C_to, and
	
	// B) Compute sum: Marginalize over the variable that
	// is in C_from but not in C_to
	
	// C) Transmit: sending the message to C_to
	
	// Select neighbors
	vector <clique *> neighbors;
	if (C_from->parent != C_from and C_from->parent != C_to) {
		neighbors.push_back(C_from->parent);
	}
	
	for (clique * C_child : C_from->children) {
		if (C_child != C_to) {
			neighbors.push_back(C_child);
		}
	}
	
	emtr::Md factor;
	factor = C_from->initialPotential;
	
	logScalingFactor = 0;
		// A. PRODUCT: Multiply messages from neighbors that are not C_to
	// cout << "A step" << endl;
	bool allZero = 1;
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			if (factor[dna_x][dna_y] != 0) allZero = 0;
			// cout << "initial factor for " << dna_x << "," << dna_y << " " << factor[dna_x][dna_y] << endl;
			
		}
	}
	if (allZero) {
		cout << "initial potential is all zero";
		throw mt_error("all zero in factor");
	}

	for (clique * C_neighbor : neighbors) {
		messageFromNeighbor = C_from->messagesFromNeighbors[C_neighbor];
		// for (int i = 0; i < 4; i ++) isnan(messageFromNeighbor)
		if (C_from->y == C_neighbor->x or C_from->y == C_neighbor->y) {
		// factor_row_i = factor_row_i (dot) message
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
				for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
					factor[dna_x][dna_y] *= messageFromNeighbor[dna_y];	
				}
			}
			if (verbose) {cout << "Performing row-wise multiplication" << endl;}
		} else if (C_neighbor->x == C_from->x or C_neighbor->y == C_from->x) {
		// factor_col_i = factor_col_i (dot) message
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
					factor[dna_x][dna_y] *= messageFromNeighbor[dna_x];
				}
			}
			if (verbose) {cout << "Performing column-wise multiplication" << endl;}
		} else {
			cout << "Check product step" << endl;
			throw mt_error("check product step");
		}		
		// Check to see if each entry in the factor is zero
		allZero = 1;	
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				if (factor[dna_x][dna_y] != 0) {
					allZero = 0;
				}
			}
		}
		if (allZero) {
			cout << "all zero in factor" << endl;
			throw mt_error("all zero in factor");}		
		// Rescale factor
		largestElement = 0;
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				if (largestElement < factor[dna_x][dna_y]) {
					largestElement = factor[dna_x][dna_y];
				}
			}
		}
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				factor[dna_x][dna_y] /= largestElement;
			}
		}
		logScalingFactor += log(largestElement);
		logScalingFactor += C_from->logScalingFactorForMessages[C_neighbor];
	}

	// cout << "B step" << endl;
	allZero = 1;
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			if (factor[dna_x][dna_y] != 0) allZero = 0;
		}
	}

	if (allZero) {
			cout << "all zero in factor" << endl;
			throw mt_error("all zero in factor");};
	// B. SUM
		// Marginalize factor by summing over common variable
	largestElement = 0;
	if (C_from->y == C_to->x or C_from->y == C_to->y) {
		// Sum over C_from->x		
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			messageToNeighbor[dna_y] = 0;
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
				messageToNeighbor[dna_y] += factor[dna_x][dna_y];
			}
		}
		if (verbose) {
			cout << "Performing column-wise summation" << endl;
		}							
	} else if (C_from->x == C_to->x or C_from->x == C_to->y) {
		// Sum over C_from->y		
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			messageToNeighbor[dna_x] = 0;
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {				
				messageToNeighbor[dna_x] += factor[dna_x][dna_y];
			}
		}
		if (verbose) {
			cout << "Performing row-wise summation" << endl;
		}							
	} else {		
		cout << "Check sum step" << endl;
		throw mt_error("Check sum step");
	}
	// Rescale message to neighbor
	largestElement = 0;
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		if (largestElement < messageToNeighbor[dna_x]) {
			largestElement = messageToNeighbor[dna_x];
		}
	}
	if (largestElement == 0) {
		cout << "Division by zero" << endl;
		throw mt_error("Division by zero");
	}
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		messageToNeighbor[dna_x] /= largestElement;
	}
	logScalingFactor += log(largestElement);
	// cout << "check point 1" << endl;
	if (verbose) {
		cout << "Sending the following message from " << C_from->name << " to " << C_to->name << endl;
		for (int dna = 0; dna < 4; dna ++) {
			cout << "dna " << dna << endl;
			cout << to_string(messageToNeighbor[dna]) << endl;			
			cout << isnan(messageToNeighbor[dna]) << endl;
			if(isnan(messageToNeighbor[dna])) {
				cout << "messageToNeighbor[" << dna << "] is not a number";
				throw mt_error("Check message to neighbor");
			} else {
				cout << messageToNeighbor[dna] << endl;
			}
		}
	}
	// cout << "check point 2" << endl;
	// cout << endl;
	
	// cout << "C step" << endl;
	// C. TRANSMIT
	if(isnan(logScalingFactor)) {
		cout << "logScalingFactor is nan" << endl;
		throw mt_error("logScalingFactor is nan");
	}
	C_to->logScalingFactorForMessages.insert(make_pair(C_from,logScalingFactor));
	C_to->messagesFromNeighbors.insert(make_pair(C_from,messageToNeighbor));
}

void cliqueTree::CalibrateTree() {
	clique * C_p; clique * C_c;

	//	Send messages from leaves to root
	// cout << "13a" << endl;
	for (pair <clique *, clique *> cliquePair : this->edgesForPostOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		this->SendMessage(C_c, C_p);
	}

	// cout << "13b" << endl;

	//	Send messages from root to leaves
	for (pair <clique *, clique *> cliquePair : this->edgesForPreOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		this->SendMessage(C_p, C_c);
	}
	//	Compute beliefs
	// cout << "13c" << endl;
	for (clique * C: this->cliques) {
		C->ComputeBelief();
	}

	// cout << "13d" << endl;
}

void cliqueTree::ComputeAllSubtreeLeaves() {
	// Compute subtree leaves for all cliques in post-order (leaves first)
	// This ensures children's subtree info is available when computing parent's
	clique * C_p; clique * C_c;
	for (pair <clique *, clique *> cliquePair : this->edgesForPostOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		C_c->ComputeSubtreeLeaves();
	}
	// Don't forget the root clique
	this->root->ComputeSubtreeLeaves();

	// Store all observed leaves from root's subtree (which is the entire tree)
	this->allObservedLeaves = this->root->subtreeLeaves;
}

void cliqueTree::ComputeAllComplementLeaves() {
	// Compute complement leaves for each clique (all observed variables NOT in its subtree)
	// This is used for memoization of downward messages
	// Must be called after ComputeAllSubtreeLeaves()

	for (clique * C : this->cliques) {
		C->complementLeaves.clear();

		// Complement = allObservedLeaves - subtreeLeaves
		// Both are sorted by vertex id, so we can use set difference
		set<SEM_vertex*> subtreeSet(C->subtreeLeaves.begin(), C->subtreeLeaves.end());

		for (SEM_vertex* leaf : this->allObservedLeaves) {
			if (subtreeSet.find(leaf) == subtreeSet.end()) {
				C->complementLeaves.push_back(leaf);
			}
		}
		// Already sorted since allObservedLeaves is sorted
	}
}

void cliqueTree::ClearAllMessageCaches() {
	for (clique * C: this->cliques) {
		C->ClearMessageCache();
	}
	this->cacheHits = 0;
	this->cacheMisses = 0;
	this->downwardCacheHits = 0;
	this->downwardCacheMisses = 0;
}

void cliqueTree::SendMessageWithMemoization(clique * C_from, clique * C_to) {
	// Memoize both upward and downward messages based on relevant observed variables

	bool isUpwardMessage = (C_from->parent == C_to);

	if (isUpwardMessage && !C_from->subtreeLeaves.empty()) {
		// UPWARD MESSAGE: depends on observed variables in C_from's subtree
		vector<int> signature = C_from->GetSubtreeSignature(this->site);

		// Check if we have this message cached
		auto it = C_from->messageCache.find(signature);
		if (it != C_from->messageCache.end()) {
			// Cache hit! Reuse the cached message
			this->cacheHits++;
			C_from->upwardCacheHits++;  // Per-edge tracking
			C_to->messagesFromNeighbors[C_from] = it->second.first;
			C_to->logScalingFactorForMessages[C_from] = it->second.second;
			return;
		}

		// Cache miss - compute the message
		this->cacheMisses++;
		C_from->upwardCacheMisses++;  // Per-edge tracking
		this->SendMessage(C_from, C_to);

		// Cache the result
		C_from->messageCache[signature] = make_pair(
			C_to->messagesFromNeighbors[C_from],
			C_to->logScalingFactorForMessages[C_from]
		);
	} else if (!isUpwardMessage && !C_to->complementLeaves.empty()) {
		// DOWNWARD MESSAGE: depends on observed variables NOT in C_to's subtree
		// The message from parent to child depends on all evidence outside child's subtree
		vector<int> signature = C_to->GetComplementSignature(this->site);

		// Check if we have this downward message cached
		auto it = C_to->downwardMessageCache.find(signature);
		if (it != C_to->downwardMessageCache.end()) {
			// Cache hit! Reuse the cached message
			this->downwardCacheHits++;
			C_to->downwardCacheHits++;  // Per-edge tracking
			C_to->messagesFromNeighbors[C_from] = it->second.first;
			C_to->logScalingFactorForMessages[C_from] = it->second.second;
			return;
		}

		// Cache miss - compute the message
		this->downwardCacheMisses++;
		C_to->downwardCacheMisses++;  // Per-edge tracking
		this->SendMessage(C_from, C_to);

		// Cache the result in the child's downward cache
		C_to->downwardMessageCache[signature] = make_pair(
			C_to->messagesFromNeighbors[C_from],
			C_to->logScalingFactorForMessages[C_from]
		);
	} else {
		// No memoization possible
		this->SendMessage(C_from, C_to);
	}
}

void cliqueTree::ClearAllCacheStatistics() {
	for (clique * C: this->cliques) {
		C->ClearCacheStatistics();
	}
}

void cliqueTree::WriteCacheStatisticsToCSV(string filename) {
	ofstream outfile(filename);
	outfile << "clique_edge,subtree_size,complement_size,hit_rate,direction" << endl;

	// Write upward message statistics (from child to parent)
	for (pair<clique*, clique*> cliquePair : this->edgesForPostOrderTreeTraversal) {
		clique* C_p; clique* C_c;
		tie(C_p, C_c) = cliquePair;

		int total = C_c->upwardCacheHits + C_c->upwardCacheMisses;
		double hit_rate = (total > 0) ? (100.0 * C_c->upwardCacheHits / total) : 0.0;
		int subtree_size = C_c->subtreeLeaves.size();

		// Use vertex names instead of IDs for better readability
		string edge_name = C_c->x->name + "-" + C_c->y->name;
		outfile << edge_name << "," << subtree_size << ","
		        << C_c->complementLeaves.size() << ","
		        << hit_rate << ",upward" << endl;
	}

	// Write downward message statistics (from parent to child)
	for (pair<clique*, clique*> cliquePair : this->edgesForPreOrderTreeTraversal) {
		clique* C_p; clique* C_c;
		tie(C_p, C_c) = cliquePair;

		int total = C_c->downwardCacheHits + C_c->downwardCacheMisses;
		double hit_rate = (total > 0) ? (100.0 * C_c->downwardCacheHits / total) : 0.0;
		int complement_size = C_c->complementLeaves.size();

		// Use vertex names instead of IDs for better readability
		string edge_name = C_c->x->name + "-" + C_c->y->name;
		outfile << edge_name << "," << C_c->subtreeLeaves.size() << ","
		        << complement_size << ","
		        << hit_rate << ",downward" << endl;
	}

	outfile.close();
	cout << "Cache statistics written to " << filename << endl;
}

void cliqueTree::CalibrateTreeWithMemoization() {
	clique * C_p; clique * C_c;

	//	Send messages from leaves to root (upward - memoize based on subtree)
	for (pair <clique *, clique *> cliquePair : this->edgesForPostOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		this->SendMessageWithMemoization(C_c, C_p);
	}

	//	Send messages from root to leaves (downward - memoize based on complement)
	for (pair <clique *, clique *> cliquePair : this->edgesForPreOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		this->SendMessageWithMemoization(C_p, C_c);  // Memoized for statistics
	}

	//	Compute beliefs
	for (clique * C: this->cliques) {
		C->ComputeBelief();
	}
}

struct EM_struct {
	string method;
	// init method  - store before starting rep
	int rep;
	// rep          - store at start of each rep
	int iter;   
	map <int, double> ecd_ll_per_iter;
	// ecd for each iteration of EM
	double ll_init;
	// ll initial store at completion of EM run
	double ll_final;
	// ll final store at completion of EM run
	string root;
	// init root_prob - store for each rep when params are initialized
	array <double,4> root_prob_init;
	// final root_prob - store if log likelihood score is maximum
	array <double,4> root_prob_final;
	// init trans_prob - store transition probability for each child node for each rep when params are initialized
	map <string, Md> trans_prob_init;	
	// init trans_prob - store transition probability for each child node for each rep when params are initialized
	map <string, Md> trans_prob_final;
};

struct EMTrifle_struct {
	vector <int> iter_trifle;
	vector <map <int,double>> ecd_ll_per_iter_for_trifle;
	vector <double> ll_initial_trifle;
	vector <double> ll_final_trifle;
	vector <map <string, Md>> trans_prob_final_trifle;
	vector <array <double, 4>> root_prob_final_trifle;
	vector <map <string, Md>> trans_prob_initial_trifle;
	vector <array <double, 4>> root_prob_initial_trifle;	
	string method;
	
	int rep;
	string root;
	EMTrifle_struct()
	: ecd_ll_per_iter_for_trifle(3),
	  iter_trifle(3),
	  ll_initial_trifle(3, numeric_limits<double>::quiet_NaN()),
	  ll_final_trifle(3, numeric_limits<double>::quiet_NaN()),
	  trans_prob_final_trifle(3),
	  root_prob_final_trifle(3),
	  trans_prob_initial_trifle(3),
	  root_prob_initial_trifle(3)
	{}
};

class SEM {
private:	

public:	
	array <int,3> num_patterns_for_layer;	
	vector <vector<int>> DNAPatternOriginalSites;
	vector <int> OriginalToCompressed;
	int compressedToFirstOriginalSite(size_t k) const;
	const vector<int>& compressedToAllOriginalSites(size_t k) const;
	int originalToCompressed(size_t origSite) const;
	bool spectral_ready_ = false;	
	// Graph T;		
	array <double, 4> alpha_pi;
	array <double, 4> alpha_M_row;
	array <double, 4> sample_dirichlet(const array <double, 4>& alpha, mt19937_64& gen);
	int largestIdOfVertexInMST = 1;
	default_random_engine generator;
	bool setParameters;
	bool verbose = 0;
	string modelForRooting;
	map <string,int> mapDNAtoInteger;
	map <int, SEM_vertex*> * vertexMap;
	vector <SEM_vertex*> vertices;
	vector <SEM_vertex*> non_root_vertices;
	map <pair<SEM_vertex *,SEM_vertex *>,emtr::Md> * M_hss;
	vector <int> DNAPatternWeights;
	vector <double> DNAPatternGapProp;
	vector <int> DNAPatternUniqueCount;
	vector <double> CumPercDNAPatternWeights;
	vector <int> AAPatternWeights;
	vector <pair<int,double>> AAsitecumweight; 
	vector <int> gaplesscompressedDNAsites;
	vector <bool> gapLessDNAFlag;
	vector <bool> gapLessAAFlag;
	vector <pattern *> patterns;
	PackedPatternStorage* packed_patterns = nullptr;
    vector <int> pattern_weights;
    int num_patterns_from_file = 0;
	int rep;	
	void addToZeroEntriesOfTransitionMatrix(emtr::Md& P);
	void addToZeroEntriesOfRootProbability(array<double,4>& p);
	int num_aa_patterns;
	int num_repetitions;	
	double thrCoarse = 50.0;
	double thrMedium = 30.0;
	double thrFine = 20.0;	
	double cum_pattern_weight_coarse;
	double cum_pattern_weight_medium;
	double cum_pattern_weight_fine = 100.0;
	vector<pair<int,int>> AAsite_and_weight;
	vector <double> cum_fractional_weight_AA;
	vector <vector <int> > sitePatternRepetitions;
	vector <int> sortedDeltaGCThresholds;
	int numberOfInputSequences;
	int numberOfVerticesInSubtree;
	int numberOfObservedVertices;
	int numberOfExternalVertices = 0;	
	int num_dna_patterns;
	int maxIter;
	double patW_coarse;
	double patW_medium;
	double patW_fine;
	double frac_iter_coarse;
	double frac_iter_medium;
	double frac_iter_fine;
	double ecdllConvergenceThreshold = 0.1;	
	vector <double> ecdllConvergenceThresholdForTrifle;
	double sumOfExpectedLogLikelihoods = 0;
	double maxSumOfExpectedLogLikelihoods = 0;
	int node_ind = 1;
	chrono::system_clock::time_point t_start_time;
	chrono::system_clock::time_point t_end_time;
	// ofstream * logFile;
	SEM_vertex * root;
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForPostOrderTreeTraversal;
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForPreOrderTreeTraversal;	
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForChowLiuTree;
	vector < pair <SEM_vertex *, SEM_vertex *>> directedEdgeList;
	map <pair <SEM_vertex *, SEM_vertex *>, double> edgeLengths;
	vector < SEM_vertex *> leaves;
	vector < SEM_vertex *> preOrderVerticesWithoutLeaves;
	vector < SEM_vertex *> postOrderVertices;
	map < pair <SEM_vertex * , SEM_vertex *>, emtr::Md > expectedCountsForVertexPair;
	map < pair <SEM_vertex * , SEM_vertex *>, emtr::Md > posteriorProbabilityForVertexPair;
	map < SEM_vertex *, array <double,4>> expectedCountsForVertex; 
	map < SEM_vertex *, array <double,4>> posteriorProbabilityForVertex;
	map <int, emtr::Md> rateMatrixPerRateCategory;
	map <int, emtr::Md> rateMatrixPerRateCategory_stored;
	map <int, double> scalingFactorPerRateCategory;
	map <int, double> scalingFactorPerRateCategory_stored;
	int numberOfRateCategories = 0;
	double maximumLogLikelihood;
	double max_log_likelihood_diri;
	double max_log_likelihood_pars;
	double max_log_likelihood_rep;
	double max_log_likelihood_hss;
	double max_log_likelihood_best;
	double F81_mu;
	emtr::Md I4by4;	
	cliqueTree * cliqueT;
	bool debug;
	bool finalIterationOfSEM;	
	bool flag_JC = 0;	
	map <string, int> nameToIdMap;
	string DNAsequenceFileName;
	string AAsequenceFileName;
	string phylip_file_name;
	string topologyFileName;
	string probabilityFileName;
	string probabilityFileName_best;
	string probabilityFileName_pars;
	string probabilityFileName_hss;
	string probabilityFileName_diri;
	string prefix_for_output_files;	
	double gap_proportion_in_pattern(const vector<int>& pat);
	int unique_non_gap_count_in_pattern(const std::vector<int>& pat);
	double sequenceLength;
	// Add vertices (and compressed sequence for leaves)
	array <double, 4> rootProbability;
	array <double, 4> rootProbability_stored;
	SEM_vertex * root_stored;
	vector <int> compressedSequenceToAddToMST;
	string nameOfSequenceToAddToMST;
	double AAlogLikelihood;
	double logLikelihood;
	double logLikelihood_exp_counts;
	double logLikelihood_current;
	// Used for updating MST
	vector <int> indsOfVerticesOfInterest;
	vector <int> indsOfVerticesToKeepInMST;
	vector <int> idsOfVerticesOfInterest;
	vector <int> idsOfObservedVertices;	
	vector <int> idsOfVerticesToRemove;
	vector <int> idsOfVerticesToKeepInMST;	
	vector <int> idsOfExternalVertices;	
	vector < tuple <int, string, vector <int> > > idAndNameAndSeqTuple;
	vector <tuple <string,string,int,int,double,double,double,double>> EMTR_results;
	vector <tuple <string,string,int,int,int,double,double,double,double>> EMTrifle_results;
	// Used for updating global phylogenetic tree
	vector < pair <int, int> > edgesOfInterest_ind;	
	vector < pair < vector <int>, vector <int> > > edgesOfInterest_seq;
	string weightedEdgeListString;
	map < string, vector <int>> sequencesToAddToGlobalPhylogeneticTree;
	vector < tuple <string, string, double>> weightedEdgesToAddToGlobalPhylogeneticTree;
	vector < tuple <string, string, double>> edgeLogLikelihoodsToAddToGlobalPhylogeneticTree;		
	map <string, double> vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree;
	map <pair<SEM_vertex *,SEM_vertex *>,double> edgeLogLikelihoodsMap;
	SEM_vertex * externalVertex;
	void ReadPatternsFromFile(string pattern_file_name, string taxon_order_file_name);
    void ComputeLogLikelihoodUsingPatterns();
    void ComputeLogLikelihoodUsingPatternsWithPropagation();
    void ComputeLogLikelihoodUsingPatternsWithPropagationAtClique(clique* target_clique);
    void ComputeLogLikelihoodUsingPatternsWithPropagationOptimized();
    void ComputeLogLikelihoodUsingPatternsWithPropagationMemoized();
    void VerifyLogLikelihoodAtAllCliques();
	void AddArc(SEM_vertex * from, SEM_vertex * to);
	void RemoveArc(SEM_vertex * from, SEM_vertex * to);
	void ClearDirectedEdges();
	void ClearUndirectedEdges();
	void ClearAllEdges();
	void AddVertex(string name, vector <int> compressedSequenceToAdd);		
	void AddWeightedEdgesFromFile(const string edgeListFileName);
	void AddEdgeLogLikelihoods(vector<tuple<string,string,double>> edgeLogLikelihoodsToAdd);
	void AddExpectedCountMatrices(map < pair <SEM_vertex * , SEM_vertex *>, emtr::Md > expectedCountsForVertexPair);	
	void AddSitePatternWeights(vector <int> sitePatternWeightsToAdd);
	void AddSitePatternRepeats(vector <vector <int> > sitePatternRepetitionsToAdd);
	void AddSequences(vector <vector <int>> sequencesToAdd);	
	void AddRootVertex();	
	void SetVertexVector();
	void SetVertexVectorExceptRoot();
	void AddAllSequences(string sequencesFileName);
	void AddNames(vector <string> namesToAdd);
	void RootedTreeAlongAnEdgeIncidentToCentralVertex();
	void RootTreeAlongAnEdgePickedAtRandom();
	void RootTreeAtAVertexPickedAtRandom();	
	void EMTrifle_started_with_parsimony_rooted_at(SEM_vertex *v, int layer);
	void EMTrifle_started_with_dirichlet_rooted_at(SEM_vertex *v, int layer);
	void EMTrifle_started_with_HSS_rooted_at(SEM_vertex *v, int layer);
	tuple<int,double,double,double,double> EM_started_with_parsimony_rooted_at(SEM_vertex *v);
	tuple<int,double,double,double,double> EM_started_with_dirichlet_rooted_at(SEM_vertex *v);	
	tuple<int,double,double,double,double> EM_started_with_HSS_parameters_rooted_at(SEM_vertex *v);
	void StoreParamsInEMCurrent(string init_or_final);
	void StoreInitialParamsInEMTrifleCurrent(int layer);
	void StoreFinalParamsInEMTrifleCurrent(int layer);
	void ComputeSumOfExpectedLogLikelihoods();
	void RootTreeAlongEdge(SEM_vertex * u, SEM_vertex * v);
	void InitializeTransitionMatricesAndRootProbability();
	void ComputeMPEstimateOfAncestralSequences();
	void ComputeMPEstimateOfAncestralSequencesForTrifle(int layer);
	void ComputeMAPEstimateOfAncestralSequences();
	void ComputeMAPEstimateOfAncestralSequencesUsingCliques();
	void SetEdgesForPreOrderTraversal();
	void SetEdgesForPostOrderTraversal();
	void SetEdgesForTreeTraversalOperations();
	void SetLeaves();
	void SetVerticesForPreOrderTraversalWithoutLeaves();	
	void ComputeMLEOfRootProbability();
	void ComputeMLEOfTransitionMatrices();	
	void ComputeMarginalProbabilitiesUsingExpectedCounts();
	void ComputePosteriorProbabilitiesUsingMAPEstimates();	
	void SetIdsOfExternalVertices();
	void ResetAncestralSequences();
	void WriteParametersOfBH(string BHparametersFileName);
	void RemoveEdgeLength(SEM_vertex * u, SEM_vertex * v);
	void AddEdgeLength(SEM_vertex * u, SEM_vertex * v, double t);
	double GetEdgeLength(SEM_vertex * u, SEM_vertex * v);
	double ComputeEdgeLength(SEM_vertex * u, SEM_vertex * v);
	float GetLengthOfSubtendingBranch(SEM_vertex * v);
	void SetEdgeLength(SEM_vertex * u, SEM_vertex * v, double t);
	void SetEdgesFromTopologyFile(string edgeListFileName);
	void SetF81Model(string baseCompositionFileName);
	void SetF81Matrix(SEM_vertex * v);
	void SetF81_mu();
	void SetRootProbabilityAsSampleBaseComposition(string baseCompositionFileName);
	string EncodeAsDNA(vector<int> sequence);	
	void ReadRootedTree(string treeFileName);
	void SetBHparameters();
	void ReparameterizeBH();
	void ReadProbabilities();
	void WriteProbabilities(string fileName);	
	int GetVertexId(string v_name);	
	SEM_vertex * GetVertex(string v_name);
	bool ContainsVertex(string v_name);	
	emtr::Md GetP_yGivenx(emtr::Md P_xy);
	emtr::Md ComputeTransitionMatrixUsingAncestralStates(SEM_vertex * p, SEM_vertex * c);
	emtr::Md ComputeTransitionMatrixUsingAncestralStatesForTrifle(SEM_vertex * p, SEM_vertex * c, int layer);
	array <double, 4> GetBaseComposition(SEM_vertex * v);
	array <double, 4> GetBaseCompositionForTrifle(SEM_vertex * v, int layer);
	array <double, 4> GetObservedCountsForVariable(SEM_vertex * v);
	void RootTreeAtVertex(SEM_vertex * r);	
	void StoreDirectedEdgeList();
	void RestoreDirectedEdgeList();
	void StoreBestProbability();
	void StoreRootAndRootProbability();
	void RestoreRootAndRootProbability();
	void StoreTransitionMatrices();	
	void RestoreTransitionMatrices();	
	void RestoreBestProbability();
	void StoreRateMatricesAndScalingFactors();
	void RestoreRateMatricesAndScalingFactors();
	void ResetPointerToRoot();
	void ResetTimesVisited();
	void SetIdsForObservedVertices(vector <int> idsOfObservedVerticesToAdd);
	void SetNumberOfInputSequences(int numOfInputSeqsToSet);	
	void ComputeMLRootedTreeForRootSearchUnderBH();	
	void ComputeMLEstimateOfBHGivenExpectedDataCompletion();		
	void SetMinLengthOfEdges();		
	void ComputeInitialEstimateOfModelParameters();
	void ComputeInitialEstimateOfModelParametersForTrifle(int layer);
	void BumpZeroEntriesOfModelParameters();
	void SetInitialEstimateOfModelParametersUsingDirichlet();
	void SetModelParametersUsingHSS();	
	void EvaluateBHModelWithRootAtCheck(string root_check_name);
	void SwapRoot();
	void SuppressRoot();	
	bool root_search;
	string init_criterion;
	string parameter_file;	
	void StorePatterns(string pattern_file_name);
	void StorePatternIndex(string taxon_order_file_name);
	void ComputeLogLikelihood();
	void ComputeTrifleLogLikelihood(int layer);
	int siteIdx;	
	void ComputeLogLikelihoodUsingExpectedDataCompletion();	
	double GetExpectedMutualInformation(SEM_vertex * u, SEM_vertex * v);
	void ResetLogScalingFactors();
	// Mutual information I(X;Y) is computed using 
	// P(X,Y), P(X), and P(Y), which in turn are computed using
	// MAP estimates
	void InitializeExpectedCounts();
	void InitializeExpectedCountsForEachVariable();
	void InitializeExpectedCountsForEachVariablePair();
	void InitializeExpectedCountsForEachEdge();
	void ResetExpectedCounts();
	void ConstructCliqueTree();	
	void ComputeExpectedCounts();
	void ComputeExpectedCountsForTrifle(int layer);
	emtr::Md GetObservedCounts(SEM_vertex * u, SEM_vertex * v);
	void AddToExpectedCounts();
	void AddToExpectedCountsForEachVariable();
	void AddToExpectedCountsForEachVariablePair();
	emtr::Md GetExpectedCountsForVariablePair(SEM_vertex * u, SEM_vertex * v);
	emtr::Md GetPosteriorProbabilityForVariablePair(SEM_vertex * u, SEM_vertex * v);
	void AddToExpectedCountsForEachEdge();	
	int ConvertDNAtoIndex(char dna);	
	char GetDNAfromIndex(int dna_index);
	void StoreEdgeListAndSeqToAdd();		
	void SetWeightedEdgesToAddToGlobalPhylogeneticTree();
	void ComputeVertexLogLikelihood(SEM_vertex * v);
	void ComputeEdgeLogLikelihood(SEM_vertex * u, SEM_vertex * v);			
	void SetDNASequencesFromFile(string sequenceFileName);	
	void CompressDNASequences();	
	void CompressAASequences();
	void SetEdgeAndVertexLogLikelihoods();	
	void SetPrefixForOutputFiles(string prefix_for_output_files_to_set);
	void WriteRootedTreeInNewickFormat(string newickFileName);	
	void WriteCliqueTreeToFile(string cliqueTreeFileName);
	void WriteRootedTreeAsEdgeList(string fileName);
	void WriteUnrootedTreeAsEdgeList(string fileName);
	void ResetData();			
	void EM_DNA_rooted_at_each_internal_vertex_started_with_dirichlet_store_results(int num_repetitions);
	void EM_DNA_rooted_at_each_internal_vertex_started_with_parsimony_store_results(int num_repetitions);
	void EM_DNA_rooted_at_each_internal_vertex_started_with_HSS_store_results(int num_repetitions);
	void EMTrifle_DNA_for_replicate(int replicate);
	void embh_aitken(int max_iterations);
	void EMTrifle_DNA_rooted_at_each_internal_vertex_started_with_parsimony_store_results();
	void EMtrifle_DNA_rooted_at_each_internal_vertex_started_with_dirichlet_store_results();
	void EMtrifle_DNA_rooted_at_each_internal_vertex_started_with_HSS_store_results();	
	void EM_rooted_at_each_internal_vertex_started_with_dirichlet(int num_repetitions);
	void EM_rooted_at_each_internal_vertex_started_with_parsimony(int num_repetitions);
	void EM_rooted_at_each_internal_vertex_started_with_HSS_par(int num_repetitions);
	void set_alpha_PI(double a1, double a2, double a3, double a4);
	void set_alpha_M_row(double a1, double a2, double a3, double a4);
	map <pair<SEM_vertex*, SEM_vertex*>,double> ML_distances;
	array <double, 4> sample_pi();
    array <double, 4> sample_M_row();
	vector <EM_struct> EM_DNA_runs_pars;
	vector <EMTrifle_struct> EMTrifle_DNA_runs_pars;
	vector <EM_struct> EM_DNA_runs_diri;
	vector <EMTrifle_struct> EMTrifle_DNA_runs_diri;
	vector <EM_struct> EM_DNA_runs_hss;
	vector <EMTrifle_struct> EMTrifle_DNA_runs_hss;	
	EM_struct EM_current{};
	EMTrifle_struct EMTrifle_current{};	
	// string emtrifle_to_json(const EMTrifle_struct& em) const;	
	int first_index_gt(const std::vector<double>& cum, double thr);
	// Select vertex for rooting Chow-Liu tree and update edges in T
	// Modify T such that T is a bifurcating tree and likelihood of updated
	// tree is equivalent to the likelihood of T
	SEM (double loglikelihood_conv_thresh, int max_EM_iter, bool verbose_flag_to_set) {
		// this->SetDayhoffRateMatrix();		
		this->alpha_pi = {100,100,100,100}; // default value
		this->alpha_M_row = {100,2,2,2};
		this->root_search = false;
		this->ecdllConvergenceThreshold = loglikelihood_conv_thresh;
		this->maxIter = max_EM_iter;		
		this->verbose = verbose_flag_to_set;		
		this->node_ind = 0;
		this->vertexMap = new map <int, SEM_vertex *> ;
		this->M_hss = new map <pair<SEM_vertex*,SEM_vertex*>,emtr::Md>;		
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		this->generator = default_random_engine(seed);
		this->I4by4 = emtr::Md{};
		for (int i = 0; i < 4; i++) {
			this->I4by4[i][i] = 1.0;
		}
		this->cliqueT = new cliqueTree;		
		mapDNAtoInteger["A"] = 0;
		mapDNAtoInteger["C"] = 1;
		mapDNAtoInteger["G"] = 2;
		mapDNAtoInteger["T"] = 3;
		this->finalIterationOfSEM = 0;
	}
	
	~SEM () {
		if (packed_patterns != nullptr) {
			delete packed_patterns;
			packed_patterns = nullptr;
    	}
		for (pair <int, SEM_vertex * > idPtrPair : * this->vertexMap) {
			delete idPtrPair.second;
		}
		this->vertexMap->clear();
		delete this->vertexMap;		
		delete this->cliqueT;
		delete this->M_hss;
	}
};

int SEM::compressedToFirstOriginalSite(size_t k) const {
    if (k >= DNAPatternOriginalSites.size()) return -1;
    const auto& v = DNAPatternOriginalSites[k];
    return v.empty() ? -1 : v.front();
}

const vector<int>& SEM::compressedToAllOriginalSites(size_t k) const {
    static const vector<int> EMPTY;
    if (k >= DNAPatternOriginalSites.size()) return EMPTY;
    return DNAPatternOriginalSites[k];
}

int SEM::originalToCompressed(size_t origSite) const {
    if (origSite >= OriginalToCompressed.size()) return -1;
    return OriginalToCompressed[origSite];
}

double SEM::gap_proportion_in_pattern(const std::vector<int>& pat) {
    if (pat.empty()) return 0.0;
    int gaps = 0;
    for (int ch : pat) if (ch < 0) ++gaps;
    return static_cast<double>(gaps) / static_cast<double>(pat.size());
}

int SEM::unique_non_gap_count_in_pattern(const std::vector<int>& pat) {    
    unordered_set<int> uniq;
    uniq.reserve(pat.size());
    for (int ch : pat) if (ch >= 0) uniq.insert(ch);
    return static_cast<int>(uniq.size());
}

void SEM::addToZeroEntriesOfRootProbability(array <double,4> & baseComposition) {
	double sum = 0.0;

    for (size_t i = 0; i < 4; ++i) {
        if (baseComposition[i] == 0.0) {
            baseComposition[i] = 0.1;
        }
        sum += baseComposition[i];
    }

	for (size_t i = 0; i < 4; ++i) {
		baseComposition[i] /= sum;
	}
    
}

void SEM::addToZeroEntriesOfTransitionMatrix(array<array<double,4>,4>& P) {
	
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (P[i][j] == 0.0) {
				P[i][j] = (i == j) ? 0.97 : 0.01;
			}
		}
	}
	
	for (int i = 0; i < 4; ++i) {
		double rowSum = 0.0;
		for (int j = 0; j < 4; ++j) rowSum += P[i][j];

		if (rowSum == 0.0 || !isfinite(rowSum)) {		
			for (int j = 0; j < 4; ++j) P[i][j] = 0.01;
			P[i][i] = 0.97;
		} else {
			for (int j = 0; j < 4; ++j) P[i][j] /= rowSum;
		}
	}
}

SEM_vertex * SEM::GetVertex(string v_name){
	bool contains_v = this->ContainsVertex(v_name);
	SEM_vertex * node;
	int node_id;
	if(!contains_v) {
        throw mt_error("v_name not found");
    }
	
	node_id = this->GetVertexId(v_name);		
	node = (*this->vertexMap)[node_id];
	return node;

}

int SEM::GetVertexId(string v_name) {
	SEM_vertex * v;
	int idToReturn = -10;
	for (pair<int,SEM_vertex*> idPtrPair : *this->vertexMap) {
		v = idPtrPair.second;
		if (v->name == v_name){
			idToReturn = v->id;						
		}
	}
	if (idToReturn == -10){
		cout << "Unable to find id for:" << v_name << endl;
	}
	return (idToReturn);
}

int SEM::ConvertDNAtoIndex(char dna){
	int value;
	switch (dna)
	{
	case 'A':
		value = 0;
		break;
	case 'C':
		value = 1;
		break;
	case 'G':
		value = 2;
		break;
	case 'T':
		value = 3;
		break;
	default:
		value = 4;
		break;
	}	
	return (value);
}

void SEM::SetVertexVector(){
	this->vertices.clear();
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		this->vertices.push_back(v);
	}
}

void SEM::SetVertexVectorExceptRoot(){
	this->non_root_vertices.clear();	
	for (SEM_vertex * v : this->vertices) {		
		if (v != this->root){
			this->non_root_vertices.push_back(v);
		}		
	}
}

void SEM::ReparameterizeBH() {
	
	// Compute pi, P(u,v) and P(v,u) for each vertex and edge using the method described in hss paper	
	
	// 1. Set pi for root	
	for (int p_id = 0; p_id < 4; ++p_id) {
		if(this->rootProbability[p_id] == 0) {cout << "invalid probability"; throw mt_error("invalid probability");}
		this->root->root_prob_hss[p_id] = this->rootProbability[p_id];
	}
	// Store transition matrices in transition_prob_hss map, store root prob in node as root_prob_hss
	SEM_vertex * p; SEM_vertex * c; array <double,4> pi_p; array <double,4> pi_c;
	M_hss->clear();
	for (pair<SEM_vertex*, SEM_vertex*> edge : this->edgesForPreOrderTreeTraversal) {
		p = edge.first;
		c = edge.second;
		
		emtr::Md M_pc = c->transitionMatrix; // transition matrix of orig BH parameters 
		emtr::Md M_cp; 					     // transition matrix of reparameterized BH
						
		(*this->M_hss)[{p,c}] = M_pc;

		// 2. Initialize pi_p and pi_c
		for (int x = 0; x < 4; x ++) pi_p[x] = p->root_prob_hss[x]; // root prob already computed
		for (int x = 0; x < 4; x ++) pi_c[x] = 0;					// root prob to be computed
		
		// 3. Compute pi_c
		for (int x = 0; x < 4 ; x ++){
			for (int y = 0; y < 4; y ++){
				pi_c[x] += pi_p[y] * M_pc[y][x];
			}
		}

		// 4. Store pi_c
		for (int x = 0; x < 4; x++) {
			c->root_prob_hss[x] = pi_c[x];
		}

		// 5. Compute M_cp for root at child		
		for (int x = 0; x < 4; x++) {
			for (int y = 0; y < 4; y++) {
				M_cp[y][x] = M_pc[x][y] * pi_p[x]/pi_c[y];			// Bayes rule as described in HSS paper
			}
		}
		
		// 6. Store M_cp
		(*this->M_hss)[{c,p}] = M_cp;
	}	
}

float SEM::GetLengthOfSubtendingBranch(SEM_vertex * v) {
	SEM_vertex * p = v->parent;
	float t;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (p->id < v->id) {
		vertexPair = make_pair(p,v);
	} else {
		vertexPair = make_pair(v,p);
	}
	t = this->edgeLengths[vertexPair];
	return (t);
}

void SEM::AddToExpectedCountsForEachVariable() {
	SEM_vertex * v;
	double siteWeight = this->DNAPatternWeights[this->cliqueT->site];	
	// Add to counts for each unobserved vertex (C->x) where C is a clique
	array <double, 4> marginalizedProbability;
	vector <SEM_vertex *> vertexList;
	for (clique * C: this->cliqueT->cliques) {
		v = C->x;
		if(v->observed){
            throw mt_error("v should not be am observed vertex");
        }
		if (find(vertexList.begin(),vertexList.end(),v) == vertexList.end()) {
			vertexList.push_back(v);
			marginalizedProbability = C->MarginalizeOverVariable(C->y);
			for (int i = 0; i < 4; i++) {
				this->expectedCountsForVertex[v][i] += marginalizedProbability[i] * siteWeight;
			}
		}
	}
	vertexList.clear();
}

void SEM::AddToExpectedCountsForEachEdge() {
	double siteWeight = this->DNAPatternWeights[this->cliqueT->site];
	emtr::Md countMatrixPerSite;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	SEM_vertex * u; SEM_vertex * v;
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
		tie (u,v) = edge;
		if (u->id < v->id) {
			vertexPair.first = u; vertexPair.second = v;
		} else {
			vertexPair.second = u; vertexPair.first = v;
		}
		countMatrixPerSite = this->cliqueT->marginalizedProbabilitiesForVariablePair[vertexPair];
		for (int dna_u = 0; dna_u < 4; dna_u ++) {
			for (int dna_v = 0; dna_v < 4; dna_v ++) {
				this->expectedCountsForVertexPair[vertexPair][dna_u][dna_v] += countMatrixPerSite[dna_u][dna_v] * siteWeight;
			}
		}
	}
}

void SEM::ComputeExpectedCounts() {
    // cout << "Initializing expected counts" << endl;
	// cout << "11a" << endl;
	this->InitializeExpectedCountsForEachVariable();
	// cout << "11b" << endl;
	this->InitializeExpectedCountsForEachEdge();
	// cout << "11c" << endl;
    //	this->ResetExpectedCounts();
    //	SEM_vertex * x; SEM_vertex * y;
//	int dna_x; int dna_y;
	bool debug = 0;
	if (debug) {
		cout << "Debug computing expected counts" << endl;
	}

	// OPTIMIZATION: Initialize base potentials ONCE before the loop
	this->cliqueT->InitializeBasePotentials();

// Iterate over sites
	// parallelize here if needed
	for (int site = 0; site < this->num_dna_patterns; site++) {
		// cout << "12a" << endl;
		// cout << "computing expected counts for site " << site << endl;
		// OPTIMIZED: Apply evidence and reset instead of full initialization
		this->cliqueT->ApplyEvidenceAndReset(site);
		// cout << "12c" << endl;
		this->cliqueT->CalibrateTree(); // site log likelihood can comptuted after this step
		// cout << "12d" << endl;
		this->cliqueT->SetMarginalProbabilitesForEachEdgeAsBelief();
		// cout << "12e" << endl;
		this->AddToExpectedCountsForEachVariable();
		// cout << "12f" << endl;
		this->AddToExpectedCountsForEachEdge();
		// cout << "12g" << endl;
	}
	// cout << "11d" << endl;
}

void SEM::ComputeMarginalProbabilitiesUsingExpectedCounts() {	
	SEM_vertex * v;
	double sum;
	// Compute posterior probability for vertex
	this->posteriorProbabilityForVertex.clear();
	array <double, 4> P_X;	
	for (pair <SEM_vertex *, array <double, 4>> vertexAndCountArray: this->expectedCountsForVertex) {
		v = vertexAndCountArray.first;
		P_X = vertexAndCountArray.second;
		sum = 0;
		for (int i = 0; i < 4; i++) {
			sum += P_X[i];
		}
		if (sum > 0) {
			for (int i = 0; i < 4; i++) {
				P_X[i] /= sum;
			}
		}
		for (int i = 0; i < 4; i++) { 
			if(isnan(P_X[i])) {
				cout << "P_X[i] is nan" << endl;
				throw mt_error("P_X[i] is nan");
			}
		}
		this->posteriorProbabilityForVertex.insert(pair<SEM_vertex * , array <double, 4>>(v,P_X));
	}
	// Compute posterior probability for vertex pair
	this->posteriorProbabilityForVertexPair.clear();
	emtr::Md P_XY;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	for (pair <pair<SEM_vertex *, SEM_vertex *>, emtr::Md> vertexPairAndCountMatrix: this->expectedCountsForVertexPair) {
		vertexPair = vertexPairAndCountMatrix.first;
		P_XY = vertexPairAndCountMatrix.second;
		sum = 0;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				sum += P_XY[i][j];
			}
		}
		if (sum > 0) {
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					P_XY[i][j] /= sum;
				}
			}
		}

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				if(isnan(P_XY[i][j])) {
					cout << "P_XY" << i << j << "is nan" << endl;
					throw mt_error("P_XY[i][j] is nan");
				}
			}
		}
		this->posteriorProbabilityForVertexPair.insert(pair<pair<SEM_vertex *, SEM_vertex *>,emtr::Md>(vertexPair,P_XY));
	}
}

void SEM::ConstructCliqueTree() {
	if (this->cliqueT == nullptr) {
		this->cliqueT = new cliqueTree;
	}
	this->cliqueT->rootSet = 0;
	SEM_vertex * root;
	for (clique * C : this->cliqueT->cliques) {
		delete C;
	}
	this->cliqueT->cliques.clear();
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
		clique * C = new clique(edge.first, edge.second);		
		this->cliqueT->AddClique(C);
		if (C->x->parent == C->x and !this->cliqueT->rootSet) {
			this->cliqueT->root = C;
			this->cliqueT->rootSet = 1;
			root = C->x;
		}
	}
	clique * C_i; clique * C_j;
	// Iterate over clique pairs and identify cliques
	// that have one vertex in common
	for (unsigned int i = 0; i < this->cliqueT->cliques.size(); i ++) {
		C_i = this->cliqueT->cliques[i];
		// Set Ci as the root clique if Ci.x is the root vertex
		for (unsigned int j = i+1; j < this->cliqueT->cliques.size(); j ++) {
			C_j = this->cliqueT->cliques[j];
			// Add edge Ci -> Cj if Ci.y = Cj.x;
			if (C_i->y == C_j->x) {
//				cout << "Case 1" << endl;
//				cout << "C_i.x, C_i.y is " << C_i->x->id << ", " << C_i->y->id << endl;
//				cout << "C_j.x, C_j.y is " << C_j->x->id << ", " << C_j->y->id << endl;
				this->cliqueT->AddEdge(C_i, C_j);
				// Add edge Cj -> Ci if Cj.y = Ci.x;
			} else if (C_j->y == C_i->x) {
//				cout << "Case 2" << endl;
//				cout << "C_i.x, C_i.y is " << C_i->x->id << ", " << C_i->y->id << endl;
//				cout << "C_j.x, C_j.y is " << C_j->x->id << ", " << C_j->y->id << endl;
				this->cliqueT->AddEdge(C_j, C_i);
				// If Ci->x = Cj->x 
				// add edge Ci -> Cj
			} else if (C_i->x == C_j->x and C_i->parent == C_i) {
//				cout << "Case 3" << endl;
//				cout << "C_i.x, C_i.y is " << C_i->x->id << ", " << C_i->y->id << endl;
				this->cliqueT->AddEdge(C_i, C_j);
				// Check to see that Ci is the root clique				
				if (this->cliqueT->root != C_i) {
					cout << "Check root of clique tree" << endl;
                    throw mt_error("Check root of clique tree");
				}
			}
			// Note that Cj can never be the root clique
			// because Ci is visited before Cj
		}
	}	
	this->cliqueT->SetLeaves();
	this->cliqueT->SetEdgesForTreeTraversalOperations();
	for (clique * C: this->cliqueT->cliques) {
		C->root_variable = root;
	}
}

void SEM::ResetExpectedCounts() {
	SEM_vertex* u; SEM_vertex* v; 
	// Reset counts for each unobserved vertex
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		if (!v->observed) {
			for (int i = 0; i < 4; i++) {
				this->expectedCountsForVertex[v][i] = 0;
			}
		}
	}
	// Reset counts for each vertex pair such that at least one vertex is not observed
	for (pair <int, SEM_vertex *> idPtrPair_1 : * this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex *> idPtrPair_2 : * this->vertexMap) {
			v = idPtrPair_2.second;
			if (!u->observed or !v->observed) {
				if (u->id < v->id) {
					this->expectedCountsForVertexPair[pair <SEM_vertex *, SEM_vertex *>(u,v)] = emtr::Md{};
				}	
			}			
		}
	}
}

void SEM::InitializeExpectedCountsForEachVariable() {
	SEM_vertex * v;
	// Initialize expected counts for each vertex
	this->expectedCountsForVertex.clear();
	array <double, 4> observedCounts;
	for (pair<int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		for (int i = 0; i < 4; i++) {
			observedCounts[i] = 0;
		}
		if (v->observed) {			
			observedCounts = this->GetObservedCountsForVariable(v);
		}
		this->expectedCountsForVertex.insert(pair<SEM_vertex *, array<double,4>>(v,observedCounts));
	}	
}

void SEM::InitializeExpectedCountsForEachVariablePair() {
	SEM_vertex * u; SEM_vertex * v;
	// Initialize expected counts for each vertex pair
	this->expectedCountsForVertexPair.clear();	
	emtr::Md countMatrix;	
	int dna_u;
	int dna_v;
	for (pair<int,SEM_vertex *> idPtrPair_1 : * this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex *> idPtrPair_2 : * this->vertexMap) {
			v = idPtrPair_2.second;
			if (u->id < v->id) {
				countMatrix = emtr::Md{};			
				if (u->observed and v->observed) {
					for (int site = 0; site < this->num_dna_patterns; site++) {
						dna_u = u->DNArecoded[site];
						dna_v = v->DNArecoded[site];
						if (dna_u < 4 && dna_v < 4) { // FIX_AMB
							countMatrix[dna_u][dna_v] += this->DNAPatternWeights[site];
						}						
					}
				}
				this->expectedCountsForVertexPair.insert(make_pair(pair <SEM_vertex *, SEM_vertex *>(u,v), countMatrix));
			}
		}
	}
}

void SEM::InitializeExpectedCountsForEachEdge() {
	// Initialize expected counts for each vertex pair
	SEM_vertex * u; SEM_vertex * v;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	this->expectedCountsForVertexPair.clear();
	emtr::Md countMatrix;
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
		countMatrix = emtr::Md{};
		tie (u,v) = edge;
		if (u->id < v->id) {
			vertexPair.first = u; vertexPair.second = v;
		} else {
			vertexPair.first = v; vertexPair.second = u;
		}
		this->expectedCountsForVertexPair.insert(make_pair(vertexPair, countMatrix));
	}
}

void SEM::ReadPatternsFromFile(string pattern_file_name, string taxon_order_file_name) {
    // First, read taxon order to establish mapping
    this->StorePatternIndex(taxon_order_file_name);
    
    ifstream inFile(pattern_file_name);
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open pattern file " << pattern_file_name << endl;
        return;
    }
    
    // Count patterns and determine number of taxa
    vector<vector<int>> temp_patterns;
    vector<int> temp_weights;
    string line;
    
    while (getline(inFile, line)) {
        if (line.empty()) continue;
        
        istringstream iss(line);
        int weight;
        iss >> weight;
        
        vector<int> pattern;
        int base;
        while (iss >> base) {
            pattern.push_back(base);
        }
        
        if (!pattern.empty()) {
            temp_patterns.push_back(pattern);
            temp_weights.push_back(weight);
        }
    }
    inFile.close();
    
    if (temp_patterns.empty()) {
        cerr << "Error: No patterns found in file" << endl;
        return;
    }
    
    num_patterns_from_file = temp_patterns.size();
    int num_taxa = temp_patterns[0].size();
    
    // Create packed storage
    if (packed_patterns != nullptr) {
        delete packed_patterns;
    }
    packed_patterns = new PackedPatternStorage(num_patterns_from_file, num_taxa);
    
    // Store patterns in packed format
    for (int i = 0; i < num_patterns_from_file; i++) {
        vector<uint8_t> pattern_uint8(temp_patterns[i].begin(), temp_patterns[i].end());
        packed_patterns->store_pattern(i, pattern_uint8);
    }
    
    // Store weights
    pattern_weights = temp_weights;
    
    // Update num_dna_patterns to match pattern file
    this->num_dna_patterns = num_patterns_from_file;
    
    // Copy weights to DNAPatternWeights for compatibility
    this->DNAPatternWeights = temp_weights;
    
    cout << "Loaded " << num_patterns_from_file << " patterns with " 
         << num_taxa << " taxa" << endl;
    cout << "Memory usage: " << packed_patterns->get_memory_bytes() 
         << " bytes (3-bit packed)" << endl;
    
    // Calculate memory savings
    size_t unpacked_size = num_patterns_from_file * num_taxa * sizeof(int);
    size_t packed_size = packed_patterns->get_memory_bytes();
    double savings_pct = 100.0 * (1.0 - (double)packed_size / unpacked_size);
    cout << "Memory savings: " << savings_pct << "% compared to int storage" << endl;
}


void SEM::StorePatternIndex(string taxon_order_file_name){
	ifstream inFile(taxon_order_file_name);
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open file " << taxon_order_file_name << endl;
        return;
    }
    
    string line;
    getline(inFile, line); // Skip header
    
    while (getline(inFile, line)) {
        if (line.empty()) continue;
        
        istringstream iss(line);
        string taxon_name;
        int position;
        
        if (getline(iss, taxon_name, ',') && (iss >> position)) {
            SEM_vertex* v = this->GetVertex(taxon_name);
            if (v != nullptr) {
                v->pattern_index = position;
            }
        }
    }    
    inFile.close();
}

void SEM::ComputeLogLikelihood() {
	this->logLikelihood = 0;
	map <SEM_vertex*,array<double,4>> conditionalLikelihoodMap;
	std::array <double,4> conditionalLikelihood;
	vector <SEM_vertex *> verticesToVisit;
	SEM_vertex * p;
	SEM_vertex * c;
	emtr::Md P;
	for (int site = 0; site < this->num_dna_patterns; site++) { // parallelize computation here 
		// if (site == 4) break;
		conditionalLikelihoodMap.clear(); 
		this->ResetLogScalingFactors();
		for (auto& edge : this->edgesForPostOrderTreeTraversal) {
			tie(p, c) = edge;
			P = c->transitionMatrix;

			p->logScalingFactors += c->logScalingFactors;

			// Initialize leaf child
			if (c->outDegree == 0) {				
				if (c->DNAcompressed[site] != 4) { // != 4
					conditionalLikelihood.fill(0.0);
					conditionalLikelihood[c->DNAcompressed[site]] = 1.0;
				} else {
					conditionalLikelihood.fill(1.0);
				}
				conditionalLikelihoodMap.insert({c, conditionalLikelihood});
			}

			// Initialize parent p if absent
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()) {
				if (p->id > this->numberOfObservedVertices - 1) {
					// cout << p->name << " is latent" << endl;
					conditionalLikelihood.fill(1.0);      // latent
				} else {
					int base = p->DNAcompressed[site];
					if (base < 4) {
						// Valid DNA base (0-3)
						conditionalLikelihood.fill(0.0);
						conditionalLikelihood[base] = 1.0;
					} else {
						// Gap (4) - treat as missing data
						conditionalLikelihood.fill(1.0);
					}
				}
				conditionalLikelihoodMap.insert({p, conditionalLikelihood});
			}

			// DP update using safer access for child
			double largestConditionalLikelihood = 0.0;
			const auto& childCL = conditionalLikelihoodMap.at(c);

			for (int dna_p = 0; dna_p < 4; ++dna_p) {
				double partialLikelihood = 0.0;
				for (int dna_c = 0; dna_c < 4; ++dna_c) {
					partialLikelihood += P[dna_p][dna_c] * childCL[dna_c];
				}
				conditionalLikelihoodMap[p][dna_p] *= partialLikelihood;
				largestConditionalLikelihood = max(largestConditionalLikelihood, conditionalLikelihoodMap[p][dna_p]);
			}

			if (largestConditionalLikelihood > 0.0) {
				for (int dna_p = 0; dna_p < 4; ++dna_p) {
					conditionalLikelihoodMap[p][dna_p] /= largestConditionalLikelihood;
				}
				p->logScalingFactors += log(largestConditionalLikelihood);
			} else {
				cout << " p name is " << p->name << endl;
				cout << " c name is " << c->name << endl;
				cout << "conditional likelihood is zero " << largestConditionalLikelihood << endl;
				for (int i = 0; i < 4; i ++) {
					for (int j = 0; j < 4; j ++) {
						cout << "c->transitionMatrix[" << i << "," << j << "] = " << c->transitionMatrix[i][j] << endl;
					}
				}	
				cout << "site is " << site << endl;
				cout << "conditionalLikelihoodMap[p]: ";
				for (int i = 0; i < 4; i ++) {
					cout << conditionalLikelihoodMap[p][i] << " ";
				}
				cout << endl;
				for (int i = 0; i < 4; i ++) {
					cout << "childCL[" << i << "] = " << childCL[i] << endl;
				}
				throw mt_error("Largest conditional likelihood value is zero");
			}
		}
		double siteLikelihood = 0.0;
		const auto& rootCL = conditionalLikelihoodMap.at(this->root);
		for (int dna = 0; dna < 4; ++dna) {
			siteLikelihood += this->rootProbability[dna] * rootCL[dna];
		}
		if (siteLikelihood <= 0.0) throw mt_error("siteLikelihood <= 0");

		this->logLikelihood += (this->root->logScalingFactors + log(siteLikelihood)) * this->DNAPatternWeights[site];
	}
}

void SEM::ComputeLogLikelihoodUsingPatterns() {
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Call ReadPatternsFromFile() first." << endl;
        return;
    }
    
    this->logLikelihood = 0;
    map<SEM_vertex*, array<double,4>> conditionalLikelihoodMap;
    array<double,4> conditionalLikelihood;
    double partialLikelihood;
    double siteLikelihood;	
    double largestConditionalLikelihood = 0;
    
    SEM_vertex* p;
    SEM_vertex* c;
    emtr::Md P;
    
    // Get pattern-to-taxon mapping
    map<int, int> pattern_index_to_vertex_index;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed && v->pattern_index >= 0) {
            pattern_index_to_vertex_index[v->pattern_index] = v->id;
        }
    }
    
    int num_taxa = packed_patterns->get_num_taxa();
    
    // Iterate over each pattern (site)
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        conditionalLikelihoodMap.clear(); 
        this->ResetLogScalingFactors();
        
        // Get the pattern for this site
        vector<uint8_t> pattern = packed_patterns->get_pattern(pattern_idx);
        
        // Build a map from vertex ID to base
        map<int, uint8_t> vertex_to_base;
        for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
            auto it = pattern_index_to_vertex_index.find(taxon_idx);
            if (it != pattern_index_to_vertex_index.end()) {
                int vertex_id = it->second;
                vertex_to_base[vertex_id] = pattern[taxon_idx];
            }
        }
        
        // Traverse tree using post-order edges
        for (auto& edge : this->edgesForPostOrderTreeTraversal) {
            tie(p, c) = edge;
            P = c->transitionMatrix;
            
            p->logScalingFactors += c->logScalingFactors;
            
            // Initialize leaf child (ONLY for outDegree == 0)
            if (c->outDegree == 0) {	
                uint8_t base = 4; // default to gap
                auto it = vertex_to_base.find(c->id);
                if (it != vertex_to_base.end()) {
                    base = it->second;
                }
                
                // FIXED: Match original behavior exactly
                if (base != 4) { // not gap
                    conditionalLikelihood.fill(0.0);
                    conditionalLikelihood[base] = 1.0;
                } else {
                    conditionalLikelihood.fill(1.0);
                }
                conditionalLikelihoodMap.insert({c, conditionalLikelihood});
            }
            
            // Initialize parent p if absent
            if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()) {
                if (p->id > this->numberOfObservedVertices - 1) {
                    // Latent (hidden) vertex - treat as missing data
                    conditionalLikelihood.fill(1.0);
                } else {
                    // Observed vertex as parent
                    // NOTE: In a properly rooted tree, observed vertices (leaves) should have outDegree == 0
                    // However, if an observed vertex appears as a parent, we need to handle it
                    uint8_t base = 4;  // default to gap
                    auto it = vertex_to_base.find(p->id);
                    if (it != vertex_to_base.end()) {
                        base = it->second;
                    }
                    
                    // CRITICAL FIX: The original code does NOT check for gaps when initializing parents
                    // It directly does: conditionalLikelihood[p->DNAcompressed[site]] = 1.0
                    // This assumes DNAcompressed never has 4 for observed parents
                    // But our pattern file HAS gaps (value 4), so we must check
                    if (base < 4) {
                        // Valid DNA base (0-3)
                        conditionalLikelihood.fill(0.0);
                        conditionalLikelihood[base] = 1.0;
                    } else {
                        // Gap (4) - treat as missing data (all states equally likely)
                        conditionalLikelihood.fill(1.0);
                    }
                }
                conditionalLikelihoodMap.insert({p, conditionalLikelihood});
            }
            
            // DP update
            largestConditionalLikelihood = 0.0;
            const auto& childCL = conditionalLikelihoodMap.at(c);
            
            for (int dna_p = 0; dna_p < 4; ++dna_p) {
                partialLikelihood = 0.0;
                for (int dna_c = 0; dna_c < 4; ++dna_c) {
                    partialLikelihood += P[dna_p][dna_c] * childCL[dna_c];
                }
                conditionalLikelihoodMap[p][dna_p] *= partialLikelihood;
                largestConditionalLikelihood = max(largestConditionalLikelihood, 
                                                  conditionalLikelihoodMap[p][dna_p]);
            }
            
            if (largestConditionalLikelihood > 0.0) {
                for (int dna_p = 0; dna_p < 4; ++dna_p) {
                    conditionalLikelihoodMap[p][dna_p] /= largestConditionalLikelihood;
                }
                p->logScalingFactors += log(largestConditionalLikelihood);
            } else {
                // Debug output (same as original)
                cout << "conditional likelihood is zero " << largestConditionalLikelihood << endl;
                cout << "pattern index: " << pattern_idx << endl;
                cout << "p->id: " << p->id << " c->id: " << c->id << endl;
                
                // Print the bases
                auto it_p = vertex_to_base.find(p->id);
                auto it_c = vertex_to_base.find(c->id);
                if (it_p != vertex_to_base.end()) {
                    cout << "p base: " << (int)it_p->second << endl;
                } else {
                    cout << "p base: not found (internal node)" << endl;
                }
                if (it_c != vertex_to_base.end()) {
                    cout << "c base: " << (int)it_c->second << endl;
                } else {
                    cout << "c base: not found (internal node)" << endl;
                }
                
                // Print transition matrix
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        cout << "c->transitionMatrix[" << i << "," << j << "] = " 
                             << c->transitionMatrix[i][j] << endl;
                    }
                }
                
                throw mt_error("Largest conditional likelihood value is zero");
            }
        }
        
        // Compute site likelihood at root
        siteLikelihood = 0.0;
        const auto& rootCL = conditionalLikelihoodMap.at(this->root);
        for (int dna = 0; dna < 4; ++dna) {
            siteLikelihood += this->rootProbability[dna] * rootCL[dna];
        }
        if (siteLikelihood <= 0.0) {
            throw mt_error("siteLikelihood <= 0");
        }
        
        // Add weighted site log-likelihood
        int weight = pattern_weights[pattern_idx];
        this->logLikelihood += (this->root->logScalingFactors + log(siteLikelihood)) * weight;
    }
}

void SEM::ComputeLogLikelihoodUsingPatternsWithPropagation() {
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Call ReadPatternsFromFile() first." << endl;
        return;
    }

    if (this->cliqueT == nullptr || this->cliqueT->root == nullptr) {
        this->ConstructCliqueTree();
    }

    this->logLikelihood = 0;
    clique* rootClique = this->cliqueT->root;

    // Get pattern-to-taxon mapping (same as in ComputeLogLikelihoodUsingPatterns)
    map<int, int> pattern_index_to_vertex_index;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed && v->pattern_index >= 0) {
            pattern_index_to_vertex_index[v->pattern_index] = v->id;
        }
    }

    int num_taxa = packed_patterns->get_num_taxa();

    // Save original DNAcompressed data for observed vertices
    map<SEM_vertex*, vector<int>> saved_DNAcompressed;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed) {
            saved_DNAcompressed[v] = v->DNAcompressed;
            // Resize to hold pattern data (one element per pattern)
            v->DNAcompressed.resize(num_patterns_from_file, 4); // Initialize with gaps
        }
    }

    // Pre-populate all patterns into DNAcompressed
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        vector<uint8_t> pattern = packed_patterns->get_pattern(pattern_idx);

        for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
            auto it = pattern_index_to_vertex_index.find(taxon_idx);
            if (it != pattern_index_to_vertex_index.end()) {
                SEM_vertex* v = (*this->vertexMap)[it->second];
                v->DNAcompressed[pattern_idx] = pattern[taxon_idx];
            }
        }
    }

    // Now iterate over each pattern using standard propagation algorithm
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        this->cliqueT->SetSite(pattern_idx);
        this->cliqueT->InitializePotentialAndBeliefs();
        this->cliqueT->CalibrateTree();

        // Marginalize over child variable Y to get P(X | data) for root variable X
        std::array<double, 4> marginalX = rootClique->MarginalizeOverVariable(rootClique->y);

        // Weight by root probabilities to get P(data, X) = P(data | X) * P(X)
        double siteLikelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += this->rootProbability[dna] * marginalX[dna];
        }

        // Combine with accumulated log scaling factors
        double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);

        // Use pattern weight
        int weight = pattern_weights[pattern_idx];
        this->logLikelihood += siteLogLikelihood * weight;
    }

    // Restore original DNAcompressed data
    for (auto& pair : saved_DNAcompressed) {
        pair.first->DNAcompressed = pair.second;
    }
}

void SEM::ComputeLogLikelihoodUsingPatternsWithPropagationAtClique(clique* target_clique) {
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Call ReadPatternsFromFile() first." << endl;
        return;
    }

    if (this->cliqueT == nullptr || this->cliqueT->root == nullptr) {
        this->ConstructCliqueTree();
    }

    this->logLikelihood = 0;

    // Get pattern-to-taxon mapping
    map<int, int> pattern_index_to_vertex_index;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed && v->pattern_index >= 0) {
            pattern_index_to_vertex_index[v->pattern_index] = v->id;
        }
    }

    int num_taxa = packed_patterns->get_num_taxa();

    // Save original DNAcompressed data for observed vertices
    map<SEM_vertex*, vector<int>> saved_DNAcompressed;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed) {
            saved_DNAcompressed[v] = v->DNAcompressed;
            v->DNAcompressed.resize(num_patterns_from_file, 4);
        }
    }

    // Pre-populate all patterns into DNAcompressed
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        vector<uint8_t> pattern = packed_patterns->get_pattern(pattern_idx);

        for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
            auto it = pattern_index_to_vertex_index.find(taxon_idx);
            if (it != pattern_index_to_vertex_index.end()) {
                SEM_vertex* v = (*this->vertexMap)[it->second];
                v->DNAcompressed[pattern_idx] = pattern[taxon_idx];
            }
        }
    }

    // OPTIMIZATION: Initialize base potentials ONCE before the loop
    this->cliqueT->InitializeBasePotentials();

    // Now iterate over each pattern
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        // OPTIMIZED: Apply evidence and reset instead of full initialization
        this->cliqueT->ApplyEvidenceAndReset(pattern_idx);
        this->cliqueT->CalibrateTree();

        double siteLikelihood = 0.0;

        // Check if target clique contains the root variable
        if (target_clique->x == this->root || target_clique->y == this->root) {
            // This clique contains the root, similar to root clique computation
            if (target_clique->x == this->root) {
                // X is root, marginalize over Y
                std::array<double, 4> marginalX = target_clique->MarginalizeOverVariable(target_clique->y);
                for (int dna = 0; dna < 4; dna++) {
                    siteLikelihood += this->rootProbability[dna] * marginalX[dna];
                }
            } else {
                // Y is root, marginalize over X
                std::array<double, 4> marginalY = target_clique->MarginalizeOverVariable(target_clique->x);
                for (int dna = 0; dna < 4; dna++) {
                    siteLikelihood += this->rootProbability[dna] * marginalY[dna];
                }
            }
        } else {
            // For non-root cliques, we need to properly incorporate the root prior
            clique* root_clique = this->cliqueT->root;

            // Get marginal probability of root variable from the root clique
            std::array<double, 4> marginalRoot;
            if (root_clique->x == this->root) {
                marginalRoot = root_clique->MarginalizeOverVariable(root_clique->y);
            } else {
                marginalRoot = root_clique->MarginalizeOverVariable(root_clique->x);
            }

            double rootPriorContribution = 0.0;
            for (int dna = 0; dna < 4; dna++) {
                rootPriorContribution += this->rootProbability[dna] * marginalRoot[dna];
            }

            siteLikelihood = rootPriorContribution;
        }

        // Combine with accumulated log scaling factors
        double siteLogLikelihood = target_clique->logScalingFactorForClique + log(siteLikelihood);

        // Use pattern weight
        int weight = pattern_weights[pattern_idx];
        this->logLikelihood += siteLogLikelihood * weight;
    }

    // Restore original DNAcompressed data
    for (auto& pair : saved_DNAcompressed) {
        pair.first->DNAcompressed = pair.second;
    }
}

void SEM::VerifyLogLikelihoodAtAllCliques() {
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Call ReadPatternsFromFile() first." << endl;
        return;
    }

    if (this->cliqueT == nullptr || this->cliqueT->root == nullptr) {
        this->ConstructCliqueTree();
    }

    // First compute reference log-likelihood at root clique
    this->ComputeLogLikelihoodUsingPatternsWithPropagation();
    double reference_ll = this->logLikelihood;
    cout << "Reference log-likelihood (at root clique): " << setprecision(11) << reference_ll << endl;

    // Now verify at each clique in the tree
    cout << "\nVerifying log-likelihood computation at all " << this->cliqueT->cliques.size() << " cliques:" << endl;

    int num_passed = 0;
    int num_failed = 0;
    double tolerance = 1e-6;

    for (clique* C : this->cliqueT->cliques) {
        this->ComputeLogLikelihoodUsingPatternsWithPropagationAtClique(C);
        double clique_ll = this->logLikelihood;

        double abs_diff = fabs(clique_ll - reference_ll);
        double rel_diff = abs_diff / fabs(reference_ll);

        bool passed = (abs_diff < tolerance) || (rel_diff < tolerance);

        if (passed) {
            num_passed++;
            cout << "  PASSED: Clique " << C->name
                 << " (X=" << C->x->name << ", Y=" << C->y->name << ")"
                 << " LL=" << setprecision(11) << clique_ll << endl;
        } else {
            num_failed++;
            cout << "  FAILED: Clique " << C->name
                 << " (X=" << C->x->name << ", Y=" << C->y->name << ")"
                 << " LL=" << setprecision(11) << clique_ll
                 << " diff=" << abs_diff << endl;
        }
    }

    cout << "\nVerification summary:" << endl;
    cout << "  Passed: " << num_passed << "/" << this->cliqueT->cliques.size() << endl;
    if (num_failed > 0) {
        cout << "  FAILED: " << num_failed << "/" << this->cliqueT->cliques.size() << endl;
    } else {
        cout << "  All cliques compute the same log-likelihood!" << endl;
    }

    // Restore reference log-likelihood
    this->logLikelihood = reference_ll;
}

void SEM::ComputeLogLikelihoodUsingPatternsWithPropagationOptimized() {
    // Optimized version: split initialization into base potential (once) and evidence (per site)
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Call ReadPatternsFromFile() first." << endl;
        return;
    }

    if (this->cliqueT == nullptr || this->cliqueT->root == nullptr) {
        this->ConstructCliqueTree();
    }

    this->logLikelihood = 0;
    clique* rootClique = this->cliqueT->root;

    // Get pattern-to-taxon mapping
    map<int, int> pattern_index_to_vertex_index;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed && v->pattern_index >= 0) {
            pattern_index_to_vertex_index[v->pattern_index] = v->id;
        }
    }

    int num_taxa = packed_patterns->get_num_taxa();

    // Save original DNAcompressed data for observed vertices
    map<SEM_vertex*, vector<int>> saved_DNAcompressed;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed) {
            saved_DNAcompressed[v] = v->DNAcompressed;
            v->DNAcompressed.resize(num_patterns_from_file, 4);
        }
    }

    // Pre-populate all patterns into DNAcompressed
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        vector<uint8_t> pattern = packed_patterns->get_pattern(pattern_idx);

        for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
            auto it = pattern_index_to_vertex_index.find(taxon_idx);
            if (it != pattern_index_to_vertex_index.end()) {
                SEM_vertex* v = (*this->vertexMap)[it->second];
                v->DNAcompressed[pattern_idx] = pattern[taxon_idx];
            }
        }
    }

    // OPTIMIZATION: Initialize base potentials ONCE before the loop
    this->cliqueT->InitializeBasePotentials();

    // Now iterate over each pattern using optimized approach
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        // Apply evidence and reset for this site (instead of full InitializePotentialAndBeliefs)
        this->cliqueT->ApplyEvidenceAndReset(pattern_idx);
        this->cliqueT->CalibrateTree();

        // Marginalize over child variable Y to get P(X | data) for root variable X
        std::array<double, 4> marginalX = rootClique->MarginalizeOverVariable(rootClique->y);

        // Weight by root probabilities
        double siteLikelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += this->rootProbability[dna] * marginalX[dna];
        }

        // Combine with accumulated log scaling factors
        double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);

        // Use pattern weight
        int weight = pattern_weights[pattern_idx];
        this->logLikelihood += siteLogLikelihood * weight;
    }

    // Restore original DNAcompressed data
    for (auto& pair : saved_DNAcompressed) {
        pair.first->DNAcompressed = pair.second;
    }
}

void SEM::ComputeLogLikelihoodUsingPatternsWithPropagationMemoized() {
    // Memoized version: caches messages based on subtree observed variable patterns
    if (packed_patterns == nullptr) {
        cerr << "Error: No patterns loaded. Call ReadPatternsFromFile() first." << endl;
        return;
    }

    if (this->cliqueT == nullptr || this->cliqueT->root == nullptr) {
        this->ConstructCliqueTree();
    }

    clique* rootClique = this->cliqueT->root;

    // Get pattern-to-taxon mapping
    map<int, int> pattern_index_to_vertex_index;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed && v->pattern_index >= 0) {
            pattern_index_to_vertex_index[v->pattern_index] = v->id;
        }
    }

    int num_taxa = packed_patterns->get_num_taxa();

    // Save original DNAcompressed data for observed vertices
    map<SEM_vertex*, vector<int>> saved_DNAcompressed;
    for (auto& pair : *this->vertexMap) {
        SEM_vertex* v = pair.second;
        if (v->observed) {
            saved_DNAcompressed[v] = v->DNAcompressed;
            v->DNAcompressed.resize(num_patterns_from_file, 4);
        }
    }

    // Pre-populate all patterns into DNAcompressed
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        vector<uint8_t> pattern = packed_patterns->get_pattern(pattern_idx);

        for (int taxon_idx = 0; taxon_idx < num_taxa; taxon_idx++) {
            auto it = pattern_index_to_vertex_index.find(taxon_idx);
            if (it != pattern_index_to_vertex_index.end()) {
                SEM_vertex* v = (*this->vertexMap)[it->second];
                v->DNAcompressed[pattern_idx] = pattern[taxon_idx];
            }
        }
    }

    // OPTIMIZATION: Initialize base potentials ONCE before the loop
    this->cliqueT->InitializeBasePotentials();

    // MEMOIZATION: Compute subtree and complement leaves for all cliques (once)
    this->cliqueT->ComputeAllSubtreeLeaves();
    this->cliqueT->ComputeAllComplementLeaves();  // For downward message memoization
    this->cliqueT->ClearAllMessageCaches();
    this->cliqueT->ClearAllCacheStatistics();  // Clear per-edge statistics

    // Timing comparison
    auto start_memoized = chrono::high_resolution_clock::now();

    double logLikelihood_memoized = 0.0;

    // Now iterate over each pattern using memoized approach
    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        // Apply evidence and reset for this site
        this->cliqueT->ApplyEvidenceAndReset(pattern_idx);

        // Use memoized calibration instead of regular CalibrateTree()
        this->cliqueT->CalibrateTreeWithMemoization();

        // Marginalize over child variable Y to get P(X | data) for root variable X
        std::array<double, 4> marginalX = rootClique->MarginalizeOverVariable(rootClique->y);

        // Weight by root probabilities
        double siteLikelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += this->rootProbability[dna] * marginalX[dna];
        }

        // Combine with accumulated log scaling factors
        double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);

        // Use pattern weight
        int weight = pattern_weights[pattern_idx];
        logLikelihood_memoized += siteLogLikelihood * weight;
    }

    auto end_memoized = chrono::high_resolution_clock::now();
    auto duration_memoized = chrono::duration_cast<chrono::microseconds>(end_memoized - start_memoized);

    // Report memoization statistics
    cout << "\n=== Memoization Statistics ===" << endl;
    cout << "Upward messages (leaves to root):" << endl;
    cout << "  Cache hits: " << this->cliqueT->cacheHits << endl;
    cout << "  Cache misses: " << this->cliqueT->cacheMisses << endl;
    double hit_rate = 100.0 * this->cliqueT->cacheHits / (this->cliqueT->cacheHits + this->cliqueT->cacheMisses);
    cout << "  Hit rate: " << hit_rate << "%" << endl;
    cout << "  Messages saved: " << this->cliqueT->cacheHits << " out of "
         << (this->cliqueT->cacheHits + this->cliqueT->cacheMisses) << " upward messages" << endl;

    cout << "Downward messages (root to leaves):" << endl;
    cout << "  Cache hits: " << this->cliqueT->downwardCacheHits << endl;
    cout << "  Cache misses: " << this->cliqueT->downwardCacheMisses << endl;
    double downward_hit_rate = 100.0 * this->cliqueT->downwardCacheHits /
        (this->cliqueT->downwardCacheHits + this->cliqueT->downwardCacheMisses);
    cout << "  Hit rate: " << downward_hit_rate << "%" << endl;

    cout << "Memoized computation time: " << duration_memoized.count() / 1000.0 << " ms" << endl;
    cout << "Log-likelihood (memoized): " << logLikelihood_memoized << endl;

    // Write per-edge cache statistics to CSV
    this->cliqueT->WriteCacheStatisticsToCSV("cache_statistics.csv");

    // Now compute without memoization for comparison
    this->cliqueT->ClearAllMessageCaches();
    auto start_regular = chrono::high_resolution_clock::now();

    double logLikelihood_regular = 0.0;

    for (int pattern_idx = 0; pattern_idx < num_patterns_from_file; pattern_idx++) {
        this->cliqueT->ApplyEvidenceAndReset(pattern_idx);
        this->cliqueT->CalibrateTree();  // Regular calibration

        std::array<double, 4> marginalX = rootClique->MarginalizeOverVariable(rootClique->y);

        double siteLikelihood = 0.0;
        for (int dna = 0; dna < 4; dna++) {
            siteLikelihood += this->rootProbability[dna] * marginalX[dna];
        }

        double siteLogLikelihood = rootClique->logScalingFactorForClique + log(siteLikelihood);
        int weight = pattern_weights[pattern_idx];
        logLikelihood_regular += siteLogLikelihood * weight;
    }

    auto end_regular = chrono::high_resolution_clock::now();
    auto duration_regular = chrono::duration_cast<chrono::microseconds>(end_regular - start_regular);

    cout << "\nRegular computation time: " << duration_regular.count() / 1000.0 << " ms" << endl;
    cout << "Log-likelihood (regular): " << logLikelihood_regular << endl;

    // Report speedup
    double speedup = (double)duration_regular.count() / duration_memoized.count();
    cout << "\nSpeedup factor: " << speedup << "x" << endl;
    cout << "Time saved: " << (duration_regular.count() - duration_memoized.count()) / 1000.0 << " ms" << endl;

    // Verify correctness
    double diff = abs(logLikelihood_memoized - logLikelihood_regular);
    cout << "\nLog-likelihood difference: " << diff << endl;
    if (diff < 1e-10) {
        cout << "PASS: Memoized and regular results match!" << endl;
    } else {
        cout << "WARNING: Results differ by " << diff << endl;
    }

    // Set the final log-likelihood
    this->logLikelihood = logLikelihood_memoized;

    // Restore original DNAcompressed data
    for (auto& pair : saved_DNAcompressed) {
        pair.first->DNAcompressed = pair.second;
    }
}

void SEM::embh_aitken(int max_iterations) {
    // ============================================================================
    // IMPROVED AITKEN ACCELERATION FOR EM CONVERGENCE
    // Uses F81 model as initial parameters
    // Disables Aitken when rate factor becomes unreliable (> 0.95)
    // ============================================================================

    const int MAX_ITERATIONS = 100;
    const double MAX_RELIABLE_RATE = 0.95;     // Aitken unreliable above this
    const int MIN_ITER_FOR_AITKEN = 3;         // Need 3 points for Aitken

    // Compute number of sites (sum of pattern weights)
    int num_sites = 0;
    for (int w : this->DNAPatternWeights) {
        num_sites += w;
    }

    // Tolerance: 1e-5 log-likelihood units per site
    const double PER_SITE_TOLERANCE = 1e-5;
    const double TOLERANCE = PER_SITE_TOLERANCE * num_sites;

    this->maxIter = max_iterations;

    // Compute initial TRUE log-likelihood using propagation algorithm
    this->ComputeLogLikelihoodUsingPatternsWithPropagation();
    double ll_0 = this->logLikelihood;

    cout << "\n====================================================" << endl;
    cout << "EM with Aitken Acceleration (F81 initialization)" << endl;
    cout << "====================================================" << endl;
    cout << "Initial LL: " << fixed << setprecision(2) << ll_0 << endl;
    cout << "Number of sites: " << num_sites << endl;
    cout << "Convergence tolerance: " << scientific << setprecision(2) << TOLERANCE
         << " (" << PER_SITE_TOLERANCE << " per site)" << endl;
    cout << string(70, '-') << endl;

    // Variables for Aitken acceleration
    double ll_prev_prev = ll_0;  // LL at n-1
    double ll_prev = ll_0;       // LL at n
    double ll_curr = ll_0;       // LL at n+1

    int final_iter = 0;
    bool converged = false;
    string convergence_reason = "";

    this->ConstructCliqueTree();

    for (int iter = 1; iter <= MAX_ITERATIONS; iter++) {
        final_iter = iter;

        // E-step: Compute expected counts
        this->ComputeExpectedCounts();
        this->ComputeMarginalProbabilitiesUsingExpectedCounts();

        // M-step: Update parameters
        this->ComputeMLEstimateOfBHGivenExpectedDataCompletion();

        // Compute new log-likelihood
        this->ComputeLogLikelihoodUsingPatternsWithPropagation();
        ll_curr = this->logLikelihood;

        double improvement = ll_curr - ll_prev;

        // Compute Aitken estimate (if applicable)
        bool use_aitken = false;
        double aitken_ll = ll_curr;
        double aitken_distance = 0.0;
        double rate_factor = 0.0;

        if (iter >= MIN_ITER_FOR_AITKEN) {
            double delta1 = ll_prev - ll_prev_prev;      // Δ_n
            double delta2 = ll_curr - ll_prev;           // Δ_{n+1}
            double denominator = delta2 - delta1;

            // Compute rate factor
            if (fabs(delta1) > 1e-12) {
                rate_factor = fabs(delta2 / delta1);
            }

            // Only use Aitken if:
            // 1. Denominator is not too small (numerical stability)
            // 2. Rate factor is reasonable (< 0.95)
            // 3. We have valid deltas
            if (fabs(denominator) > 1e-10 &&
                rate_factor > 0.01 &&
                rate_factor < MAX_RELIABLE_RATE) {

                // Aitken acceleration formula
                aitken_ll = ll_curr - (delta2 * delta2) / denominator;
                aitken_distance = fabs(aitken_ll - ll_curr);
                use_aitken = true;
            }
        }

        // Display iteration info
        cout << "Iter " << setw(3) << iter << ": LL = "
             << fixed << setprecision(2) << ll_curr
             << " (+" << scientific << setprecision(2) << improvement << ")";

        if (use_aitken) {
            cout << " | Rate: " << fixed << setprecision(3) << rate_factor
                 << " | Aitken LL: " << fixed << setprecision(2) << aitken_ll
                 << " | Dist: " << scientific << setprecision(2) << aitken_distance;
        } else if (iter >= MIN_ITER_FOR_AITKEN) {
            cout << " | Rate: " << fixed << setprecision(3) << rate_factor
                 << " | Aitken: disabled (rate too high)";
        }
        cout << endl;

        // Convergence check
        if (use_aitken && aitken_distance < TOLERANCE) {
            converged = true;
            convergence_reason = "Aitken criterion (distance < " +
                                to_string(TOLERANCE) + ")";
        } else if (improvement < TOLERANCE) {
            converged = true;
            convergence_reason = "Standard criterion (improvement < " +
                                to_string(TOLERANCE) + ")";
        } else if (improvement < 0) {
            cout << "WARNING: Likelihood decreased! Stopping." << endl;
            converged = true;
            convergence_reason = "Likelihood decrease detected";
        }

        if (converged) {
            break;
        }

        // Update for next iteration
        ll_prev_prev = ll_prev;
        ll_prev = ll_curr;

        if (iter == MAX_ITERATIONS) {
            convergence_reason = "Maximum iterations reached";
        }
    }

    // ============================================================================
    // FINALIZATION
    // ============================================================================

    cout << string(70, '-') << endl;
    if (converged) {
        cout << "Converged after " << final_iter << " iterations" << endl;
        cout << "Reason: " << convergence_reason << endl;
    } else {
        cout << "Reached maximum iterations (" << MAX_ITERATIONS << ")" << endl;
    }

    cout << "\n====================================================" << endl;
    cout << "Final Results:" << endl;
    cout << "  Initial LL:  " << fixed << setprecision(2) << ll_0 << endl;
    cout << "  Final LL:    " << fixed << setprecision(2) << this->logLikelihood << endl;
    cout << "  Improvement: " << fixed << setprecision(2)
         << (this->logLikelihood - ll_0) << endl;
    cout << "  Iterations:  " << final_iter << endl;
    cout << "  LL/site:     " << fixed << setprecision(6)
         << (this->logLikelihood / num_sites) << endl;
    cout << "====================================================" << endl;
}

emtr::Md SEM::GetP_yGivenx(emtr::Md P_xy) {
	emtr::Md P_yGivenx = emtr::Md{};
	array <double, 4> P_x;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		P_x[dna_x] = 0;
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			P_x[dna_x] += P_xy[dna_x][dna_y];
		}
	}
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			// P_yGivenx[dna_x][dna_y] = P_xy[dna_x][dna_y] / P_x[dna_x];
			if (P_xy[dna_x][dna_y] == 0 && P_x[dna_x] == 0) {
				P_yGivenx[dna_x][dna_y] = 0;				
			} else {						
				P_yGivenx[dna_x][dna_y] = P_xy[dna_x][dna_y] / P_x[dna_x];
			}						
			if (isnan(P_yGivenx[dna_x][dna_y])) {
				cout << "P_yGivenx[" << dna_x << "][" << dna_y << "]" << " is nan " << endl;
				cout << "P_xy[" << dna_x << "][" << dna_y << "]" << " is " << P_xy[dna_x][dna_y]  << endl;
				cout << "P_x[" << dna_x << "] is " << P_x[dna_x] << endl;				
				throw mt_error ("nan found");
			}
		}
	}
	return (P_yGivenx);
}

void SEM::ComputeMLEstimateOfBHGivenExpectedDataCompletion() {	
	SEM_vertex * x; SEM_vertex * y;
	emtr::Md P_xy;
	for (pair <int, SEM_vertex*> idPtrPair : *this->vertexMap) {
		y = idPtrPair.second;
		x = y->parent;
		if (x != y) {
			if (x->id < y->id) {
				P_xy = this->posteriorProbabilityForVertexPair[make_pair(x,y)];
			} else {
				// Check following step				
				P_xy = emtr::MT(this->posteriorProbabilityForVertexPair[make_pair(y,x)]);
				
			}
			// MLE of transition matrices
			// cout << "eight eight" << endl;
			y->transitionMatrix = this->GetP_yGivenx(P_xy);
			// cout << "nine nine" << endl;
		} else {			
			// MLE of root probability
			this->rootProbability = this->posteriorProbabilityForVertex[y];
			y->rootProbability = this->rootProbability;
			y->transitionMatrix = emtr::Md{};
			for (int i = 0; i < 4; i ++) {
				y->transitionMatrix[i][i] = 1.0;
			}
			// cout << "ten ten" << endl;
		}
	}
}

void SEM::RootTreeAtVertex(SEM_vertex* r) {	
	this->ClearDirectedEdges();
	vector <SEM_vertex*> verticesToVisit;
	vector <SEM_vertex*> verticesVisited;
	SEM_vertex * p;	
	verticesToVisit.push_back(r);	
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		p = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(p);
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex* c : p->neighbors) {
			if (find(verticesVisited.begin(),verticesVisited.end(),c)==verticesVisited.end()) {
				p->AddChild(c);
				c->AddParent(p);				
				verticesToVisit.push_back(c);
				numberOfVerticesToVisit += 1;
			}
		}
	}
	this->root = r;	
	this->SetEdgesForTreeTraversalOperations();
}

void SEM::ClearDirectedEdges() {
	// cout << "Resetting times visited " << endl;
	this->ResetTimesVisited();
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		v->parent = v;
		v->children.clear();		
		v->inDegree = 0;
		v->outDegree = 0;		
	}
}

void SEM::ResetTimesVisited() {
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;		
		v->timesVisited = 0;		
	}
}

array <double, 4> SEM::GetObservedCountsForVariable(SEM_vertex * v) {
	array <double, 4> observedCounts;
	for (int i = 0; i < 4; i++) {
		observedCounts[i] = 0;
 	}
	for (int site = 0; site < this->num_dna_patterns; site ++) {
		if (v->DNArecoded[site] < 4) { // FIX_AMB
			observedCounts[v->DNArecoded[site]] += this->DNAPatternWeights[site];
		}		
	}
	return (observedCounts);
}

void SEM::EvaluateBHModelWithRootAtCheck(string root_check_name) {
	this->cliqueT = nullptr;
	SEM_vertex * r = this->GetVertex(root_check_name);
	this->ReparameterizeBH();
	this->RootTreeAtVertex(r);
	this->SetModelParametersUsingHSS();
	this->ComputeLogLikelihoodUsingPatternsWithPropagationOptimized();
	cout << "Log-likelihood with root at " << root_check_name << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
}

void SEM::SetModelParametersUsingHSS() {
	
	// cout << "root is set at " << this->root->name << endl;
	// set root probability 
	for (int x = 0; x < 4; x++) {
		this->rootProbability[x] = this->root->root_prob_hss[x];
		this->root->rootProbability[x] = this->root->root_prob_hss[x];
		// cout << "root probability for " << x << " is " << this->rootProbability[x] << endl;
	}


	SEM_vertex * p; SEM_vertex * c; emtr::Md M_pc;
	for (pair<SEM_vertex*,SEM_vertex*> edge : this->edgesForPostOrderTreeTraversal) {		
		p = edge.first;
		c = edge.second;		
		M_pc = (*this->M_hss)[{p,c}];	
		c->transitionMatrix = M_pc;
	}
}

void SEM::ResetLogScalingFactors() {
	for (pair <int, SEM_vertex * > idPtrPair : * this->vertexMap){
		idPtrPair.second->logScalingFactors = 0;
	}
}

void SEM::SetEdgesForTreeTraversalOperations() {
	this->SetEdgesForPreOrderTraversal();
	this->SetEdgesForPostOrderTraversal();
	this->SetVerticesForPreOrderTraversalWithoutLeaves();	
	this->SetLeaves();
}

void SEM::SetEdgesForPreOrderTraversal() {
	this->edgesForPreOrderTreeTraversal.clear();
	vector <SEM_vertex*> verticesToVisit;	
	SEM_vertex * p;	
	verticesToVisit.push_back(this->root);
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		p = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex* c : p->children){
			this->edgesForPreOrderTreeTraversal.push_back(pair<SEM_vertex*, SEM_vertex*>(p,c));
			verticesToVisit.push_back(c);
			numberOfVerticesToVisit += 1;
		}
	}
}

void SEM::SetLeaves() {
	this->leaves.clear();
	for (pair<int, SEM_vertex*> idPtrPair : * this->vertexMap){
		if (idPtrPair.second->outDegree == 0) {
			this->leaves.push_back(idPtrPair.second);
		}
	}
}

void SEM::SetEdgesForPostOrderTraversal() {	
	vector <SEM_vertex*> verticesToVisit;	
	SEM_vertex* c;
	SEM_vertex* p;	
	for (pair<int,SEM_vertex*> idPtrPair : *this->vertexMap){
		idPtrPair.second->timesVisited = 0;		
	}	
	if (this->leaves.size()== 0) {
		this->SetLeaves();
	}
	this->edgesForPostOrderTreeTraversal.clear();	
	verticesToVisit = this->leaves;
	int numberOfVerticesToVisit = verticesToVisit.size();	
	while (numberOfVerticesToVisit > 0) {
		c = verticesToVisit[numberOfVerticesToVisit -1];
		verticesToVisit.pop_back();
		numberOfVerticesToVisit -= 1;
		if (c != c->parent) {
			p = c->parent;
			this->edgesForPostOrderTreeTraversal.push_back(pair<SEM_vertex*, SEM_vertex*>(p,c));
			p->timesVisited += 1;
			if (p->timesVisited == p->outDegree) {
				verticesToVisit.push_back(p);
				numberOfVerticesToVisit += 1;
			}
		}
	}
}


void SEM::SetVerticesForPreOrderTraversalWithoutLeaves() {
	this->preOrderVerticesWithoutLeaves.clear();
	for (pair <SEM_vertex*, SEM_vertex*> edge : this->edgesForPreOrderTreeTraversal) {
		if (find(this->preOrderVerticesWithoutLeaves.begin(),this->preOrderVerticesWithoutLeaves.end(),edge.first) == this->preOrderVerticesWithoutLeaves.end()){
			this->preOrderVerticesWithoutLeaves.push_back(edge.first);
		}
	}
}

void SEM::SetEdgesFromTopologyFile(const string edgeListFileName) {
	SEM_vertex * u; SEM_vertex * v;
	vector <string> splitLine;
	string u_name; string v_name; double t;	
	int num_edges = 0;
	vector <int> emptySequence;
	ifstream inputFile(edgeListFileName.c_str());
	for(string line; getline(inputFile, line );) {
		num_edges++;
		vector<string> splitLine = emtr::split_ws(line);
		u_name = splitLine[0];
		v_name = splitLine[1];
		t = stod(splitLine[2]);
		if (this->ContainsVertex(u_name)) {
			u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		} else {
			u = new SEM_vertex(this->node_ind,emptySequence);
			u->name = u_name;
			u->id = this->node_ind;
			this->vertexMap->insert(pair<int,SEM_vertex*>(u->id,u));
			this->nameToIdMap.insert(make_pair(u->name,u->id));			
			this->node_ind += 1;
			if(!this->ContainsVertex(u_name)){
                throw mt_error("check why u is not in vertex map");
            }
		}

		if (this->ContainsVertex(v_name)) {
			v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		} else {			
			v = new SEM_vertex(this->node_ind,emptySequence);
			v->name = v_name;
			v->id = this->node_ind;
			this->vertexMap->insert(pair<int,SEM_vertex*>(v->id,v));
			this->nameToIdMap.insert(make_pair(v->name,v->id));
			this->node_ind += 1;
			if(!this->ContainsVertex(v_name)){
                throw mt_error("Check why v is not in vertex map");
            }
		}
		u->AddNeighbor(v);
		v->AddNeighbor(u);		
		if (u->id < v->id) {
			this->edgeLengths.insert(make_pair(make_pair(u,v),t));			
		} else {
			this->edgeLengths.insert(make_pair(make_pair(v,u),t));
		}		
	}
	inputFile.close();
	cout << "number of edges in topology file is " << num_edges << endl;
}

void SEM::SetRootProbabilityAsSampleBaseComposition(string baseCompositionFileName) {
	ifstream inFile(baseCompositionFileName);
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open file " << baseCompositionFileName << endl;
        return;
    }
    
    string line;
    
    // Parse the file line by line
    while (getline(inFile, line)) {
        istringstream iss(line);
        string firstToken;
        double frequency;
        
        // Read the first token (base index: 0, 1, 2, or 3)
        if (iss >> firstToken >> frequency) {
            // Check if it's a base index (0, 1, 2, or 3)
            if (firstToken == "0" || firstToken == "1" || 
                firstToken == "2" || firstToken == "3") {
                int baseIndex = stoi(firstToken);
                this->rootProbability[baseIndex] = frequency;
            }
        }
    }
    inFile.close();	
}

void SEM::SetF81_mu() {
	// base composition π
    array <double, 4> pi = this->rootProbability; // pi[0..3] = {A,C,G,T}

    // compute S2 = sum pi_i^2
    double S2 = 0.0;
    for (int k = 0; k < 4; ++k) S2 += pi[k] * pi[k];

    // μ chosen so that expected rate = 1 
    // guard against pathological S2≈1 when probability is a point mass
    const double denom = std::max(1e-14, 1.0 - S2);
    this->F81_mu = 1.0 / denom;
	cout << "F81 mu is " << this->F81_mu << endl;
}

void SEM::SetF81Matrix(SEM_vertex* v) {	
    // fetch branch length t (as expected subs/site)
    double t = this->GetLengthOfSubtendingBranch(v);
	// base composition π
	array <double, 4> pi = this->rootProbability; // pi[0..3] = {A,C,G,T}
	assert (t > 0);
    const double e = exp(-this->F81_mu * t);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double p;
            if (i == j) {
                p = pi[i] + (1.0 - pi[i]) * e;
            } else {
                p = pi[j] * (1.0 - e);
            }
            v->transitionMatrix[i][j] = p;
        }
    }
}

void SEM::SetF81Model(string baseCompositionFileName) {
	this->SetRootProbabilityAsSampleBaseComposition(baseCompositionFileName);
	this->SetF81_mu();
	cout << "number of non-root vertices is " << this->non_root_vertices.size() << endl;
	for (SEM_vertex * v: this->non_root_vertices) SetF81Matrix(v);
}

void SEM::CompressDNASequences() {
    this->DNAPatternWeights.clear();
    this->gapLessDNAFlag.clear();
    this->DNAPatternGapProp.clear();
    this->DNAPatternUniqueCount.clear();

    std::vector<std::vector<int>> uniquePatterns;
    std::map<std::vector<int>, std::vector<int>> uniquePatternsToSitesWherePatternRepeats;

    // Clear per-leaf compressed storage
    for (unsigned int i = 0; i < this->leaves.size(); ++i) {
        SEM_vertex* l_ptr = this->leaves[i];
        l_ptr->DNAcompressed.clear();
    }

    // Build unique patterns and where they repeat
    const int numberOfSites = static_cast<int>(this->leaves[0]->DNArecoded.size());
    std::vector<int> sitePattern;
    sitePattern.reserve(this->leaves.size());

    for (int site = 0; site < numberOfSites; ++site) {
        sitePattern.clear();
        for (SEM_vertex* l_ptr : this->leaves) {
            sitePattern.push_back(l_ptr->DNArecoded[site]);
        }

        auto it = uniquePatternsToSitesWherePatternRepeats.find(sitePattern);
        if (it != uniquePatternsToSitesWherePatternRepeats.end()) {
            it->second.push_back(site);
        } else {
            uniquePatterns.push_back(sitePattern);
            uniquePatternsToSitesWherePatternRepeats[uniquePatterns.back()] = std::vector<int>(1, site);
        }
    }

    // Weights and gap-less flags (initial order), plus new stats
    for (const std::vector<int>& pat : uniquePatterns) {
        int w = static_cast<int>(uniquePatternsToSitesWherePatternRepeats[pat].size());
        this->DNAPatternWeights.push_back(w);

        bool gapLess = true;
        for (int ch : pat) { if (ch < 0) { gapLess = false; break; } }
        this->gapLessDNAFlag.push_back(gapLess);

        // (i) proportion of gaps (per pattern, across leaves)
        this->DNAPatternGapProp.push_back(gap_proportion_in_pattern(pat));

        // (ii) number of unique non-gap characters (per pattern)
        this->DNAPatternUniqueCount.push_back(unique_non_gap_count_in_pattern(pat));
    }

    this->num_dna_patterns = static_cast<int>(this->DNAPatternWeights.size());
    if (this->num_dna_patterns == 0) {
        this->CumPercDNAPatternWeights.clear();
        return;
    }

    // Sort pattern indices by decreasing weight
    std::vector<size_t> order(this->DNAPatternWeights.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
        [this](size_t a, size_t b) {
            return this->DNAPatternWeights[a] > this->DNAPatternWeights[b];
        }
    );

    this->DNAPatternOriginalSites.clear();
    this->DNAPatternOriginalSites.resize(order.size());

    this->OriginalToCompressed.assign(numberOfSites, -1);

    for (size_t k = 0; k < order.size(); ++k) {
        const size_t pidx = order[k];
        const std::vector<int>& sites =
            uniquePatternsToSitesWherePatternRepeats[ uniquePatterns[pidx] ];

        this->DNAPatternOriginalSites[k] = sites;

        for (int s : sites) {
            if (s >= 0 && s < numberOfSites) {
                this->OriginalToCompressed[s] = static_cast<int>(k);
            }
        }
    }

    // Rebuild DNAcompressed and metadata in sorted order
    for (SEM_vertex* l_ptr : this->leaves) {
        l_ptr->DNAcompressed.clear();
        l_ptr->DNAcompressed.reserve(order.size());
    }

    std::vector<int>    newWeights;     newWeights.reserve(order.size());
    std::vector<bool>   newGapLess;     newGapLess.reserve(order.size());
    std::vector<double> newGapProp;     newGapProp.reserve(order.size());
    std::vector<int>    newUniqueCount; newUniqueCount.reserve(order.size());

    for (size_t k = 0; k < order.size(); ++k) {
        const size_t pidx = order[k];
        const std::vector<int>& pat = uniquePatterns[pidx];

        // compressed columns per leaf
        for (unsigned int i = 0; i < pat.size(); ++i) {
            SEM_vertex* l_ptr = this->leaves[i];
            l_ptr->DNAcompressed.push_back(pat[i]);
        }
       
        newWeights.push_back(this->DNAPatternWeights[pidx]);
        newGapLess.push_back(this->gapLessDNAFlag[pidx]);
        newGapProp.push_back(this->DNAPatternGapProp[pidx]);
        newUniqueCount.push_back(this->DNAPatternUniqueCount[pidx]);
    }

    this->DNAPatternWeights.swap(newWeights);
    this->gapLessDNAFlag.swap(newGapLess);
    this->DNAPatternGapProp.swap(newGapProp);
    this->DNAPatternUniqueCount.swap(newUniqueCount);
    
    this->CumPercDNAPatternWeights.clear();
    this->CumPercDNAPatternWeights.reserve(this->DNAPatternWeights.size());

    const double totalW = std::accumulate(
        this->DNAPatternWeights.begin(), this->DNAPatternWeights.end(), 0.0);

    double running = 0.0;
    if (totalW > 0.0) {
        for (size_t i = 0; i < this->DNAPatternWeights.size(); ++i) {
            running += static_cast<double>(this->DNAPatternWeights[i]);
            double pct = (running / totalW) * 100.0;
            if (pct > 100.0) pct = 100.0;
            this->CumPercDNAPatternWeights.push_back(pct);
        }
        this->CumPercDNAPatternWeights.back() = 100.0;
    } else {        
        for (size_t i = 0; i < this->DNAPatternWeights.size(); ++i) {
            this->CumPercDNAPatternWeights.push_back(0.0);
        }
    }
}

void SEM::SetDNASequencesFromFile(string sequenceFileName) {
	this->leaves.clear();
	this->DNAsequenceFileName = sequenceFileName;
	vector <int> recodedSequence;
	recodedSequence.clear();
	unsigned int site = 0;
	unsigned int seq_len = 0;
	(void)seq_len; // suppress unused warning
	int dna_index;
	ifstream inputFile(sequenceFileName.c_str());
	string seqName;
	string seq = "";
	SEM_vertex * u;
	this->numberOfInputSequences = 0;
	this->numberOfObservedVertices = 0;
	for(string line; getline(inputFile, line );) {
		if (line[0]=='>') {
			if (seq != "") {
				for (char const dna: seq) {
					if (!isspace(dna)) {
						dna_index = this->ConvertDNAtoIndex(dna);						
						recodedSequence.push_back(dna_index);					
						site += 1;							
						}						
				}
				this->AddVertex(seqName,recodedSequence);
				u = this->GetVertex(seqName);
				this->leaves.push_back(u);
				u->observed = 1;
				this->numberOfInputSequences++;
				this->numberOfObservedVertices++;								
				recodedSequence.clear();
			} 
			seqName = line.substr(1,line.length());
			seq = "";
			site = 0;			
		}
		else {
			seq += line ;
		}		
	}		
	for (char const dna: seq) {
		if (!isspace(dna)) {
			dna_index = this->ConvertDNAtoIndex(dna);			
			recodedSequence.push_back(dna_index);
			site += 1;
		}
	}
	this->AddVertex(seqName,recodedSequence);
	u = this->GetVertex(seqName);
	this->leaves.push_back(u);
	u->observed = 1;
	this->numberOfInputSequences++;
	this->numberOfObservedVertices++;								
	recodedSequence.clear();
    seq_len = recodedSequence.size();
	recodedSequence.clear();
	inputFile.close(); 
}

bool SEM::ContainsVertex(string v_name) {	
	if (this->nameToIdMap.find(v_name) == this->nameToIdMap.end()) {		
		return (0);
	} else {
		return (1);
	}
}

void SEM::AddVertex(string u_name, vector <int> emptySequence) {	
	SEM_vertex * u = new SEM_vertex(this->node_ind,emptySequence);
	u->name = u_name;
	u->id = this->node_ind;
	this->node_ind++;
	this->vertexMap->insert(pair<int,SEM_vertex*>(u->id,u));
	this->nameToIdMap.insert(make_pair(u->name,u->id));	
}

manager::manager(const string edge_list_file_name,
		   const string fasta_file_name,
    	   const string pattern_file_name,
		     const string taxon_order_file_name,
           const string base_composition_file_name,
           const string root_optim_name,
           const string root_check_name) {
		this->prefix_for_output_files = "";		
		this->verbose = 0;		
		this->max_EM_iter = max_iter;
		this->SetDNAMap();

		this->P = new SEM(0.0,100,this->verbose);
		this->P->SetDNASequencesFromFile(fasta_file_name);
		this->P->CompressDNASequences();
		this->P->SetEdgesFromTopologyFile(edge_list_file_name);
		this->P->SetVertexVector();
		SEM_vertex * root_optim = this->P->GetVertex(root_optim_name);
		this->P->RootTreeAtVertex(root_optim);
		this->P->SetVertexVectorExceptRoot();
		cout << "Number of non-root vertices is " << this->P->non_root_vertices.size() << endl;		
		cout << "Number of unique site patterns is " << this->P->num_dna_patterns << endl;		
		this->P->SetF81Model(base_composition_file_name); // assign transition probabilities
		// [] compute log-likelihood score using pruning algorithm with fasta input
		cout << "\n=== Computing log-likelihood using compressed sequences ===" << endl;
		this->P->ComputeLogLikelihood();
		cout << "log-likelihood using pruning algorithm and compressed sequences is " << setprecision(11) << this->P->logLikelihood << endl;

		if (!pattern_file_name.empty() && !taxon_order_file_name.empty()) {
			cout << "\n=== Computing log-likelihood using packed patterns ===" << endl;
			this->P->ReadPatternsFromFile(pattern_file_name, taxon_order_file_name);
			this->P->ComputeLogLikelihoodUsingPatterns();
			cout << "log-likelihood using pruning algorithm and packed patterns is "
			     << setprecision(11) << this->P->logLikelihood << endl;

			// Compute using OPTIMIZED propagation algorithm (split initialization)
			cout << "\n=== Computing log-likelihood using OPTIMIZED propagation algorithm ===" << endl;
			this->P->ComputeLogLikelihoodUsingPatternsWithPropagationOptimized();
			cout << "log-likelihood using OPTIMIZED propagation algorithm is "
			     << setprecision(11) << this->P->logLikelihood << endl;

			// Compute using MEMOIZED propagation algorithm (with timing comparison)
			cout << "\n=== Computing log-likelihood using MEMOIZED propagation algorithm ===" << endl;
			this->P->ComputeLogLikelihoodUsingPatternsWithPropagationMemoized();
			cout << "log-likelihood using MEMOIZED propagation algorithm is "
			     << setprecision(11) << this->P->logLikelihood << endl;
		}
		this->P->embh_aitken(100);		
		// [] compute log-likelihood score using pruning algorithm with pattern input
		// [] compute log-likelihood score using propagation algorithm with pattern input
		/* [] reuse messages for branch patterns and compute log-likelihood
		 using propagation algorithm with */
		// the following is for BH modelgre		
		// this->RunEMWithAitken();		
		this->P->EvaluateBHModelWithRootAtCheck(root_check_name);

		this->P->VerifyLogLikelihoodAtAllCliques();
		// set root estimate and root test
		}

		manager::~manager() {			
			delete this->P;		
		}

void manager::SetDNAMap() {
	this->mapDNAtoInteger["A"] = 0;
	this->mapDNAtoInteger["C"] = 1;
	this->mapDNAtoInteger["G"] = 2;
	this->mapDNAtoInteger["T"] = 3;
}