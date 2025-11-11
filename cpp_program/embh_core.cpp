#include "embh_core.hpp"
#include "embh_utils.hpp"

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


using namespace std; 
using namespace emtr;


int ll_precision = 14;

///...///...///...///...///...///...///...///...///... SEM_vertex ///...///...///...///...///...///...///...///...///

class pattern {
	public:
 		int weight;
		vector <char> characters;
	pattern (int weightToSet, vector <char> charactersToSet) {
		this->weight = weightToSet;
		this->characters = charactersToSet;
	}
};

///...///...///...///...///...///...///...///...///... 3-BIT PACKED PATTERN STORAGE ///...///...///...///...///...///...///...///...///

// 3-bit packed pattern storage class
// Stores DNA bases (0-4) using 3 bits per base for 60% memory savings
// Values: 0=A, 1=C, 2=G, 3=T, 4=gap
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

///...///...///...///...///...///...///...///...///... SEM_vertex ///...///...///...///...///...///...///...///...///

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

///...///...///...///...///...///...///... clique ...///...///...///...///...///...///...///

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
	emtr::Md belief;
	// P(X,Y)
	
	void SetInitialPotentialAndBelief(int site);
	
	// If the clique contains an observed variable then initializing
	// the potential is the same as restricting the corresponding
	// CPD to row corresponding to observed variable
	void AddNeighbor(clique * C);
	clique (SEM_vertex * x, SEM_vertex * y) {
		this->x = x;
		this->y = y;
		this->name = to_string(x->id) + "-" + to_string(y->id);
		this->logScalingFactorForClique = 0;
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
		double message_sum = 0;		
				
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
	int matchingCase;
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

///...///...///...///...///...///...///...///... clique tree ...///...///...///...///...///...///...///...///

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
	void SetEdgesForTreeTraversalOperations();
	void WriteCliqueTreeAndPathLengthForCliquePairs(string fileName);
	emtr::Md GetP_XZ(SEM_vertex * X, SEM_vertex * Y, SEM_vertex * Z);	
	SEM_vertex * GetCommonVariable(clique * Ci, clique * Cj);
	tuple <SEM_vertex *,SEM_vertex *,SEM_vertex *> GetXYZ(clique * Ci, clique * Cj);
	cliqueTree () {
		rootSet = 0;
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

///...///...///...///...///...///...///...///... Params Struct ///...///...///...///...///...///...///...///...///

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


///...///...///...///...///...///...///...///...///... SEM ...///...///...///...///...///...///...///...///...///

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
	void SetInitialEstimateOfModelParametersUsingHSS();	
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

int SEM::first_index_gt(const std::vector<double>& cum, double thr) {
    constexpr double EPS = 1e-12;
    auto it = std::upper_bound(cum.begin(), cum.end(), thr + EPS);
    return static_cast<int>(it - cum.begin());
}



array <double, 4> SEM::sample_pi() {
	return sample_dirichlet(this->alpha_pi, rng());
}

array <double, 4> SEM::sample_M_row() {
	return sample_dirichlet(this->alpha_M_row, rng());
}

array <double, 4> SEM::sample_dirichlet(const array<double, 4>& alpha, mt19937_64& gen) {
        array<double, 4> x{};
        double sum = 0.0;
        for (size_t i = 0; i < 4; ++i) {
            gamma_distribution<double> gamma(alpha[i], 1.0);
            x[i] = gamma(gen);
            sum += x[i];
        }
        for (auto& v : x) v /= sum;
        return x;
    }

void SEM::set_alpha_PI(double a1, double a2, double a3, double a4) {
	this->alpha_pi[0] = a1;
	this->alpha_pi[1] = a2;
	this->alpha_pi[2] = a3;
	this->alpha_pi[3] = a4;	
}

void SEM::set_alpha_M_row(double a1, double a2, double a3, double a4) {
	this->alpha_M_row[0] = a1;
	this->alpha_M_row[1] = a2;
	this->alpha_M_row[2] = a3;
	this->alpha_M_row[3] = a4;
}

void SEM::SetWeightedEdgesToAddToGlobalPhylogeneticTree() {
	this->weightedEdgesToAddToGlobalPhylogeneticTree.clear();
	int u_id; int v_id;
	SEM_vertex * u; SEM_vertex * v; 
	string u_name; string v_name;
	double t;	
	for (pair <int, int> edge_ind : this->edgesOfInterest_ind) {
		tie (u_id, v_id) = edge_ind;
		u = (*this->vertexMap)[u_id];
		v = (*this->vertexMap)[v_id];
		u_name = u->name;
		v_name = v->name;
		t = this->ComputeEdgeLength(u,v);
		this->weightedEdgesToAddToGlobalPhylogeneticTree.push_back(make_tuple(u_name,v_name,t));
	}
}



void SEM::SetIdsForObservedVertices(vector <int> idsOfObservedVerticesToAdd) {
	this->idsOfObservedVertices = idsOfObservedVerticesToAdd;
}

void SEM::SetNumberOfInputSequences(int numOfInputSeqsToSet) {
	this->numberOfInputSequences = numOfInputSeqsToSet;	
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

void SEM::SuppressRoot() {
	SEM_vertex * c_l;
	SEM_vertex * c_r;
	bool proceed = this->root->outDegree == 2;		
	if (proceed) {		
		c_l = this->root->children[0];		
		c_r = this->root->children[1];		
		c_l->AddNeighbor(c_r);
		c_r->AddNeighbor(c_l);
		c_l->RemoveNeighbor(this->root);
		c_r->RemoveNeighbor(this->root);
		this->RemoveArc(this->root,c_l);
		this->RemoveArc(this->root,c_r);
	}
}

void SEM::SwapRoot() {
	SEM_vertex * root_current;
	SEM_vertex * vertexNamedHRoot;
	vector <SEM_vertex *> childrenOfCurrentRoot;
	vector <SEM_vertex *> childrenOfVertexNamedHRoot;
	int n = this->numberOfObservedVertices;	
	if (this->root->name != "h_root") {
		root_current = this->root;
		childrenOfCurrentRoot = root_current->children;
		
		vertexNamedHRoot = (*this->vertexMap)[((2*n)-2)];		
		childrenOfVertexNamedHRoot = vertexNamedHRoot->children;
		
		// Swap children of root
		for (SEM_vertex * c: childrenOfVertexNamedHRoot) {
			this->RemoveArc(vertexNamedHRoot,c);
			this->AddArc(root_current,c);
		}
		
		for (SEM_vertex * c: childrenOfCurrentRoot) {			
			this->RemoveArc(root_current,c);
			this->AddArc(vertexNamedHRoot,c);
		}
		
		vertexNamedHRoot->rootProbability = root_current->rootProbability;
		root_current->transitionMatrix = vertexNamedHRoot->transitionMatrix;
		vertexNamedHRoot->transitionMatrix = this->I4by4;
		
		this->AddArc(vertexNamedHRoot->parent,root_current);
		this->RemoveArc(vertexNamedHRoot->parent,vertexNamedHRoot);
		this->root = vertexNamedHRoot;
		this->SetLeaves();	
		this->SetEdgesForPreOrderTraversal();
		this->SetVerticesForPreOrderTraversalWithoutLeaves();
		this->SetEdgesForPostOrderTraversal();

	}	
}


char SEM::GetDNAfromIndex(int dna_index){
	char bases[4] = {'A', 'C', 'G', 'T'};
	if (dna_index > 0 && dna_index < 4){
		return bases[dna_index];
	} else if (dna_index == 4) {
		return '-';
	}
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

void SEM::SetBHparameters() {
	this->ReadProbabilities();	
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

void SEM::ReadProbabilities() {
    std::ifstream inputFile(this->probabilityFileName_best);
    if (!inputFile) {
        throw mt_error("Failed to open probability file: " + this->probabilityFileName_best);
    }

    std::string line;
    // skip first two header lines
    if (!std::getline(inputFile, line) || !std::getline(inputFile, line)) {
        throw mt_error("Probability file too short (missing headers)");
    }

    std::string node_name;
	// string node_parent_name;
    double prob; int i, j;

    while (std::getline(inputFile, line)) {
        vector<string> splitLine = emtr::split_ws(line);
        const int num_words = static_cast<int>(splitLine.size());
		// cout << num_words << endl;
        switch (num_words) {
            case 8: { 
				// node_parent_name = splitLine[5];
                node_name = splitLine[7];
                break;
            }
            case 16: {                
                SEM_vertex* n = this->GetVertex(node_name);
                for (int p_id = 0; p_id < 16; ++p_id) {
                    i = p_id / 4;
                    j = p_id % 4;
                    try	{
						prob = stod(splitLine[p_id]); 
					}
					catch(const exception& e) {
						prob = 0;
						cout << "setting to 0 small prob value " << splitLine[p_id] << " not converted by stod" << endl;
						// (*this->logFile) << "setting to 0 small prob value " << splitLine[p_id] << " not converted by stod" << endl;
					}
					n->transitionMatrix[i][j] = prob;
                }
                break;
            }
            case 9: {                
                node_name = splitLine[3];
                SEM_vertex* n = this->GetVertex(node_name);
                this->RootTreeAtVertex(n);
                break;
            }
            case 4: {                
                for (int p_id = 0; p_id < 4; ++p_id) {
                    prob = std::stod(splitLine[p_id]);
                    this->rootProbability[p_id] = prob;
                }
                break;
            }
            default:
                std::cerr << "ReadProbabilities: unexpected token count (" << num_words
                          << ") on line: " << line << "\n";
                break;
        }
    }
}

void SEM::WriteProbabilities(string fileName) {
	ofstream probabilityFile;
	probabilityFile.open(fileName);
	SEM_vertex * v;
	probabilityFile << "transition matrix for edge from " << "parent_name" << " to " << "child_name" << endl;
	
	for (int row = 0; row < 4; row++) {
		for (int col = 0; col < 4; col++) {
			probabilityFile << "p(" << this->GetDNAfromIndex(col) << "|" << this->GetDNAfromIndex(row) << ")";
			if (row ==3 and col ==3) {
				continue;
			} else {
				probabilityFile << " ";
			}
			
		}		
	}
	probabilityFile << endl;

	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		if (v != v->parent) {
			probabilityFile << "transition matrix for edge from " << v->parent->name << " to " << v->name << endl;
			for (int row = 0; row < 4; row++) {
				for (int col = 0; col < 4; col++) {
					probabilityFile << v->transitionMatrix[row][col];
					if (row ==3 and col ==3){
						continue;
					} else {
						probabilityFile << " ";
					}
				}				
			}
			probabilityFile << endl;
		}
	}

	probabilityFile << "probability at root " << this->root->name << " is ";
	for (int row = 0; row < 3; row++) {
		probabilityFile << "p(" << this->GetDNAfromIndex(row) << ") ";		
	}
	probabilityFile << "p(" << this->GetDNAfromIndex(3) << ") " << endl;

	for (int row = 0; row < 3; row++) {
		probabilityFile << this->rootProbability[row] << " ";
	}
	probabilityFile << this->rootProbability[3] << endl;
	probabilityFile.close();
}

void SEM::ReadRootedTree(string treeFileName) {
	string u_name;
	string v_name;
	int u_id;
	int v_id;
	SEM_vertex * u;
	SEM_vertex * v;
	vector <string> splitLine;
	vector <string> leafNames;
	vector <string> ancestorNames;
	vector <string> nonRootVertexNames;	
	string rootName = "";
	vector <int> emptySequence;
	v_id = 0;
	ifstream edgeListFile(treeFileName.c_str());
	for (string line; getline(edgeListFile, line);) {
		vector<string> splitLine = emtr::split_ws(line);
		u_name = splitLine[0];		
		v_name = splitLine[1];
		if (find(nonRootVertexNames.begin(),nonRootVertexNames.end(),v_name) == nonRootVertexNames.end()) {
			nonRootVertexNames.push_back(v_name);
		}		
		if (find(ancestorNames.begin(),ancestorNames.end(),u_name)==ancestorNames.end()) {
			ancestorNames.push_back(u_name);
		}
		if (find(leafNames.begin(),leafNames.end(),v_name)==leafNames.end()) {
			if(!emtr::starts_with(v_name, "h_")) {
				leafNames.push_back(v_name);
			}
		}
	}
	for (string name: leafNames) {
		SEM_vertex * v = new SEM_vertex(v_id, emptySequence);
		v->name = name;
		v->observed = 1;
		this->vertexMap->insert(pair<int,SEM_vertex*>(v_id,v));
		v_id += 1;
	}
	// Remove root from ancestor names
	for (string name: ancestorNames) {
		if (find(nonRootVertexNames.begin(),nonRootVertexNames.end(),name)==nonRootVertexNames.end()){
			rootName = name;
		}
	}
	this->numberOfObservedVertices = leafNames.size();
	int n = this->numberOfObservedVertices;			
	// Change root name
	
	ancestorNames.erase(remove(ancestorNames.begin(), ancestorNames.end(), rootName), ancestorNames.end());
	for (string name: ancestorNames) {
		SEM_vertex * v = new SEM_vertex(v_id,emptySequence);
		v->name = name;
		this->vertexMap->insert(pair <int,SEM_vertex*> (v_id,v));
		v_id += 1;
	}
	
	this->root = new SEM_vertex (((2 * n) - 2), emptySequence);	
	this->root->name = rootName;
	this->vertexMap->insert(pair <int,SEM_vertex*> (((2 * n) - 2), this->root));
	edgeListFile.clear();
	edgeListFile.seekg(0, ios::beg);
	for (string line; getline(edgeListFile, line);) {
		vector<string> splitLine = emtr::split_ws(line);
		u_name = splitLine[0];
		v_name = splitLine[1];
		u_id = this->GetVertexId(u_name);
		v_id = this->GetVertexId(v_name);
		u = (*this->vertexMap)[u_id];
		v = (*this->vertexMap)[v_id];
		u->AddChild(v);
		v->AddParent(u);
	}
	edgeListFile.close();	
	this->SetLeaves();
	// cout << "Number of leaves is " << this->leaves.size() << endl;
	this->SetEdgesForPreOrderTraversal();
	// cout << "Number of edges for pre order traversal is " << this->edgesForPreOrderTreeTraversal.size() << endl;
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
	// cout << "Number of vertices for pre order traversal is " << this->preOrderVerticesWithoutLeaves.size() << endl;
	this->SetEdgesForPostOrderTraversal();
	// cout << "Number of edges for post order traversal is " << this->edgesForPostOrderTreeTraversal.size() << endl;
}


void SEM::AddAllSequences(string fileName) {
	vector <int> recodedSequence;
	ifstream inputFile(fileName.c_str());
	string v_name;
	string seq = "";	
	int v_id;
	vector <string> vertexNames;	
	vector <vector <int>> allSequences;	
	for (string line; getline(inputFile, line );) {
		if (line[0]=='>') {
			if (seq != "") {				
				for (char const dna: seq) {
					recodedSequence.push_back(mapDNAtoInteger[string(1,toupper(dna))]);					
				}				
				v_id = this->GetVertexId(v_name);				
				(*this->vertexMap)[v_id]->DNArecoded = recodedSequence;				
				recodedSequence.clear();
			}
			v_name = line.substr(1,line.length());
			seq = "";
		} else {
			seq += line ;
		}
	}
	inputFile.close();
	
	for (char const dna: seq) {
		recodedSequence.push_back(mapDNAtoInteger[string(1,toupper(dna))]);
	}
	
	v_id = this->GetVertexId(v_name);
	(*this->vertexMap)[v_id]->DNArecoded = recodedSequence;
	
	int numberOfSites = recodedSequence.size();	
	this->num_dna_patterns = numberOfSites;
	this->sequenceLength = numberOfSites;
	recodedSequence.clear();
	
	this->DNAPatternWeights.clear();
	
	for (int i = 0; i < numberOfSites; i++) {
		this->DNAPatternWeights.push_back(1);
	}	
}

void SEM::ResetAncestralSequences() {
	vector<int> gappedSequence ;
	for (int i = 0; i < this->num_dna_patterns; i++) gappedSequence.push_back(4);
	for (pair <int,SEM_vertex*> idPtrPair : *this->vertexMap) {
		if (!idPtrPair.second->observed) {
			// cout << " resetting for " << idPtrPair.second->name << endl;
			idPtrPair.second->DNAcompressed = gappedSequence;
		}
	}
}

void SEM::RemoveEdgeLength(SEM_vertex * u, SEM_vertex * v) {
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	this->edgeLengths.erase(vertexPair);
}

void SEM::AddEdgeLength(SEM_vertex * u, SEM_vertex * v, double t) {	
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	this->edgeLengths[vertexPair] = t;
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

double SEM::GetEdgeLength(SEM_vertex * u, SEM_vertex * v) {
	double t;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	t = this->edgeLengths[vertexPair];
	return (t);
}

double SEM::ComputeEdgeLength(SEM_vertex * u, SEM_vertex * v) {
	double t = 0;
	int dna_u; int dna_v; 
	for (int site = 0; site < this->num_dna_patterns; site++) {
		dna_u = u->DNArecoded[site];
		dna_v = v->DNArecoded[site];
		if (dna_u != dna_v) {
			t += this->DNAPatternWeights[site];
		}
	}
	t /= this->sequenceLength;	
	return (t);
}

void SEM::SetEdgeLength(SEM_vertex * u, SEM_vertex * v, double t) {
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	this->edgeLengths[vertexPair] = t;
}

void SEM::StoreEdgeListAndSeqToAdd() {
	this->weightedEdgeListString = "";		
	this->sequencesToAddToGlobalPhylogeneticTree.clear();
	this->weightedEdgesToAddToGlobalPhylogeneticTree.clear();
	SEM_vertex * u; SEM_vertex * v;	
	double t;	
	for (pair <SEM_vertex *, SEM_vertex *> vertexPair : this->edgesForPostOrderTreeTraversal) {
		tie (u, v) = vertexPair;		
		if (u->parent != u) {
			if (v == this->externalVertex and !this->finalIterationOfSEM) {
				this->compressedSequenceToAddToMST = u->DNArecoded;
				this->nameOfSequenceToAddToMST = u->name;				
			} else {
				t = this->ComputeEdgeLength(u,v);			
//				cout << "Adding edge 1 " << u->name << "\t" << v->name << endl;
				this->weightedEdgeListString += u->name + "\t" + v->name + "\t" + to_string(t) + "\n";
				this->weightedEdgesToAddToGlobalPhylogeneticTree.push_back(make_tuple(u->name,v->name,t));
			}
		}
	}	
	u = this->root->children[0];
	v = this->root->children[1];
	if ((v != this->externalVertex and u!= this->externalVertex) or this->finalIterationOfSEM) {
		t = this->ComputeEdgeLength(u,v);
//		cout << "Adding edge 2 " << u->name << "\t" << v->name << endl;
		this->weightedEdgeListString += u->name + "\t" + v->name + "\t" + to_string(t) + "\n";
		this->weightedEdgesToAddToGlobalPhylogeneticTree.push_back(make_tuple(u->name,v->name,t));
	} else if (u == this->externalVertex) {
		this->compressedSequenceToAddToMST = v->DNArecoded;
		this->nameOfSequenceToAddToMST = v->name;
	} else {
		if (v != this->externalVertex){
            throw mt_error("v should be equal to external vertex");
        }
		this->compressedSequenceToAddToMST = u->DNArecoded;
		this->nameOfSequenceToAddToMST = u->name;
	}
//	cout << "Name of external sequence is " << this->externalVertex->name << endl;
//	cout << "Name of sequence to add to MST is " << this->nameOfSequenceToAddToMST << endl;
	// Add sequences of all vertices except the following vertices
	// 1) root, 2) external vertex
	for (pair <int, SEM_vertex * > idPtrPair : * this->vertexMap) {
		u = idPtrPair.second;		
		if (u->parent != u){
			if (u != this->externalVertex) {
				this->sequencesToAddToGlobalPhylogeneticTree[u->name] = u->DNArecoded;
			} else if (this->finalIterationOfSEM) {
				this->sequencesToAddToGlobalPhylogeneticTree[u->name] = u->DNArecoded;
			}
		}		
	}	
}

emtr::Md SEM::ComputeTransitionMatrixUsingAncestralStatesForTrifle(SEM_vertex * p, SEM_vertex * c, int layer) {	
	emtr::Md P = emtr::Md{};				
	int dna_p; int dna_c;
	
	for (int site = 0; site < this->num_patterns_for_layer[layer]; site ++) {
		dna_p = p->DNAcompressed[site];
		dna_c = c->DNAcompressed[site];
		if (dna_p < 4 && dna_c < 4) {
			P[dna_p][dna_c] += this->DNAPatternWeights[site];
		}
	}

	double rowSum;
	for (int i = 0; i < 4; i ++) {
		rowSum = 0;
		for (int j = 0; j < 4; j ++) {
			rowSum += P[i][j];
		}
		if (rowSum > 0) {
			for (int j = 0; j < 4; j ++) {
				 P[i][j] /= rowSum;
			}
		}		
	}

	// add 0.01 to off diagonal zero entries
	// add 0.97 to diagonal zero entries

	for (int i = 0; i < 4; i ++) {
		for (int j = 0; j < 4; j ++) {
			if (P[i][j] == 0) {
				if (i == j) {
					P[i][j] = 0.97;
				} else {
					P[i][j] = 0.01;
				}
			}
		}
	}

	for (int i = 0; i < 4; i ++) {
		for (int j = 0; j < 4; j ++) {
			rowSum = 0;
			for (int j = 0; j < 4; j ++) {
				rowSum += P[i][j];
			}
			for (int j = 0; j < 4; j ++) {
				P[i][j] /= rowSum;
			}
		}
	}

	return P;
}

emtr::Md SEM::ComputeTransitionMatrixUsingAncestralStates(SEM_vertex * p, SEM_vertex * c) {	
	emtr::Md P = emtr::Md{};			
	int dna_p; int dna_c;
	// cout << p->name << "\t" << c->name << endl;
	for (int site = 0; site < this->num_dna_patterns; site ++) {		
		dna_p = p->DNAcompressed[site];
		dna_c = c->DNAcompressed[site];
		if (dna_p < 4 and dna_c < 4) {
			P[dna_p][dna_c] += this->DNAPatternWeights[site];
		}
	}

	double rowSum;
	for (int i = 0; i < 4; i ++) {
		rowSum = 0;
		for (int j = 0; j < 4; j ++) {
			rowSum += P[i][j];
		}
		for (int j = 0; j < 4; j ++) {
			 P[i][j] /= rowSum;
		}
	}
	return P;
}

void SEM::SetPrefixForOutputFiles(string prefix_for_output_files_to_set){
	this->prefix_for_output_files = prefix_for_output_files_to_set;
}

void SEM::WriteRootedTreeInNewickFormat(string newickFileName) {
	vector <SEM_vertex *> verticesToVisit;
	SEM_vertex * c;
	SEM_vertex * p;	
	double edgeLength;	
	for (pair <int, SEM_vertex *> idAndVertex : * this->vertexMap) {
		idAndVertex.second->timesVisited = 0;
		if (idAndVertex.second->children.size() == 0) {
			idAndVertex.second->newickLabel = idAndVertex.second->name;
			verticesToVisit.push_back(idAndVertex.second);
		} else {
			idAndVertex.second->newickLabel = "";
		}
	}
	
	pair <SEM_vertex *, SEM_vertex * > vertexPair;
	unsigned int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		c = verticesToVisit[numberOfVerticesToVisit -1];
		verticesToVisit.pop_back();
		numberOfVerticesToVisit -= 1;
		if (c->parent != c) {
			p = c->parent;
			if (p->id < c->id) {
				vertexPair = make_pair(p,c);
			} else {
				vertexPair = make_pair(c,p);
			}
			p->timesVisited += 1;			
			if (this->edgeLengths.find(vertexPair) == this->edgeLengths.end()) {
				edgeLength = 0.1;
			} else {
				edgeLength = this->edgeLengths[vertexPair];
			}
			if (p->timesVisited == int(p->children.size())) {
				p->newickLabel += "," + c->newickLabel + ":" + to_string(edgeLength) + ")";
				verticesToVisit.push_back(p);
				numberOfVerticesToVisit += 1;
			} else if (p->timesVisited == 1) {
				p->newickLabel += "(" + c->newickLabel + ":" + to_string(edgeLength);
			} else {
				p->newickLabel += "," + c->newickLabel + ":" + to_string(edgeLength);
			}			
		}
	}
	ofstream newickFile;
	newickFile.open(newickFileName);
	newickFile << this->root->newickLabel << ";" << endl;
	newickFile.close();
}

void SEM::WriteCliqueTreeToFile(string cliqueTreeFileName) {
	ofstream cliqueTreeFile;
	cliqueTreeFile.open(cliqueTreeFileName);
	clique * parentClique;
	for (clique * childClique : this->cliqueT->cliques) {
		if (childClique->parent != childClique) {
			parentClique = childClique->parent;
			cliqueTreeFile << parentClique->x->name + "_" + parentClique->y->name +"\t";
			cliqueTreeFile << childClique->x->name + "_" + childClique->y->name << "\t";
			cliqueTreeFile << "0.01" << endl;
		}
	}
	cliqueTreeFile.close();
}

void SEM::WriteUnrootedTreeAsEdgeList(string fileName) {
	ofstream treeFile;
	treeFile.open(fileName);
	SEM_vertex * v;
	double t;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		for (SEM_vertex * n : v->neighbors) {
			if (v->id < n->id) {
				t = this->GetEdgeLength(v,n);
				treeFile << v->name << "\t" << n->name << "\t" << t << endl;
			}
		}
	}
	treeFile.close();
}


void SEM::WriteParametersOfBH(string fileName) {
	ofstream parameterFile;
	parameterFile.open(fileName);		
	
	parameterFile << "Root probability for vertex " << this->root->name << " is " << endl;
	for (int i = 0; i < 4; i++) {
		parameterFile << this->rootProbability[i] << "\t";
	}
	parameterFile << endl;


	SEM_vertex * c; SEM_vertex * p;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap){
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {								
			parameterFile << "Transition matrix for " << p->name << " to " << c->name << " is " << endl;
			string trans_par_string = "";
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					trans_par_string.append(to_string(c->transitionMatrix[i][j]) + " ");					
				}
			}
			if (!trans_par_string.empty() && trans_par_string.back() == ' ') trans_par_string.pop_back();    		
			parameterFile << trans_par_string << endl;
		}
	}	
	parameterFile.close();
}

void SEM::WriteRootedTreeAsEdgeList(string fileName) {
	ofstream treeFile;
	treeFile.open(fileName);
	double t;
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		if (v != v->parent) {
			t = this->GetEdgeLength(v,v->parent);
			treeFile << v->parent->name << "\t" << v->name << "\t" << t << endl;
		}
	}
	treeFile.close();
}

void SEM::RootTreeAtAVertexPickedAtRandom() {
	cout << "num of observed vertices is " << this->numberOfObservedVertices << endl;	
	int n = this->numberOfObservedVertices;
	uniform_int_distribution <int> distribution_v(n,(2*n-3));
	int v_ind = distribution_v(generator);
	cout << "index of vertex selected for rooting is " << v_ind << endl;
	cout << "number of vertices is " << this->vertexMap->size() << endl;
	SEM_vertex * v = (*this->vertexMap)[v_ind];
	cout << "vertex selected for rooting is " << v->id << endl;
	this->RootTreeAtVertex(v);
	
}

void SEM::RootTreeAlongAnEdgePickedAtRandom() {
	int n = this->numberOfObservedVertices;
//	int numOfVertices = this->vertexMap->size();
	uniform_int_distribution <int> distribution_v(0,(2*n-3));
	int v_ind = distribution_v(generator);
	SEM_vertex * v = (*this->vertexMap)[v_ind];	
	int numOfNeighbors = v->neighbors.size();
//	cout << "Number of neighbors of v are " << numOfNeighbors << endl;
	uniform_int_distribution <int> distribution_u(0,numOfNeighbors-1);	
	int u_ind_in_neighborList = distribution_u(generator);
	SEM_vertex * u = v->neighbors[u_ind_in_neighborList];
//	cout << "Rooting tree along edge ";
//	cout << u->name << "\t" << v->name << endl;
	this->RootTreeAlongEdge(u,v);
}


void SEM::RootTreeAlongEdge(SEM_vertex * u, SEM_vertex * v) {
	// Remove lengths of edges incident to root if necessary
	if (this->root->children.size() == 2) {
		SEM_vertex * c_l = this->root->children[0];
		SEM_vertex * c_r = this->root->children[1];
		this->edgeLengths.erase(make_pair(this->root,c_l));
		this->edgeLengths.erase(make_pair(this->root,c_r));
	}
	this->ClearDirectedEdges();
	SEM_vertex * c;
	this->root->AddChild(u);
	this->root->AddChild(v);
	
	SEM_vertex * c_l = this->root->children[0];
	SEM_vertex * c_r = this->root->children[1];
	this->edgeLengths.insert(make_pair(make_pair(this->root,c_l),0.001));
	this->edgeLengths.insert(make_pair(make_pair(this->root,c_r),0.001));	
	u->AddParent(this->root);
	v->AddParent(this->root);
	vector <SEM_vertex *> verticesToVisit;
	vector <SEM_vertex *> verticesVisited;
	verticesToVisit.push_back(u);
	verticesToVisit.push_back(v);
	verticesVisited.push_back(u);
	verticesVisited.push_back(v);
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {		
		c = verticesToVisit[numberOfVerticesToVisit - 1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(c);
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex * n: c->neighbors) {
			if (find(verticesVisited.begin(),verticesVisited.end(),n)==verticesVisited.end()) {
				verticesToVisit.push_back(n);
				numberOfVerticesToVisit += 1;
				c->AddChild(n);
				n->AddParent(c);
			}
		}
	}	
	this->SetLeaves();
//	cout << "Number of leaves is " << this->leaves.size() << endl;
	this->SetEdgesForPreOrderTraversal();
//	cout << "Number of edges for pre order traversal is " << this->edgesForPreOrderTreeTraversal.size() << endl;
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
//	cout << "Number of vertices for pre order traversal is " << this->preOrderVerticesWithoutLeaves.size() << endl;
	this->SetEdgesForPostOrderTraversal();
//	cout << "Number of edges for post order traversal is " << this->edgesForPostOrderTreeTraversal.size() << endl;
}

void SEM::InitializeTransitionMatricesAndRootProbability() {
	// ASR via MP
	// MLE of transition matrices and root probability
		
	vector <SEM_vertex *> verticesToVisit;

	SEM_vertex * p; int numberOfPossibleStates; int pos;
	map <SEM_vertex *, vector <int>> VU;
	map <SEM_vertex *, int> V;
	for (int site = 0; site < this->num_dna_patterns; site++) {
		VU.clear(); V.clear();
		// Set VU and V for leaves;
		for (SEM_vertex * v : this->leaves) {
			V[v] = v->DNArecoded[site];
			VU[v].push_back(v->DNArecoded[site]);
		}
		// Set VU for ancestors
		for (int v_ind = preOrderVerticesWithoutLeaves.size()-1; v_ind > -1; v_ind--) {
			p = this->preOrderVerticesWithoutLeaves[v_ind];
			map <int, int> dnaCount;
			for (int dna = 0; dna < 4; dna++) {
				dnaCount[dna] = 0;
			}
			for (SEM_vertex * c : p->children) {
				for (int dna: VU[c]) {
					dnaCount[dna] += 1;
				}
			}
			int maxCount = 0;
			for (pair<int, int> dnaCountPair: dnaCount) {
				if (dnaCountPair.second > maxCount) {
					maxCount = dnaCountPair.second;
				}
			}			
			for (pair<int, int> dnaCountPair: dnaCount) {
				if (dnaCountPair.second == maxCount) {
					VU[p].push_back(dnaCountPair.first);					
				}
			}			
		}
		// Set V for ancestors
		for (SEM_vertex * v : this->preOrderVerticesWithoutLeaves) {
			if (v->parent == v) {
			// Set V for root
				if (VU[v].size()==1) {
					V[v] = VU[v][0];
				} else {
					numberOfPossibleStates = VU[v].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V[v] = VU[v][pos];
				}				
			} else {
				p = v->parent;
				if (find(VU[v].begin(),VU[v].end(),V[p])==VU[v].end()){
					numberOfPossibleStates = VU[v].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V[v] = VU[v][pos];					
				} else {
					V[v] = V[p];
				}				
			}
			// Push states to compressedSequence
			v->DNArecoded.push_back(V[v]);
		}
	}
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

void SEM::AddToExpectedCountsForEachVariablePair() {
	SEM_vertex * u; SEM_vertex * v;
	double siteWeight = this->DNAPatternWeights[this->cliqueT->site];	
	pair <SEM_vertex *, SEM_vertex*> vertexPair;
	emtr::Md countMatrixPerSite;
	for (pair<int,SEM_vertex*> idPtrPair_1 : *this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex*> idPtrPair_2 : *this->vertexMap) {
			v = idPtrPair_2.second;			
			if (u->id < v->id) {
				if (!u->observed or !v->observed) {
					vertexPair = pair <SEM_vertex *, SEM_vertex *>(u,v);
					countMatrixPerSite = this->cliqueT->marginalizedProbabilitiesForVariablePair[vertexPair];
					for (int dna_u = 0; dna_u < 4; dna_u ++) {
						for (int dna_v = 0; dna_v < 4; dna_v ++) {
							this->expectedCountsForVertexPair[vertexPair][dna_u][dna_v] += countMatrixPerSite[dna_u][dna_v] * siteWeight;
						}
					}
				}
			}
		}
	}
}

void SEM::AddExpectedCountMatrices(map < pair <SEM_vertex * , SEM_vertex *>, emtr::Md > expectedCountsForVertexPairToAdd) {
	string u_name;
	string v_name;
	SEM_vertex * u;
	SEM_vertex * v;
	pair <SEM_vertex *, SEM_vertex *> edge;
	emtr::Md CountMatrix;
	for (pair <pair <SEM_vertex * , SEM_vertex *>, emtr::Md> mapElem: expectedCountsForVertexPairToAdd) {
		u_name = mapElem.first.first->name;
		v_name = mapElem.first.second->name;
		CountMatrix = mapElem.second;
		u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		if (u->id < v->id) {
			this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(u,v)] = CountMatrix;
		} else {			
			this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(v,u)] = emtr::MT(CountMatrix);

		}
	}	//this->expectedCountsForVertexPair
}


emtr::Md SEM::GetExpectedCountsForVariablePair(SEM_vertex * u, SEM_vertex * v) {
	emtr::Md C_pc;
	if (u->id < v->id) {
		C_pc = this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(u,v)];	
	} else {		
		C_pc = emtr::MT(this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(v,u)]);
	}
	return (C_pc);
}

emtr::Md SEM::GetPosteriorProbabilityForVariablePair(SEM_vertex * u, SEM_vertex * v) {
	emtr::Md P;
	if (u->id < v->id) {
		P = this->posteriorProbabilityForVertexPair[pair<SEM_vertex *, SEM_vertex *>(u,v)];
	} else {		
		P = emtr::MT(this->posteriorProbabilityForVertexPair[pair<SEM_vertex *, SEM_vertex *>(v,u)]);
	}
	return (P);
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

void SEM::AddToExpectedCounts() {
	SEM_vertex * u; SEM_vertex * v;
	double siteWeight = this->DNAPatternWeights[this->cliqueT->site];	
	// Add to counts for each unobserved vertex (C->x) where C is a clique
	array <double, 4> marginalizedProbability;
	vector <SEM_vertex *> vertexList;
	for (clique * C: this->cliqueT->cliques) {
		v = C->x;
		if(v->observed){
            throw mt_error("v should not be observed");
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
	// Add to counts for each vertex pair
	pair <SEM_vertex *, SEM_vertex*> vertexPair;
	emtr::Md countMatrixPerSite;
	for (pair<int,SEM_vertex*> idPtrPair_1 : *this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex*> idPtrPair_2 : *this->vertexMap) {
			v = idPtrPair_2.second;			
			if (u->id < v->id) {
				if (!u->observed or !v->observed) {
					vertexPair = pair <SEM_vertex *, SEM_vertex *>(u,v);
					countMatrixPerSite = this->cliqueT->marginalizedProbabilitiesForVariablePair[vertexPair];					
//					if (u->name == "l_1" or v->name == "l_1") {
//						cout << "Count matrix for " << u->name << ", " << v->name << " for site " << this->cliqueT->site << " is" << endl;
// 						cout << countMatrixPerSite << endl;
//					}
					for (int dna_u = 0; dna_u < 4; dna_u ++) {
						for (int dna_v = 0; dna_v < 4; dna_v ++) {
							this->expectedCountsForVertexPair[vertexPair][dna_u][dna_v] += countMatrixPerSite[dna_u][dna_v] * siteWeight;
						}
					}
				}
			}
		}
	}
}

emtr::Md SEM::GetObservedCounts(SEM_vertex * u, SEM_vertex * v) {	
	emtr::Md countMatrix = emtr::Md{};
	int dna_u; int dna_v;
	for (int i = 0; i < this->sequenceLength; i++) {
		dna_u = u->DNArecoded[i];
		dna_v = v->DNArecoded[i];
		countMatrix[dna_u][dna_v] += this->DNAPatternWeights[i];
	}
	return (countMatrix);
}

void SEM::ComputeExpectedCountsForTrifle(int layer) {
    // cout << "Initializing expected counts" << endl;
	// cout << "11a" << endl;
	this->InitializeExpectedCountsForEachVariable();
	// cout << "11b" << endl;
	this->InitializeExpectedCountsForEachEdge();
	// cout << "11c" << endl;
    //	this->ResetExpectedCounts();
    //	SEM_vertex * x; SEM_vertex * y;
	emtr::Md P_XY;
	//	int dna_x; int dna_y;
	bool debug = 0;
	if (debug) {
		cout << "Debug computing expected counts" << endl;
	}
	// Iterate over sites	
    // cout << "number of sites for layer " << layer << " is " << this->num_patterns_for_layer[layer] << endl;
	for (int site = 0; site < this->num_patterns_for_layer[layer]; site++) {
		// cout << "12a site" << site << " layer " << layer << " root " << this->root->name << endl;
		// cout << "computing expected counts for site " << site << endl;
		this->cliqueT->SetSite(site);		
		// cout << "12b" << endl;
		this->cliqueT->InitializePotentialAndBeliefs();	// set min root prob to 0.2 and min trans prob to 0.01
		// cout << "12c" << endl;
		this->cliqueT->CalibrateTree();
		// cout << "12d" << endl;
		this->cliqueT->SetMarginalProbabilitesForEachEdgeAsBelief(); // set min root prob to 0.2 and min trans prob to 0.01
		// cout << "12e" << endl;
		this->AddToExpectedCountsForEachVariable();
		// cout << "12f" << endl;
		this->AddToExpectedCountsForEachEdge();		
		// cout << "12g" << endl;
	}
	// cout << "11d" << endl;
}

// TODO: modify this to compute log-likelihood score using clique tree
void SEM::ComputeExpectedCounts() { 
    // cout << "Initializing expected counts" << endl;
	// cout << "11a" << endl;
	this->InitializeExpectedCountsForEachVariable();
	// cout << "11b" << endl;
	this->InitializeExpectedCountsForEachEdge();
	// cout << "11c" << endl;
    //	this->ResetExpectedCounts();
    //	SEM_vertex * x; SEM_vertex * y; 
	emtr::Md P_XY;
//	int dna_x; int dna_y;
	bool debug = 0;
	if (debug) {
		cout << "Debug computing expected counts" << endl;
	}
// Iterate over sites
	// parallelize here if needed
	for (int site = 0; site < this->num_dna_patterns; site++) {
		// cout << "12a" << endl;
		// cout << "computing expected counts for site " << site << endl;
		this->cliqueT->SetSite(site);		
		// cout << "12b" << endl;
		this->cliqueT->InitializePotentialAndBeliefs();
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

void SEM::ComputeMAPEstimateOfAncestralSequencesUsingCliques() {
	this->logLikelihood = 0;
	this->ResetAncestralSequences();
	this->ConstructCliqueTree();
	clique * rootClique = this->cliqueT->root;
	SEM_vertex * v;
	map <SEM_vertex *, int> verticesVisitedMap;
	array <double, 4> posteriorProbability;
	int maxProbState;
	double maxProb;
	for (int site = 0; site < this->num_dna_patterns; site++) {		
		this->cliqueT->SetSite(site);		
		this->cliqueT->InitializePotentialAndBeliefs();		
		this->cliqueT->CalibrateTree();
		this->logLikelihood += rootClique->logScalingFactorForClique * this->DNAPatternWeights[site];
		verticesVisitedMap.clear();
		for (clique * C: this->cliqueT->cliques) {
			v = C->x;
			if (verticesVisitedMap.find(v) == verticesVisitedMap.end()) {
				posteriorProbability = C->MarginalizeOverVariable(C->y);
				maxProb = -1; maxProbState = -1;
				for (int i = 0; i < 4; i ++) {
					if (posteriorProbability[i] > maxProb) {
						maxProb = posteriorProbability[i];
						maxProbState = i;
					}
				}
				if(maxProbState == -1) {
                    throw mt_error("Check prob assignment");
                }
				v->DNArecoded.push_back(maxProbState);
				verticesVisitedMap.insert(make_pair(v,v->id));
			}
		}
	}	
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

void SEM::InitializeExpectedCounts() {
	SEM_vertex * u; SEM_vertex * v;
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
						countMatrix[dna_u][dna_v] += this->DNAPatternWeights[site];
					}
				}
				this->expectedCountsForVertexPair.insert(make_pair(pair <SEM_vertex *, SEM_vertex *>(u,v), countMatrix));
			}
		}
	}
}

void SEM::ResetPointerToRoot() {
	//	Make sure that the pointer this->root stores the location
	//  of the vertex with in degree 0
	for (pair<int,SEM_vertex *> idPtrPair : *this->vertexMap) {
		if (idPtrPair.second->inDegree == 0){
			this->root = idPtrPair.second;
		}
	}
}

void SEM::AddArc(SEM_vertex * from, SEM_vertex * to) {
	to->AddParent(from);
	from->AddChild(to);
}

void SEM::RemoveArc(SEM_vertex * from, SEM_vertex * to) {	
	to->parent = to;
	to->inDegree -= 1;
	from->outDegree -= 1;	
	int ind = find(from->children.begin(),from->children.end(),to) - from->children.begin();
	from->children.erase(from->children.begin()+ind);	
}

void SEM::StoreBestProbability() {
	this->StoreRootAndRootProbability();
	this->StoreTransitionMatrices();
}

void SEM::StoreRootAndRootProbability() {
	this->root_stored = this->root;
	this->rootProbability_stored = this->rootProbability;
}

void SEM::StoreTransitionMatrices() {
	for (pair <int,SEM_vertex*> idPtrPair : * this->vertexMap) {
		idPtrPair.second->transitionMatrix_stored = idPtrPair.second->transitionMatrix;
	}
}

void SEM::RestoreBestProbability() {
	this->RestoreRootAndRootProbability();
	this->RestoreTransitionMatrices();
	this->RootTreeAtVertex(this->root);
}

void SEM::RestoreRootAndRootProbability() {
	this->rootProbability = this->rootProbability_stored;
	this->root = this->root_stored;
	this->root->rootProbability = this->rootProbability;	
}

void SEM::RestoreTransitionMatrices() {
	for (pair<int,SEM_vertex*> idPtrPair : * this->vertexMap) {
		idPtrPair.second->transitionMatrix = idPtrPair.second->transitionMatrix_stored;
	}
}

void SEM::StoreRateMatricesAndScalingFactors() {
	this->rateMatrixPerRateCategory_stored = this->rateMatrixPerRateCategory;
	this->scalingFactorPerRateCategory_stored = this->scalingFactorPerRateCategory;
}

void SEM::RestoreRateMatricesAndScalingFactors() {
	this->rateMatrixPerRateCategory = this->rateMatrixPerRateCategory_stored;
	this->scalingFactorPerRateCategory = this->scalingFactorPerRateCategory_stored;
}


void SEM::ComputeLogLikelihoodUsingExpectedDataCompletion() {
	this->logLikelihood = 0;
	array <double, 4> S_r = this->expectedCountsForVertex[this->root];
	for (int dna_r = 0; dna_r < 4; dna_r ++) {
		if (this->rootProbability[dna_r] > 0) {
			this->logLikelihood += S_r[dna_r] * log(this->rootProbability[dna_r]);
		}	
	}
	// Contribution of edges
	SEM_vertex * p; SEM_vertex * c;
	emtr::Md S_pc;
	emtr::Md P;
	for (pair <int, SEM_vertex *> idPtrPair : *this->vertexMap) {
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			P = c->transitionMatrix;
			if (p->id < c->id) {
				S_pc = this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(p,c)];	
			} else {				
				S_pc = emtr::MT(this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(c,p)]);
			}
			for (int dna_p = 0; dna_p < 4; dna_p ++) {
				for (int dna_c = 0; dna_c < 4; dna_c ++) {
					if(S_pc[dna_p][dna_c] < 0) {
						cout << "Expected counts for " << p->name << "\t" << c->name << " is " << endl;						
						throw mt_error("Check counts");
					}
					if (P[dna_p][dna_c] > 0) {
						this->logLikelihood += S_pc[dna_p][dna_c] * log(P[dna_p][dna_c]);
					}
				}
			}
		}
	}
}

// Case 1: Observed vertices may have out degree > 0
// Case 2: Root may have out degree = one
// Case 3: Directed tree (rooted) with vertices with outdegree 2 or 0.
// layers 0, 1 and 2
// coarse: 0, coarse: 1, coarse: 2 
void SEM::ComputeTrifleLogLikelihood(int layer) {
	this->logLikelihood = 0;
	map <SEM_vertex*,array<double,4>> conditionalLikelihoodMap;
	std::array <double,4> conditionalLikelihood;
	double partialLikelihood;
	double siteLikelihood;	
	double largestConditionalLikelihood = 0;
	double currentProb;			
	vector <SEM_vertex *> verticesToVisit;	
	SEM_vertex * p;
	SEM_vertex * c;
	emtr::Md P;
	// cout << "14a" << endl;
	for (int site = 0; site < this->num_patterns_for_layer[layer]; site++) {
		// cout << "14b" << endl;
    	conditionalLikelihoodMap.clear();
		// cout << "14c" << endl;
    	this->ResetLogScalingFactors();
		// cout << "14d" << endl;
		for (auto& edge : this->edgesForPostOrderTreeTraversal) {
			// cout << "15a" << endl;
			tie(p, c) = edge;
			P = c->transitionMatrix;			
			p->logScalingFactors += c->logScalingFactors;

			// Initialize leaf child
			if (c->outDegree == 0) {
				if (c->DNAcompressed[site] != 4) {
					conditionalLikelihood.fill(0.0);
					conditionalLikelihood[c->DNAcompressed[site]] = 1.0;
				} else {
					conditionalLikelihood.fill(1.0);
				}
				conditionalLikelihoodMap.insert({c, conditionalLikelihood});
			}
			// cout << "15b" << endl;

			// Initialize parent p if absent
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()) {
				if (p->id > this->numberOfObservedVertices - 1) {
					conditionalLikelihood.fill(1.0);      // latent
				} else {
					conditionalLikelihood.fill(0.0);      // observed
					conditionalLikelihood[p->DNAcompressed[site]] = 1.0;
				}
				conditionalLikelihoodMap.insert({p, conditionalLikelihood});
			}
			// cout << "15c" << endl;
			
			double largestConditionalLikelihood = 0.0;
			const auto& childCL = conditionalLikelihoodMap.at(c);
			// cout << "15d" << endl;
			for (int dna_p = 0; dna_p < 4; ++dna_p) {
				double partialLikelihood = 0.0;
				for (int dna_c = 0; dna_c < 4; ++dna_c) {
					partialLikelihood += P[dna_p][dna_c] * childCL[dna_c];
				}
				conditionalLikelihoodMap[p][dna_p] *= partialLikelihood;
				largestConditionalLikelihood = std::max(largestConditionalLikelihood, conditionalLikelihoodMap[p][dna_p]);
			}
			// cout << "15e" << endl;						
			if (largestConditionalLikelihood > 0.0) {
				for (int dna_p = 0; dna_p < 4; ++dna_p) {
					conditionalLikelihoodMap[p][dna_p] /= largestConditionalLikelihood;
				}
				p->logScalingFactors += log(largestConditionalLikelihood);
			} else {
				cout << "conditional likelihood is zero " << largestConditionalLikelihood << endl;
				cout << "p dna " << p->DNAcompressed[site] << endl;
				cout << "c dna " << c->DNAcompressed[site] << endl;
				for (int i = 0; i < 4; i ++) {
					for (int j = 0; j < 4; j ++) {
						cout << "c->transitionMatrix[" << i << "," << j << "] = " << c->transitionMatrix[i][j] << endl;
					}
				}
				throw mt_error("Largest conditional likelihood value is zero");
			}
			// cout << "15f" << endl;
		}
		// cout << "14e" << endl;
		double siteLikelihood = 0.0;
		const auto& rootCL = conditionalLikelihoodMap.at(this->root);
		for (int dna = 0; dna < 4; ++dna) {
			siteLikelihood += this->rootProbability[dna] * rootCL[dna];
		}
		// cout << "14f" << endl;
		if (siteLikelihood <= 0.0) throw mt_error("siteLikelihood <= 0");
		// cout << "14g" << endl;
		this->logLikelihood += (this->root->logScalingFactors + log(siteLikelihood)) * this->DNAPatternWeights[site];
		// cout << "14h" << endl;
	}
}

void SEM::StorePatterns(string pattern_file_name) {
	this->patterns.clear();
	

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

// Case 1: Observed vertices may have out degree > 0
// Case 2: Root may have out degree = one
// Case 3: Directed tree (rooted) with vertices with outdegree 2 or 0.
// Felsenstein's pruning algorithm
void SEM::ComputeLogLikelihood() {
	this->logLikelihood = 0;
	map <SEM_vertex*,array<double,4>> conditionalLikelihoodMap;
	std::array <double,4> conditionalLikelihood;
	double partialLikelihood;
	double siteLikelihood;	
	double largestConditionalLikelihood = 0;
	double currentProb;			
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

string SEM::EncodeAsDNA(vector<int> sequence){
	string allDNA = "AGTC";
	string dnaSequence = "";
	for (int s : sequence){
		dnaSequence += allDNA[s];
	}
	return dnaSequence;
}

void SEM::EM_DNA_rooted_at_each_internal_vertex_started_with_dirichlet_store_results(int num_repetitions) {
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
		
	this->max_log_likelihood_diri = -1 * pow(10,5);
	
	vector<int> vertex_indices_to_visit;
	vertex_indices_to_visit.reserve(std::max(0, num_vertices - n));
	
	for (int v_ind = n; v_ind < num_vertices; ++v_ind) vertex_indices_to_visit.push_back(v_ind);

	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	// cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);

    int num_vertices_to_visit = vertex_indices_to_visit.size();
	for (int v_i = 0; v_i < num_vertices_to_visit; v_i++) {
        const int v_ind = vertex_indices_to_visit[v_i];
		SEM_vertex * v = (*this->vertexMap)[v_ind];
		// cout << "node " << v_i+1 << " of " << num_vertices_to_visit << ":" << v->name << endl;
		if(v->degree != 3) throw mt_error("Expect internal nodes to have degree three");
		
		for (int rep = 0; rep < num_repetitions; rep++) {
			this->EM_current = EM_struct{};
			this->EM_current.method = "dirichlet";
			this->EM_current.root = v->name;
			this->EM_current.rep = rep + 1;			
			auto tup = this->EM_started_with_dirichlet_rooted_at(v);
			EM_struct EM_diri{this->EM_current};
			this->EM_DNA_runs_diri.push_back(EM_diri);
			const int    iter                    = std::get<0>(tup);
            const double logLikelihood_diri      = std::get<1>(tup);
            const double loglikelihood_ecd_first = std::get<2>(tup);
            const double loglikelihood_ecd_final = std::get<3>(tup);
            const double logLikelihood_final     = std::get<4>(tup);

			emtr::push_result(
                this->EMTR_results,                 // vector<tuple<string,string,int,int,double,double,double,double>>
                "dirichlet",                        // init_method
                v->name,                            // root
                rep + 1,                            // repetition (1-based)
                iter,
                logLikelihood_diri,                 // ll_initial
                loglikelihood_ecd_first,
                loglikelihood_ecd_final,
                logLikelihood_final
            );

			if (this->max_log_likelihood_diri < logLikelihood_final) {				
				this->max_log_likelihood_diri = logLikelihood_final;				
				if (this->max_log_likelihood_best < this->max_log_likelihood_diri) {
					this->max_log_likelihood_best = this->max_log_likelihood_diri;
					this->StoreRootAndRootProbability();
					this->StoreTransitionMatrices();
				}
			}
		}
	}		
	cout << "max log likelihood obtained using Dirichlet parameters is " << setprecision(10) << this->max_log_likelihood_diri << endl;
}

void SEM::EMtrifle_DNA_rooted_at_each_internal_vertex_started_with_dirichlet_store_results() {
  int n = this->numberOfObservedVertices;
  int num_vertices = this->vertexMap->size();
  this->max_log_likelihood_diri = -1 * std::pow(10.0, 5.0);

  std::vector<int> vertex_indices_to_visit;
  vertex_indices_to_visit.reserve(std::max(0, num_vertices - n));
  for (int v_ind = n; v_ind < num_vertices; ++v_ind) vertex_indices_to_visit.push_back(v_ind);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);

  const int nodeTotal = static_cast<int>(vertex_indices_to_visit.size());


  int v_ind;
  SEM_vertex* v;

  for (int rep = 0; rep < num_repetitions; rep++) {
    std::cout << "rep " << rep + 1 << std::endl;

    for (int v_i = 0; v_i < static_cast<int>(vertex_indices_to_visit.size()); v_i++) {
      v_ind = vertex_indices_to_visit[v_i];
      v = (*this->vertexMap)[v_ind];
      std::cout << "node " << v_i + 1 << " of " << nodeTotal << ":" << v->name << std::endl;

      if (v->degree != 3) throw mt_error("Expect internal nodes to have degree three");

      // this->EMTrifle_current = EMTrifle_struct{};
      // this->EMTrifle_current.method = "dirichlet";
      // this->EMTrifle_current.root = v->name;
      // this->EMTrifle_current.rep = rep + 1;

      for (int layer = 0; layer < 3; layer++) {
        this->EMTrifle_started_with_dirichlet_rooted_at(v, layer);

        double ll_val = this->EMTrifle_current.ll_final_trifle[layer]; // if available        
      }
	//   cout << "[EM_Trifle] " << this->emtrifle_to_json(this->EMTrifle_current) << endl;

      if (this->max_log_likelihood_diri < EMTrifle_current.ll_final_trifle[2]) {
        this->max_log_likelihood_diri = EMTrifle_current.ll_final_trifle[2];
        if (this->max_log_likelihood_best < this->max_log_likelihood_diri) {
          this->max_log_likelihood_best = this->max_log_likelihood_diri;
          this->StoreRootAndRootProbability();
          this->StoreTransitionMatrices();
        }
      }
      // this->EMTrifle_DNA_runs_diri.emplace_back(std::move(this->EMTrifle_current));
    }
  }
  std::cout << "max log likelihood obtained using Dirichlet parameters is "
            << std::setprecision(10) << this->max_log_likelihood_diri << std::endl;
}




void SEM::EM_rooted_at_each_internal_vertex_started_with_dirichlet(int num_repetitions) {
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
	SEM_vertex * v;
	vector <double> loglikelihoodscoresForEachRepetition;
	ofstream loglikelihood_node_rep_file;
	loglikelihood_node_rep_file.open(this->prefix_for_output_files + ".dirichlet_rooting_initial_final_rep_loglik");
    loglikelihood_node_rep_file << "root" << "\t"										
                                << "rep" << "\t"
                                << "iter" << "\t"
                                << "ll dirichlet" << "\t"
                                << "ecd-ll first" << "\t"
                                << "ecd-ll final" << "\t"
                                << "ll final" << endl;
	this->max_log_likelihood_diri = -1 * pow(10,5);
	double logLikelihood_pars;
	double loglikelihood_ecd_first;
	double loglikelihood_ecd_final;
	double logLikelihood_final;
	int iter;	
	tuple <int,double,double,double,double> iter_dirill_edllfirst_edllfinal_llfinal;
	vector<int> vertex_indices_to_visit;
	
	for (int v_ind = n; v_ind < num_vertices; v_ind++) {
		vertex_indices_to_visit.push_back(v_ind);
	}
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	// cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);
    int v_ind;	
	for (int v_i = 0; v_i < vertex_indices_to_visit.size(); v_i++){
        v_ind = vertex_indices_to_visit[v_i];
		v = (*this->vertexMap)[v_ind];
		cout << "node " << v_i+1 << ":" << v->name << endl;
		if(v->degree != 3){
            throw mt_error("Expect internal nodes to have degree three");
        }
		loglikelihoodscoresForEachRepetition.clear();
		for (int rep = 0; rep < num_repetitions; rep++) {					
			iter_dirill_edllfirst_edllfinal_llfinal = this->EM_started_with_dirichlet_rooted_at(v);			
			iter = get<0>(iter_dirill_edllfirst_edllfinal_llfinal);
			logLikelihood_pars = get<1>(iter_dirill_edllfirst_edllfinal_llfinal);
			loglikelihood_ecd_first = get<2>(iter_dirill_edllfirst_edllfinal_llfinal);
			loglikelihood_ecd_final = get<3>(iter_dirill_edllfirst_edllfinal_llfinal);
			logLikelihood_final = get<4>(iter_dirill_edllfirst_edllfinal_llfinal);
			loglikelihood_node_rep_file << v->name << "\t"										
										<< rep +1 << "\t"
										<< iter << "\t"
										<< setprecision(ll_precision) << logLikelihood_pars << "\t"
										<< setprecision(ll_precision) << loglikelihood_ecd_first << "\t"
										<< setprecision(ll_precision) << loglikelihood_ecd_final << "\t"
										<< setprecision(ll_precision) << logLikelihood_final << endl;
			if (this->max_log_likelihood_diri < logLikelihood_final) {							
				this->max_log_likelihood_diri = logLikelihood_final;
				if (this->max_log_likelihood_best < this->max_log_likelihood_diri) {
					this->max_log_likelihood_best = this->max_log_likelihood_diri;
					this->StoreRootAndRootProbability();
					this->StoreTransitionMatrices();
				}				
			}
		}
	}
	
	loglikelihood_node_rep_file.close();
	cout << "max log likelihood, precision 10, obtained using Dirichlet parameters is " << setprecision(10) << this->max_log_likelihood_diri << endl;	
}

void SEM::EMTrifle_DNA_for_replicate(int rep) {
  this->max_log_likelihood_rep = -1 * std::pow(10.0, 5.0);
  int n = this->numberOfObservedVertices;
  int num_vertices = this->vertexMap->size();  

  std::vector<int> vertex_indices_to_visit;
  vertex_indices_to_visit.reserve(std::max(0, num_vertices - n));
  string node_name;
  SEM_vertex * v;
  int v_ind;
  
  for (int i = 1; i < n-1; i++) {
	node_name = "h_" + to_string(i);	
	cout << node_name << endl;
	vertex_indices_to_visit.emplace_back(this->GetVertexId(node_name));
  }
  
    
  const int nodeTotal = static_cast<int>(vertex_indices_to_visit.size());

  // -------------------------------------------------------------------------
  
  for (int v_i = 0; v_i < nodeTotal; ++v_i) {	
    v_ind = vertex_indices_to_visit[v_i];
    v = (*this->vertexMap)[v_ind];
	// ---- Dirichlet (layers 0..2) ----   
	this->EMTrifle_current = EMTrifle_struct{};
	this->EMTrifle_current.method = "dirichlet";
	this->EMTrifle_current.root = v->name;
	this->EMTrifle_current.rep = rep;    
	for (int layer = 0; layer < 3; ++layer) {
      this->EMTrifle_started_with_dirichlet_rooted_at(v, layer);      
	  double ll_val = this->EMTrifle_current.ll_final_trifle[layer];
	  cout << "dirichlet" << "\t" << v_i + 1 << "\t" << layer << "\t" << v->name << "\t" << ll_val << endl;
    }
	// cout << this->emtrifle_to_json(this->EMTrifle_current) << endl;
	// cout << "[EM_Trifle] " << this->emtrifle_to_json(this->EMTrifle_current) << endl;
    
	if (this->max_log_likelihood_rep < EMTrifle_current.ll_final_trifle[2]) {
      	this->max_log_likelihood_rep = EMTrifle_current.ll_final_trifle[2];
      	if (this->max_log_likelihood_best < this->max_log_likelihood_rep) {
				this->max_log_likelihood_best = this->max_log_likelihood_rep;
				this->StoreRootAndRootProbability();
				this->StoreTransitionMatrices();
      		}
    	}
	}
	
	for (int v_i = 0; v_i < nodeTotal; ++v_i) {
		v_ind = vertex_indices_to_visit[v_i];
    	v = (*this->vertexMap)[v_ind];
		// ---- Parsimony (layers 0..2) ----
		this->EMTrifle_current = EMTrifle_struct{};
		this->EMTrifle_current.method = "parsimony";
		this->EMTrifle_current.root = v->name;
		this->EMTrifle_current.rep = rep;		
		for (int layer = 0; layer < 3; ++layer) {
			this->EMTrifle_started_with_parsimony_rooted_at(v, layer);
			double ll_val = this->EMTrifle_current.ll_final_trifle[layer];			
			cout << "parsimony" << "\t" << v_i + 1 << "\t" << layer << "\t" << v->name << "\t" << ll_val << endl;
			}		
		// cout << this->emtrifle_to_json(this->EMTrifle_current) << endl;
		// cout << "[EM_Trifle] " << this->emtrifle_to_json(this->EMTrifle_current) << endl;

		if (this->max_log_likelihood_rep < EMTrifle_current.ll_final_trifle[2]) {
			this->max_log_likelihood_rep = EMTrifle_current.ll_final_trifle[2];
			if (this->max_log_likelihood_best < this->max_log_likelihood_rep) {
				this->max_log_likelihood_best = this->max_log_likelihood_rep;
				this->StoreRootAndRootProbability();
				this->StoreTransitionMatrices();
			}
		}
	}
	
	cout << "Restoring BH model with best probability" << endl;
    this->RestoreBestProbability();	
    this->ReparameterizeBH();
	cout << "BH reparameterized" << endl;
		
	for (int v_i = 0; v_i < nodeTotal; ++v_i) {
		v_ind = vertex_indices_to_visit[v_i];
    	v = (*this->vertexMap)[v_ind];
		// ---- HSS (layers 0..2) ----
		this->EMTrifle_current = EMTrifle_struct{};
		this->EMTrifle_current.method = "hss";
		this->EMTrifle_current.root = v->name;
		this->EMTrifle_current.rep = rep;		
		for (int layer = 0; layer < 3; ++layer) {
			this->EMTrifle_started_with_HSS_rooted_at(v, layer);
			double ll_val = this->EMTrifle_current.ll_final_trifle[layer];
			cout << "hss" << "\t" << v_i + 1 << "\t" << layer << "\t" << v->name << "\t" << ll_val << endl;
		}		
		// cout << this->emtrifle_to_json(this->EMTrifle_current) << endl;
		// cout << "[EM_Trifle] " << this->emtrifle_to_json(this->EMTrifle_current) << endl;
	}
}


void SEM::EMTrifle_DNA_rooted_at_each_internal_vertex_started_with_parsimony_store_results() {
  int n = this->numberOfObservedVertices;
  int num_vertices = this->vertexMap->size();
  this->max_log_likelihood_pars = -1 * std::pow(10.0, 5.0);

  std::vector<int> vertex_indices_to_visit;
  vertex_indices_to_visit.reserve(std::max(0, num_vertices - n));
  for (int v_ind = n; v_ind < num_vertices; ++v_ind) vertex_indices_to_visit.push_back(v_ind);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);

  const int nodeTotal = static_cast<int>(vertex_indices_to_visit.size());

  int v_ind;
  SEM_vertex* v;

  for (int rep = 0; rep < num_repetitions; rep++) {
    std::cout << "rep " << rep + 1 << std::endl;

    for (int v_i = 0; v_i < static_cast<int>(vertex_indices_to_visit.size()); v_i++) {
      v_ind = vertex_indices_to_visit[v_i];
      v = (*this->vertexMap)[v_ind];
      std::cout << "node " << v_i + 1 << " of " << nodeTotal << ":" << v->name << std::endl;

      if (v->degree != 3) throw mt_error("Expect internal nodes to have degree three");

      for (int layer = 0; layer < 3; layer++) {
        this->EMTrifle_started_with_parsimony_rooted_at(v, layer);

        
        double ll_val = this->EMTrifle_current.ll_final_trifle[layer];
      }

      if (this->max_log_likelihood_pars < EMTrifle_current.ll_final_trifle[2]) {
        this->max_log_likelihood_pars = EMTrifle_current.ll_final_trifle[2];
        if (this->max_log_likelihood_best < this->max_log_likelihood_pars) {
          this->max_log_likelihood_best = this->max_log_likelihood_pars;
          this->StoreRootAndRootProbability();
          this->StoreTransitionMatrices();
        }
      }      
    }
  }
  std::cout << "max log likelihood obtained using Parsimony parameters is "
            << std::setprecision(10) << this->max_log_likelihood_pars << std::endl;
}



void SEM::EM_DNA_rooted_at_each_internal_vertex_started_with_parsimony_store_results(int num_repetitions) {
	int n = this->numberOfObservedVertices;	
	int num_vertices = this->vertexMap->size();	
	this->max_log_likelihood_pars = -1 * pow(10,5);
	
	vector<int> vertex_indices_to_visit;
	vertex_indices_to_visit.reserve(std::max(0, num_vertices - n));
	
	for (int v_ind = n; v_ind < num_vertices; ++v_ind) vertex_indices_to_visit.push_back(v_ind);

	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	int num_vertices_to_visit = vertex_indices_to_visit.size();
	cout << "randomizing the order in which " << num_vertices_to_visit << " nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);
    
	for (int v_i = 0; v_i < vertex_indices_to_visit.size(); v_i++) {
		const int v_ind = vertex_indices_to_visit[v_i];
		SEM_vertex * v = (*this->vertexMap)[v_ind];
		cout << "node " << v_i+1 << " of " << num_vertices_to_visit << ":" << v->name << endl;				
		if (v->degree != 3) throw mt_error("Expect internal nodes to have degree three");
		// cout << "here1" << endl;
		for (int rep = 0; rep < num_repetitions; rep++) {
			// cout << " size of EM_DNA_runs_pars is " << this->EM_DNA_runs_pars.size() << endl;
			this->EM_current = EM_struct{};
			this->EM_current.method = "parsimony";
			this->EM_current.root = v->name;
			this->EM_current.rep = rep + 1;
			auto tup = this->EM_started_with_parsimony_rooted_at(v);			
			// cout << string_EM_pars << endl;
			EM_struct EM_pars = this->EM_current;
			this->EM_DNA_runs_pars.push_back(EM_pars);			
			const int    iter                    = std::get<0>(tup);
            const double logLikelihood_pars      = std::get<1>(tup);
            const double loglikelihood_ecd_first = std::get<2>(tup);
            const double loglikelihood_ecd_final = std::get<3>(tup);
            const double logLikelihood_final     = std::get<4>(tup);
			// cout << "initial " << logLikelihood_pars << "iter " << iter << endl;
			// cout << "final " << logLikelihood_final << "iter " << iter << endl;

			emtr::push_result(
                this->EMTR_results,                 // vector<tuple<string,string,int,int,double,double,double,double>>
                "parsimony",                        // init_method
                v->name,                            // root
                rep + 1,                            // repetition (1-based)
                iter,
                logLikelihood_pars,                 // ll_initial
                loglikelihood_ecd_first,
                loglikelihood_ecd_final,
                logLikelihood_final
            );

			if (this->max_log_likelihood_pars < logLikelihood_final) {				
				this->max_log_likelihood_pars = logLikelihood_final;				
				if (this->max_log_likelihood_best < this->max_log_likelihood_pars) {
					this->max_log_likelihood_best = this->max_log_likelihood_pars;
					this->StoreRootAndRootProbability();
					this->StoreTransitionMatrices();
				}
			}
		}
	}
		
	cout << "max log likelihood obtained using Parsimony parameters is " << setprecision(10) << this->max_log_likelihood_pars << endl;	
}


void SEM::EM_rooted_at_each_internal_vertex_started_with_parsimony(int num_repetitions) {
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
	SEM_vertex * v;
	vector <double> loglikelihoodscoresForEachRepetition;
	ofstream loglikelihood_node_rep_file;
	loglikelihood_node_rep_file.open(this->prefix_for_output_files + ".pars_rooting_initial_final_rep_loglik");
    loglikelihood_node_rep_file << "root" << "\t"										
                                << "rep" << "\t"
                                << "iter" << "\t"
                                << "ll pars" << "\t"
                                << "ecd-ll first" << "\t"
                                << "ecd-ll final" << "\t"
                                << "ll final" << endl;
	this->max_log_likelihood_pars = -1 * pow(10,5);
	double logLikelihood_pars;
	double loglikelihood_ecd_first;
	double loglikelihood_ecd_final;
	double logLikelihood_final;
	int iter;	
	tuple <int,double,double,double,double> iter_parsll_edllfirst_edllfinal_llfinal;
	vector<int> vertex_indices_to_visit;
	
	for (int v_ind = n; v_ind < num_vertices; v_ind++) {
		vertex_indices_to_visit.push_back(v_ind);
	}
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);
    int v_ind;
	for (int v_i = 0; v_i < vertex_indices_to_visit.size(); v_i++){
        v_ind = vertex_indices_to_visit[v_i];
		v = (*this->vertexMap)[v_ind];
		cout << "node " << v_i+1 << ":" << v->name << endl;
		if(v->degree != 3){
            throw mt_error("Expect internal nodes to have degree three");
        }
		loglikelihoodscoresForEachRepetition.clear();
		for (int rep = 0; rep < num_repetitions; rep++) {				
			iter_parsll_edllfirst_edllfinal_llfinal = this->EM_started_with_parsimony_rooted_at(v);			
			iter = get<0>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_pars = get<1>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_ecd_first = get<2>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_ecd_final = get<3>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_final = get<4>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_node_rep_file << v->name << "\t"										
										<< rep +1 << "\t"
										<< iter << "\t"
										<< setprecision(ll_precision) << logLikelihood_pars << "\t"
										<< setprecision(ll_precision) << loglikelihood_ecd_first << "\t"
										<< setprecision(ll_precision) << loglikelihood_ecd_final << "\t"
										<< setprecision(ll_precision) << logLikelihood_final << endl;
			if (this->max_log_likelihood_pars < logLikelihood_final) {				
				this->max_log_likelihood_pars = logLikelihood_final;
				if (this->max_log_likelihood_best < this->max_log_likelihood_pars) {
					this->max_log_likelihood_best = this->max_log_likelihood_pars;
					this->StoreRootAndRootProbability();
					this->StoreTransitionMatrices();
				}
			}
		}
	}
	
	loglikelihood_node_rep_file.close();	
	cout << "max log likelihood obtained using Parsimony parameters is " << setprecision(10) << this->max_log_likelihood_pars << endl;
	// (*this->logFile) << "max log likelihood obtained using Parsimony parameters is " << setprecision(ll_precision) << max_log_likelihood << endl;	
}


void SEM::EM_DNA_rooted_at_each_internal_vertex_started_with_HSS_store_results(int num_repetitions) {
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
		
	this->max_log_likelihood_hss = -1 * pow(10,5);
	
	vector<int> vertex_indices_to_visit;
	vertex_indices_to_visit.reserve(std::max(0, num_vertices - n));
	
	for (int v_ind = n; v_ind < num_vertices; ++v_ind) vertex_indices_to_visit.push_back(v_ind);

	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);

    int num_vertices_to_visit = vertex_indices_to_visit.size();
	for (int v_i = 0; v_i < vertex_indices_to_visit.size(); v_i++){
        const int v_ind = vertex_indices_to_visit[v_i];
		SEM_vertex * v = (*this->vertexMap)[v_ind];
		cout << "node " << v_i+1 << " of " << num_vertices_to_visit << ":" << v->name << endl;
		if(v->degree != 3) throw mt_error("Expect internal nodes to have degree three");
		
		for (int rep = 0; rep < num_repetitions; rep++) {
			this->EM_current = EM_struct{};
			this->EM_current.method = "hss";
			this->EM_current.root = v->name;
			this->EM_current.rep = rep + 1;
			auto tup = this->EM_started_with_HSS_parameters_rooted_at(v);
			EM_struct EM_hss{this->EM_current};
			this->EM_DNA_runs_hss.push_back(EM_hss);
			const int    iter                    = std::get<0>(tup);
            const double logLikelihood_hss       = std::get<1>(tup);
            const double loglikelihood_ecd_first = std::get<2>(tup);
            const double loglikelihood_ecd_final = std::get<3>(tup);
            const double logLikelihood_final     = std::get<4>(tup);

			emtr::push_result(
                this->EMTR_results,
                "hss",
                v->name,
                rep + 1,
                iter,
                logLikelihood_hss,
                loglikelihood_ecd_first,
                loglikelihood_ecd_final,
                logLikelihood_final
            );


			if (this->max_log_likelihood_hss < logLikelihood_final) {				
				this->max_log_likelihood_hss = logLikelihood_final;				
				if (this->max_log_likelihood_best < this->max_log_likelihood_hss) {
					this->max_log_likelihood_best = this->max_log_likelihood_hss;
					this->StoreRootAndRootProbability();
					this->StoreTransitionMatrices();
				}
			}
		}
	}
		
	cout << "max log likelihood obtained using HSS parameters is " << setprecision(10) << this->max_log_likelihood_hss << endl;	
}

void SEM::EMtrifle_DNA_rooted_at_each_internal_vertex_started_with_HSS_store_results() {
  int n = this->numberOfObservedVertices;
  int num_vertices = this->vertexMap->size();
  this->max_log_likelihood_hss = -1 * std::pow(10.0, 5.0);

  std::vector<int> vertex_indices_to_visit;
  vertex_indices_to_visit.reserve(std::max(0, num_vertices - n));
  for (int v_ind = n; v_ind < num_vertices; ++v_ind) vertex_indices_to_visit.push_back(v_ind);

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);

  const int nodeTotal = static_cast<int>(vertex_indices_to_visit.size());

  int v_ind;
  SEM_vertex* v;

  for (int rep = 0; rep < num_repetitions; rep++) {
    std::cout << "rep " << rep + 1 << std::endl;

    for (int v_i = 0; v_i < static_cast<int>(vertex_indices_to_visit.size()); v_i++) {
      v_ind = vertex_indices_to_visit[v_i];
      v = (*this->vertexMap)[v_ind];
      std::cout << "node " << v_i + 1 << " of " << nodeTotal << ":" << v->name << std::endl;

      if (v->degree != 3) throw mt_error("Expect internal nodes to have degree three");

      // this->EMTrifle_current = EMTrifle_struct{};
      // this->EMTrifle_current.method = "hss";
      // this->EMTrifle_current.root = v->name;
      // this->EMTrifle_current.rep = rep + 1;

      for (int layer = 0; layer < 3; layer++) {
        this->EMTrifle_started_with_HSS_rooted_at(v, layer);

        double ll_val = this->EMTrifle_current.ll_final_trifle[layer]; // if available
      }

      if (this->max_log_likelihood_hss < this->EMTrifle_current.ll_final_trifle[2]) {
        this->max_log_likelihood_hss = this->EMTrifle_current.ll_final_trifle[2];
        if (this->max_log_likelihood_best < this->max_log_likelihood_hss) {
          this->max_log_likelihood_best = this->max_log_likelihood_hss;
          this->StoreRootAndRootProbability();
          this->StoreTransitionMatrices();
        }
      }
      // this->EMTrifle_DNA_runs_hss.emplace_back(std::move(this->EMTrifle_current));
    }
  }
  std::cout << "max log likelihood obtained using HSS is "
            << std::setprecision(10) << this->max_log_likelihood_hss << std::endl;
}


void SEM::EM_rooted_at_each_internal_vertex_started_with_HSS_par(int num_repetitions) {
	// cout << "convergence threshold for EM is " << this->logLikelihoodConvergenceThreshold << endl;
	// (* this->logFile) << "convergence threshold for EM is " << this->logLikelihoodConvergenceThreshold << endl;
	// cout << "maximum number of EM iterations allowed is " << this->maxIter << endl;
	// (* this->logFile) << "maximum number of EM iterations allowed is " << this->maxIter << endl;
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
	SEM_vertex * v;
	vector <double> loglikelihoodscoresForEachRepetition;
	ofstream loglikelihood_node_rep_file;
	loglikelihood_node_rep_file.open(this->prefix_for_output_files + ".HSS_rooting_initial_final_rep_loglik");
    loglikelihood_node_rep_file << "root" << "\t"
                                << "rep" << "\t"
                                << "iter" << "\t"
                                << "ll HSS" << "\t"
                                << "ecd ll first" << "\t"
                                << "ecd ll final" << "\t"
                                << "ll final" << endl;
	double max_log_likelihood = -1 * pow(10,5);	
	double logLikelihood_pars;
	double loglikelihood_ecd_first;
	double loglikelihood_ecd_final;
	double logLikelihood_final;
	int iter;	
	tuple <int,double,double,double,double> iter_parsll_edllfirst_edllfinal_llfinal;
	vector<int> vertex_indices_to_visit;
	
	for (int v_ind = n; v_ind < num_vertices; v_ind++) {
		vertex_indices_to_visit.push_back(v_ind);
	}
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);
	int v_ind;
	for (int v_i = 0; v_i < vertex_indices_to_visit.size(); v_i++){
        v_ind = vertex_indices_to_visit[v_i];
		v = (*this->vertexMap)[v_ind];
		cout << "node " << v_i+1 << ":" << v->name << endl;		
		if(v->degree != 3){
            throw mt_error("Expect internal nodes to have degree three");
        }
		loglikelihoodscoresForEachRepetition.clear();
		for (int rep = 0; rep < num_repetitions; rep++) {					
			iter_parsll_edllfirst_edllfinal_llfinal = this->EM_started_with_HSS_parameters_rooted_at(v);
			iter = get<0>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_pars = get<1>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_ecd_first = get<2>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_ecd_final = get<3>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_final = get<4>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_node_rep_file << v->name << "\t"										
										<< rep +1 << "\t"
										<< iter << "\t"
										<< setprecision(ll_precision) << logLikelihood_pars << "\t"
										<< setprecision(ll_precision) << loglikelihood_ecd_first << "\t"
										<< setprecision(ll_precision) << loglikelihood_ecd_final << "\t"
										<< setprecision(ll_precision) << logLikelihood_final << endl;
			if (max_log_likelihood < logLikelihood_final) {				
				// this->WriteProbabilities();
				max_log_likelihood = logLikelihood_final;
			}
		}
	}
	loglikelihood_node_rep_file.close();
	cout << "max log likelihood obtained using HSS parameters is " << setprecision(10) << max_log_likelihood << endl;
	// (*this->logFile) << "max log likelihood obtained using HSS parameters is " << setprecision(ll_precision) << max_log_likelihood << endl;	
}

void SEM::EMTrifle_started_with_dirichlet_rooted_at(SEM_vertex *v, int layer) {
	// iterate over each internal node		
	this->RootTreeAtVertex(v);
	if (layer == 0) {		
		this->SetInitialEstimateOfModelParametersUsingDirichlet();		
	}
	this->BumpZeroEntriesOfModelParameters();
	this->StoreInitialParamsInEMTrifleCurrent(layer);
	// cout << "10d" << endl;
	this->ComputeTrifleLogLikelihood(layer);
	// cout << "10e" << endl;		
	this->EMTrifle_current.ll_initial_trifle[layer] = this->logLikelihood;

	cout << "initial log likelihood is " << this->EMTrifle_current.ll_initial_trifle[layer] << endl;
	
	double logLikelihood_exp_data_previous;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;		
	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	logLikelihood_exp_data_previous = -1 * pow(10,4);
	this->ConstructCliqueTree();
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		// cout << "iteration " << iter << endl;			
		// cout << "here 2" << endl;
		this->ComputeExpectedCountsForTrifle(layer);
		// cout << "here 3" << endl;
		this->ComputeMarginalProbabilitiesUsingExpectedCounts();
		// cout << "here 4" << endl;
		this->ComputeMLEstimateOfBHGivenExpectedDataCompletion();
		// cout << "here 5" << endl;
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		// cout << "here 6" << endl;
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;		
		if (iter == 1) {			
			logLikelihood_exp_data_first = this->logLikelihood;
            logLikelihood_exp_data_previous = this->logLikelihood;
			this->EMTrifle_current.ecd_ll_per_iter_for_trifle[layer][iter] = this->logLikelihood;
		} else if ((this->logLikelihood > logLikelihood_exp_data_previous + this->ecdllConvergenceThresholdForTrifle[layer]) and (iter < this->maxIter)) {
			logLikelihood_exp_data_previous = this->logLikelihood;
			this->EMTrifle_current.ecd_ll_per_iter_for_trifle[layer][iter] = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_previous;
		}
	}
	this->ComputeTrifleLogLikelihood(layer);
	this->StoreFinalParamsInEMTrifleCurrent(layer);	
	this->EMTrifle_current.iter_trifle[layer] = iter;
	this->EMTrifle_current.ll_final_trifle[layer] = this->logLikelihood;	
	cout << "final log likelihood is " << this->EMTrifle_current.ll_final_trifle[layer] << endl;
	return ;
	// return tuple<int,double,double,double,double>(iter,logLikelihood_pars,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
}


void SEM::EMTrifle_started_with_HSS_rooted_at(SEM_vertex *v, int layer) {	
	// iterate over each internal node	
	this->RootTreeAtVertex(v);
	if (layer == 0) {
		this->SetInitialEstimateOfModelParametersUsingHSS();		
	}
	this->BumpZeroEntriesOfModelParameters();
	this->StoreInitialParamsInEMTrifleCurrent(layer);
	// cout << "10d" << endl;
	this->ComputeTrifleLogLikelihood(layer);
	// cout << "10e" << endl;		
	this->EMTrifle_current.ll_initial_trifle[layer] = this->logLikelihood;
	cout << "initial log likelihood is " << this->EMTrifle_current.ll_initial_trifle[layer] << endl;
	
	// this->StoreParamsInEMCurrent("init");
	// cout << "10d" << endl;
	this->ResetAncestralSequences();
	
	double logLikelihood_exp_data_previous;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	
	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// cout << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	logLikelihood_exp_data_previous = -1 * pow(10,4);
	this->ConstructCliqueTree();
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		// cout << "iteration " << iter << endl;			
		// cout << "here 2" << endl;
		this->ComputeExpectedCountsForTrifle(layer);
		// cout << "here 3" << endl;
		this->ComputeMarginalProbabilitiesUsingExpectedCounts();
		// cout << "here 4" << endl;
		this->ComputeMLEstimateOfBHGivenExpectedDataCompletion();
		// cout << "here 5" << endl;
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		// cout << "here 6" << endl;
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;		
		if (iter == 1) {			
			logLikelihood_exp_data_first = this->logLikelihood;
            logLikelihood_exp_data_previous = this->logLikelihood;
			this->EMTrifle_current.ecd_ll_per_iter_for_trifle[layer][iter] = this->logLikelihood;
		} else if ((this->logLikelihood > logLikelihood_exp_data_previous + this->ecdllConvergenceThresholdForTrifle[layer]) and (iter < this->maxIter)) {
			logLikelihood_exp_data_previous = this->logLikelihood;
			this->EMTrifle_current.ecd_ll_per_iter_for_trifle[layer][iter] = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_previous;
		}
	}
	this->ComputeTrifleLogLikelihood(layer);
	this->StoreFinalParamsInEMTrifleCurrent(layer);	
	this->EMTrifle_current.iter_trifle[layer] = iter;
	this->EMTrifle_current.ll_final_trifle[layer] = this->logLikelihood;	
	cout << "final log likelihood is " << this->EMTrifle_current.ll_final_trifle[layer] << endl;
	return ;
}

void SEM::EMTrifle_started_with_parsimony_rooted_at(SEM_vertex *v, int layer) {
	// iterate over each internal node	
	this->RootTreeAtVertex(v);
	if (layer == 0) {		
		this->ComputeMPEstimateOfAncestralSequencesForTrifle(layer);
		this->ComputeInitialEstimateOfModelParametersForTrifle(layer);
	}
	this->BumpZeroEntriesOfModelParameters();
	this->StoreInitialParamsInEMTrifleCurrent(layer);	
	// cout << "10d" << endl;
	this->ComputeTrifleLogLikelihood(layer);
	// cout << "10e" << endl;		
	this->EMTrifle_current.ll_initial_trifle[layer] = this->logLikelihood;
	cout << "initial log likelihood is " << this->EMTrifle_current.ll_initial_trifle[layer] << endl;
	
	// this->StoreParamsInEMCurrent("init");
	// cout << "10d" << endl;
	this->ResetAncestralSequences();
	
	double logLikelihood_exp_data_previous;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	
	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// cout << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	logLikelihood_exp_data_previous = -1 * pow(10,4);
	this->ConstructCliqueTree();
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		// cout << "iteration " << iter << endl;			
		// cout << "here 2" << endl;
		this->ComputeExpectedCountsForTrifle(layer);
		// cout << "here 3" << endl;
		this->ComputeMarginalProbabilitiesUsingExpectedCounts();
		// cout << "here 4" << endl;
		this->ComputeMLEstimateOfBHGivenExpectedDataCompletion();
		// cout << "here 5" << endl;
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		// cout << "here 6" << endl;
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;		
		if (iter == 1) {			
			logLikelihood_exp_data_first = this->logLikelihood;
            logLikelihood_exp_data_previous = this->logLikelihood;
			this->EMTrifle_current.ecd_ll_per_iter_for_trifle[layer][iter] = this->logLikelihood;
		} else if ((this->logLikelihood > logLikelihood_exp_data_previous + this->ecdllConvergenceThresholdForTrifle[layer]) and (iter < this->maxIter)) {
			logLikelihood_exp_data_previous = this->logLikelihood;
			this->EMTrifle_current.ecd_ll_per_iter_for_trifle[layer][iter] = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_previous;
		}
	}
	this->ComputeTrifleLogLikelihood(layer);
	this->StoreFinalParamsInEMTrifleCurrent(layer);	
	this->EMTrifle_current.iter_trifle[layer] = iter;
	this->EMTrifle_current.ll_final_trifle[layer] = this->logLikelihood;	
	cout << "final log likelihood is " << this->EMTrifle_current.ll_final_trifle[layer] << endl;
	return ;
	// return tuple<int,double,double,double,double>(iter,logLikelihood_pars,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
}

tuple <int,double,double,double,double> SEM::EM_started_with_parsimony_rooted_at(SEM_vertex *v) {
	//	cout << "10a" << endl;
	// iterate over each internal node	
	this->RootTreeAtVertex(v);
	// cout << "10b" << endl;
	this->ComputeMPEstimateOfAncestralSequences();	
	// cout << "10c" << endl;
	this->ComputeInitialEstimateOfModelParameters();
	this->StoreParamsInEMCurrent("init");
	this->ComputeLogLikelihood();
	// cout << "Initial value of log-likelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "10d" << endl;
	this->ResetAncestralSequences();
	double logLikelihood_pars;
	double logLikelihood_exp_data_previous;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	logLikelihood_pars = this->logLikelihood;	
	this->EM_current.ll_init = logLikelihood_pars;
	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// cout << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	logLikelihood_exp_data_previous = -1 * pow(10,4);
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;					
		this->ConstructCliqueTree();			
		// 2. Compute expected counts
		this->ComputeExpectedCounts();

		this->ComputeMarginalProbabilitiesUsingExpectedCounts();
		
		this->ComputeMLEstimateOfBHGivenExpectedDataCompletion();	
		
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();		
		
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;		
		if (iter == 1){			
			logLikelihood_exp_data_first = this->logLikelihood;
            logLikelihood_exp_data_previous = this->logLikelihood;
			this->EM_current.ecd_ll_per_iter[iter] = this->logLikelihood;
		} else if ((this->logLikelihood > logLikelihood_exp_data_previous + this->ecdllConvergenceThreshold) and (iter < this->maxIter)) {
			logLikelihood_exp_data_previous = this->logLikelihood;
			this->EM_current.ecd_ll_per_iter[iter] = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_previous;
		}
	}
	this->ComputeLogLikelihood();
	this->StoreParamsInEMCurrent("final");
	this->EM_current.iter = iter;
	this->EM_current.ll_final = this->logLikelihood;
	
	return tuple<int,double,double,double,double>(iter,logLikelihood_pars,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
}

/*
1. Number of iterations until EM converges
2. Initial log-likelihood with HSS based parameters
3. Expected-data log-likelihood after one EM iteration
4. Expected-data log-likelihood after last EM iteration
5. Final log-likelihood using EM based parameters
*/

tuple <int,double,double,double,double> SEM::EM_started_with_HSS_parameters_rooted_at(SEM_vertex *v) {
	//	cout << "10a" << endl;
	// iterate over each internal node	
	this->RootTreeAtVertex(v);
	// cout << "10b" << endl;	
	this->SetInitialEstimateOfModelParametersUsingHSS();
	this->StoreParamsInEMCurrent("init");
	this->ComputeLogLikelihood();	
	// cout << "10d" << endl;
	this->ResetAncestralSequences();
	double logLikelihood_hss;
	double logLikelihood_exp_data_previous;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	logLikelihood_hss = this->logLikelihood;
	this->EM_current.ll_init = logLikelihood_hss;
	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// cout << "log-likelihood computed by marginalization using HSS parameters is " << setprecision(ll_precision) << logLikelihood_hss << endl;
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using HSS parameters is " << setprecision(ll_precision) << logLikelihood_hss << endl;
	logLikelihood_exp_data_previous = -1 * pow(10,4);
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;		
		this->ConstructCliqueTree();			
		// 2. Compute expected counts
		this->ComputeExpectedCounts();

		this->ComputeMarginalProbabilitiesUsingExpectedCounts();

		this->ComputeMLEstimateOfBHGivenExpectedDataCompletion();
		
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;		
		if (iter == 1) {
			logLikelihood_exp_data_previous = this->logLikelihood;
			logLikelihood_exp_data_first = this->logLikelihood;
			this->EM_current.ecd_ll_per_iter[iter] = this->logLikelihood;
		} else if ((this->logLikelihood > logLikelihood_exp_data_previous + this->ecdllConvergenceThreshold) and (iter < this->maxIter)) {
			logLikelihood_exp_data_previous = this->logLikelihood;
			this->EM_current.ecd_ll_per_iter[iter] = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_previous;
		}
	}
	this->ComputeLogLikelihood();
	this->StoreParamsInEMCurrent("final");
	this->EM_current.iter = iter;
	this->EM_current.ll_final = this->logLikelihood;
	// cout << "log-likelihood computed by marginalization after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// (*this->logFile) << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	
	
	return tuple<int,double,double,double,double>(iter,logLikelihood_hss,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
}


/*
1. Number of iterations until EM converges
2. Initial log-likelihood with parsimony based parameters
3. Expected-data log-likelihood after one EM iteration
4. Expected-data log-likelihood after last EM iteration
5. Final log-likelihood using EM based parameters
*/

tuple <int,double,double,double,double> SEM::EM_started_with_dirichlet_rooted_at(SEM_vertex *v) {
	//	cout << "10a" << endl;
	// iterate over each internal node	
	this->RootTreeAtVertex(v); // take outside
	// cout << "10c" << endl;
	this->SetInitialEstimateOfModelParametersUsingDirichlet();
	this->StoreParamsInEMCurrent("init"); // take outside
	this->ComputeLogLikelihood();
	// cout << "Initial value of log-likelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "10d" << endl;
	this->ResetAncestralSequences();
	double logLikelihood_diri;
	double logLikelihood_exp_data_current;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	logLikelihood_diri = this->logLikelihood;
	this->EM_current.ll_init = logLikelihood_diri;	
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_diri << endl;
	logLikelihood_exp_data_current = -1 * pow(10,4);
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;		
		this->ConstructCliqueTree();			
		// 2. Compute expected counts
		this->ComputeExpectedCounts();

		this->ComputeMarginalProbabilitiesUsingExpectedCounts();
		
		this->ComputeMLEstimateOfBHGivenExpectedDataCompletion();
		
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;		
		if (iter == 1){
			logLikelihood_exp_data_first = this->logLikelihood;
			logLikelihood_exp_data_current = this->logLikelihood;			
			this->EM_current.ecd_ll_per_iter[iter] = this->logLikelihood;
		} else if ((this->logLikelihood > logLikelihood_exp_data_current + this->ecdllConvergenceThreshold) and (iter < this->maxIter)) {
			logLikelihood_exp_data_current = this->logLikelihood;
			this->EM_current.ecd_ll_per_iter[iter] = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_current;
		}
	}
	this->ComputeLogLikelihood();
	this->StoreParamsInEMCurrent("final"); // switch to three regime
	this->EM_current.iter = iter;
	this->EM_current.ll_final = this->logLikelihood;	
	// cout << "log-likelihood computed by marginalization using EM parameters " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using EM parameters " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// (*this->logFile) << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	
	return tuple<int,double,double,double,double>(iter,logLikelihood_diri,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
}

/*
1. Number of iterations until EM converges
2. Initial log-likelihood with parsimony based parameters
3. Expected-data log-likelihood after one EM iteration
4. Expected-data log-likelihood after last EM iteration
5. Final log-likelihood using EM based parameters
*/

// tuple <int,double,double,double,double> SEM::EM_started_with_parsimony_rooted_at(SEM_vertex *v) {
// 	// iterate over each internal node	
// 	this->RootTreeAtVertex(v);
// 	// cout << "10a" << endl;
// 	// select sites that don't have gap
// 	this->ComputeMPEstimateOfAncestralSequences(); // iterate over all sites including gaps	
// 	// cout << "10b" << endl;
// 	this->ComputeInitialEstimateOfModelParameters(); // skip sites that have gaps
// 	// cout << "10c" << endl;
// 	this->StoreParamsInEMCurrent("init");
// 	// cout << "10d" << endl;
// 	// cout << this->EM_current.root_prob_init[0] << "\t" << this->EM_current.root_prob_init[1] 
// 	// << "\t" << this->EM_current.root_prob_init[2] 
// 	// << "\t" << this->EM_current.root_prob_init[3] << endl;
// 	this->ComputeLogLikelihood(); // set conditional likelihood to 1 for all char having a gap	
// 	// cout << "Initial value of log-likelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;
// 	// cout << "10e" << endl;
// 	this->ResetAncestralSequences();
// 	// cout << "10f" << endl;
// 	double logLikelihood_pars;
// 	double logLikelihood_exp_data_current;
// 	double logLikelihood_exp_data_first;
// 	double logLikelihood_exp_data_final;
// 	int iter = 0;	
// 	bool continueIterations = 1;
// 	this->debug = 0;
// 	bool verbose = 0;
// 	logLikelihood_pars = this->logLikelihood;
// 	this->EM_current.ll_init = logLikelihood_pars;
// 	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
// 	// cout << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
// 	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
// 	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
// 	logLikelihood_exp_data_current = -1 * pow(10,4);
// 	while (continueIterations) {
// 		// t_start_time = chrono::high_resolution_clock::now();
// 		iter += 1;
// 		// if (verbose) {
// 		// 	cout << "Iteration no. " << iter << endl;
// 		// 	(*this->logFile) << "Iteration no. " << iter << endl;
// 		// }
// 		// 1. Construct clique tree		
// 		// if (verbose) {
// 		// 	cout << "Construct clique tree" << endl;
// 		// 	(*this->logFile) << "Iteration no. " << iter << endl;
			
// 		// }		
		
// 		this->ConstructCliqueTree();
// 		// cout << "10g" << endl;
					
// 		// 2. Compute expected counts
// 		this->ComputeExpectedCounts();
// 		// cout << "10h" << endl;

// 		this->ComputePosteriorProbabilitiesUsingExpectedCounts();
// 		// cout << "10i" << endl;

		
// 		// 3. Optimize model parameters
// 		// if (verbose) {
// 		// 	cout << "Optimize model parameters given expected counts" << endl;
// 		// }
		
// 		// cout << "10j" << endl;
		
// 		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
// 		// cout << "10k" << endl;
		
// 		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
// 		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
// 		this->EM_current.ecd_ll_per_iter[iter] = this->logLikelihood;
// 		if (iter == 1){
// 			// cout << "10l" << endl;
// 			logLikelihood_exp_data_first = this->logLikelihood;
// 			logLikelihood_exp_data_current = this->logLikelihood;			
// 		} else if ((this->logLikelihood > logLikelihood_exp_data_current + this->ecdllConvergenceThreshold) and (iter < this->maxIter)) {
// 			logLikelihood_exp_data_current = this->logLikelihood;
// 			// cout << "10m" << endl;
// 		} else {
// 			continueIterations = 0;
// 			logLikelihood_exp_data_final = logLikelihood_exp_data_current;
// 			// cout << "10n" << endl;
// 		}
// 	}
// 	this->ComputeLogLikelihood();
// 	// cout << "10o" << endl;
// 	this->StoreParamsInEMCurrent("final");
// 	// cout << "10p" << endl;
// 	this->EM_current.iter = iter;
// 	this->EM_current.ll_final = this->logLikelihood;	
// 	// cout << "log-likelihood computed by marginalization using EM parameters " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
// 	// cout << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
// 	// (*this->logFile) << "log-likelihood computed by marginalization using EM parameters " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
// 	// (*this->logFile) << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
// 	// cout << "10q" << endl;
// 	return tuple<int,double,double,double,double>(iter,logLikelihood_pars,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
// }


void SEM::ComputeSumOfExpectedLogLikelihoods() {
	this->sumOfExpectedLogLikelihoods = 0;
	this->sumOfExpectedLogLikelihoods += this->root->vertexLogLikelihood;
	double edgeLogLikelihood;	
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPostOrderTreeTraversal) {
		if (this->edgeLogLikelihoodsMap.find(edge) == this->edgeLogLikelihoodsMap.end()) {
//			cout << edge.first->name << "\t" << edge.second->name << endl;
		} else {
//			cout << edge.first->name << "\t" << edge.second->name << endl;
			edgeLogLikelihood = this->edgeLogLikelihoodsMap[edge];
			this->sumOfExpectedLogLikelihoods += edgeLogLikelihood;
		}				
	}
}

void SEM::ComputeMLRootedTreeForRootSearchUnderBH() {
	vector < SEM_vertex *> verticesToVisit = this->preOrderVerticesWithoutLeaves;	
	double logLikelihood_max = 0;
	int numberOfVerticesVisited = 0;	
	for (SEM_vertex * v : verticesToVisit) {
		numberOfVerticesVisited += 1;
		this->RootTreeAtVertex(v);		
		this->ComputeMLEstimateOfBHGivenExpectedDataCompletion();
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		if ((numberOfVerticesVisited < 2) or (logLikelihood_max < this->logLikelihood)) {
			logLikelihood_max = this->logLikelihood;
			this->StoreRootAndRootProbability();
			this->StoreTransitionMatrices();			
			this->StoreDirectedEdgeList();
		}
	}
	this->RestoreRootAndRootProbability();
	this->RestoreTransitionMatrices();
	this->RestoreDirectedEdgeList();
	this->SetEdgesForTreeTraversalOperations();
	this->logLikelihood = logLikelihood_max;
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


void SEM::SetMinLengthOfEdges() {	
	for (pair<pair<SEM_vertex * , SEM_vertex * >,double> edgeAndLengthsPair: this->edgeLengths){		
		if (edgeAndLengthsPair.second < pow(10,-7)){
			this->edgeLengths[edgeAndLengthsPair.first]  = pow(10,-7);
		}
	}
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


void SEM::StoreDirectedEdgeList() {
	SEM_vertex * p;
	SEM_vertex * c;
	this->directedEdgeList.clear();
	for(pair <int, SEM_vertex*> idPtrPair : *this->vertexMap){
		c = idPtrPair.second;
		if (c->parent != c) {			
			p = c->parent;
			this->directedEdgeList.push_back(make_pair(p,c));
		}
	}
}

void SEM::RestoreDirectedEdgeList() {
	this->ClearDirectedEdges();
	SEM_vertex * p;
	SEM_vertex * c;
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->directedEdgeList) {
		tie (p, c) = edge;
		c->AddParent(p);
		p->AddChild(c);		
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

void SEM::ClearUndirectedEdges() {
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		v->degree = 0;
		v->neighbors.clear();
	}
}

void SEM::ClearAllEdges() {
	this->ResetTimesVisited();
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		v->parent = v;
		v->children.clear();
		v->neighbors.clear();
		v->degree = 0;
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

array <double, 4> SEM::GetBaseCompositionForTrifle(SEM_vertex * v, int layer) {
	array <double, 4> baseCompositionArray;
	for (int dna = 0; dna < 4; dna ++) {
		baseCompositionArray[dna] = 0;
	}
	int dna_v;
	// cout << "here 12a" << endl;
	for (int site = 0; site < this->num_patterns_for_layer[layer]; site ++) {
		dna_v = v->DNAcompressed[site];
		if (dna_v != 4) baseCompositionArray[dna_v] += this->DNAPatternWeights[site];
	}
	// cout << "here 12b" << endl;
	int non_gap_sites = 0;
	for (int dna = 0; dna < 4; dna ++) {
		non_gap_sites += baseCompositionArray[dna];
	}
	// cout << "here 12c" << endl;
	for (int dna = 0; dna < 4; dna ++) {
		baseCompositionArray[dna] /= float(non_gap_sites);		
	}	

	return (baseCompositionArray);
}

array <double, 4> SEM::GetBaseComposition(SEM_vertex * v) {
	array <double, 4> baseCompositionArray;
	for (int dna = 0; dna < 4; dna ++) {
		baseCompositionArray[dna] = 0;
	}
	int dna_v;
	for (int site = 0; site < this->num_dna_patterns; site ++) {
		dna_v = v->DNAcompressed[site];
		if (dna_v != 4) baseCompositionArray[dna_v] += this->DNAPatternWeights[site];
	}
	int non_gap_sites = 0;
	for (int dna = 0; dna < 4; dna ++) {
		non_gap_sites += baseCompositionArray[dna_v];
	}
	for (int dna = 0; dna < 4; dna ++) {
		baseCompositionArray[dna] /= float(non_gap_sites);
	}
	return (baseCompositionArray);
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


void SEM::ComputeMLEOfRootProbability() {
	this->rootProbability = GetBaseComposition(this->root);
	this->root->rootProbability = this->rootProbability;
}

void SEM::ComputeMLEOfTransitionMatrices() {
	SEM_vertex * c; SEM_vertex * p;
	bool debug = 0;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap){
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			c->transitionMatrix = this->ComputeTransitionMatrixUsingAncestralStates(p,c);
			// if (debug) {
			// 	cout << "Estimated transition matrix is" << endl;
			// 	cout << c->transitionMatrix << endl;
			// }
		}
	}
	
}

void SEM::SetInitialEstimateOfModelParametersUsingDirichlet() {

	array <double, 4> pi_diri = SEM::sample_pi();

	this->rootProbability = pi_diri;
	this->root->rootProbability = pi_diri;

	SEM_vertex * p; SEM_vertex * c; emtr::Md M_pc;
	for (pair<SEM_vertex*,SEM_vertex*> edge : this->edgesForPostOrderTreeTraversal) {		
		p = edge.first;
		c = edge.second;

		for (int row = 0; row < 4; row++) {
			array <double, 4> M_row_diri = SEM::sample_M_row();
			int diri_index = 1;
			for (int col = 0; col < 4; col++) {
				if (row == col){
					M_pc[row][col] = M_row_diri[0];
				} else {
					M_pc[row][col] = M_row_diri[diri_index++];
				}
			}
		}		
		c->transitionMatrix = M_pc;
	}
}

void SEM::StoreInitialParamsInEMTrifleCurrent(int layer) {	
	SEM_vertex * c;		
	for (int dna = 0; dna < 4; dna ++) this->EMTrifle_current.root_prob_initial_trifle[layer][dna] = this->rootProbability[dna];
	for (int dna = 0; dna < 4; dna ++) {
		if(this->rootProbability[dna] == 0) {
			cout << "root prob zero";
			exit(-1);
		}

	}
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		c = idPtrPair.second;
		this->EMTrifle_current.trans_prob_initial_trifle[layer][c->name] = c->transitionMatrix;
	}
}

void SEM::StoreFinalParamsInEMTrifleCurrent(int layer) {	
	SEM_vertex * c;		
	for (int dna = 0; dna < 4; dna ++) this->EMTrifle_current.root_prob_final_trifle[layer][dna] = this->rootProbability[dna];
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		c = idPtrPair.second;
		this->EMTrifle_current.trans_prob_final_trifle[layer][c->name] = c->transitionMatrix;
	}
}

void SEM::StoreParamsInEMCurrent(string init_or_final) {
	for (int dna = 0; dna < 4; dna ++) {
		if (init_or_final == "init") {			
			this->EM_current.root_prob_init[dna] = this->rootProbability[dna];			
		} else if (init_or_final == "final") {
			this->EM_current.root_prob_final[dna] = this->rootProbability[dna];			
		} else {
			mt_error("argument not recognized");
		}		 	
	}

	SEM_vertex * c;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		c = idPtrPair.second;		
		if (c->parent != c) {			
			if (init_or_final == "init") {
				this->EM_current.trans_prob_init[c->name] =  c->transitionMatrix;
			} else if (init_or_final == "final") {
				this->EM_current.trans_prob_final[c->name] = c->transitionMatrix;
			} else {
				mt_error("argument not recognized");
			}
		}
	}
}

void SEM::SetInitialEstimateOfModelParametersUsingHSS() {
	
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

void SEM::BumpZeroEntriesOfModelParameters() {
	// characters in next layer not present in previous layer may have
	// zero probabilities 
	bool debug = 0;
	// cout << "here 11a" << endl;
	this->addToZeroEntriesOfRootProbability(this->rootProbability);
	this->root->rootProbability = this->rootProbability;
	for (int i = 0; i < 4; i++) {
		if(isnan(this->rootProbability[i])) {
			cout << "root prob not valid" << endl;
			throw mt_error("root prob not valid");
		}
	}
	if (debug) {
		cout << "Root probability is " << endl;
		for (int i = 0; i < 4; i++) {
			// cout << this->rootProbability[i] << "\t";
		}
		cout << endl;
	}
	// cout << "here 11b" << endl;
	SEM_vertex * c; SEM_vertex * p;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {			
			this->addToZeroEntriesOfTransitionMatrix(c->transitionMatrix);			
			if (debug) {
				cout << "Transition matrix for " << p->name << " to " << c->name << " is " << endl;
				// cout << c->transitionMatrix[2][3] << endl;
			}			
		}
	}
}

void SEM::ComputeInitialEstimateOfModelParametersForTrifle(int layer) {
	bool debug = 0;
	// cout << "here 11a" << endl;
	this->rootProbability = GetBaseCompositionForTrifle(this->root, layer);
	this->root->rootProbability = this->rootProbability;
	for (int i = 0; i < 4; i++) {
		if(isnan(this->rootProbability[i])) {
			cout << "root prob not valid" << endl;
			throw mt_error("root prob not valid");
		}
	}
	if (debug) {
		cout << "Root probability is " << endl;
		for (int i = 0; i < 4; i++) {
			// cout << this->rootProbability[i] << "\t";
		}
		cout << endl;
	}
	// cout << "here 11b" << endl;
	SEM_vertex * c; SEM_vertex * p;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			c->transitionMatrix = this->ComputeTransitionMatrixUsingAncestralStatesForTrifle(p,c,layer);
			
			if (debug) {
				cout << "Transition matrix for " << p->name << " to " << c->name << " is " << endl;
				// cout << c->transitionMatrix[2][3] << endl;
			}			
		}
	}
	// cout << "here 11c" << endl;
	if (debug) {
		cout << "Transition matrices have been computed" << endl;
	}	
}

void SEM::ComputeInitialEstimateOfModelParameters() {
	bool debug = 0;
	this->rootProbability = GetBaseComposition(this->root);	
	this->root->rootProbability = this->rootProbability;
	if (debug) {
		// cout << "Root probability is " << endl;
		// for (int i = 0; i < 4; i++) {
		// 	cout << this->rootProbability[i] << "\t";
		// }
		// cout << endl;
	}

	SEM_vertex * c; SEM_vertex * p;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap){
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			c->transitionMatrix = this->ComputeTransitionMatrixUsingAncestralStates(p,c);		
			if (debug) {
				cout << "Transition matrix for " << p->name << " to " << c->name << " is " << endl;
				// cout << c->transitionMatrix[2][3] << endl;
			}			
		}
	}
	if (debug) {
		cout << "Transition matrices have been computed" << endl;
	}	
}


void SEM::ResetLogScalingFactors() {
	for (pair <int, SEM_vertex * > idPtrPair : * this->vertexMap){
		idPtrPair.second->logScalingFactors = 0;
	}
}

void SEM::ComputeMPEstimateOfAncestralSequencesForTrifle(int layer) {
	SEM_vertex * p;	
	map <SEM_vertex * , int> V;
	map <SEM_vertex * , vector<int>> VU;					
	map <int, int> dnaCount;
	int pos;
	int maxCount; int numberOfPossibleStates;
	this->ResetAncestralSequences();
	if (this->preOrderVerticesWithoutLeaves.size() == 0) {
		// cout << "setting vertices for preorder traversal" << endl;
		this->SetVerticesForPreOrderTraversalWithoutLeaves();
	} else {
		// cout << "vertices for preorder traversal already set" << endl;
	}
	
	// cout << "p 1" << endl;

	// cout << "site patterns for layer is " << this->num_patterns_for_layer[layer] << endl;
	for (int site = 0; site < this->num_patterns_for_layer[layer]; site++) {		
		V.clear();
		VU.clear();		
	//	Compute V and VU for leaves	
		// if (site==0) cout << "p 2" << endl;
		for (SEM_vertex * c : this->leaves) {			
			V.insert(make_pair(c,c->DNAcompressed[site]));
			vector <int> vectorToAdd;
			vectorToAdd.push_back(c->DNAcompressed[site]);
			VU.insert(make_pair(c,vectorToAdd));	
		}		
	//	Set VU for ancestors
		for (SEM_vertex * c : this->preOrderVerticesWithoutLeaves) {
			vector <int> vectorToAdd;
			VU.insert(make_pair(c,vectorToAdd));
		}				
		for (int p_ind = this->preOrderVerticesWithoutLeaves.size()-1; p_ind > -1; p_ind--) {			
			p = preOrderVerticesWithoutLeaves[p_ind];
			// cout << p->name << " rf \t" << endl;			
			dnaCount.clear();
			for (int dna = 0; dna < 5; dna++) {
				dnaCount[dna] = 0;
			}
			for (SEM_vertex * c : p->children) {
				for (int dna: VU[c]) {
					dnaCount[dna] += 1;
				}
			}
			maxCount = 0;
			for (pair <int, int> dnaCountPair: dnaCount) {
				if (dnaCountPair.second > maxCount) {
					maxCount = dnaCountPair.second;
				}
			}			
			for (pair <int, int> dnaCountPair: dnaCount) {
				if (dnaCountPair.second == maxCount) {
					VU[p].push_back(dnaCountPair.first);
				}
			}			
		}			
		
	// Set V for ancestors
		for (SEM_vertex * c : preOrderVerticesWithoutLeaves) {
			if (c->parent == c) {			
			// Set V for root
				if (VU[c].size()==1) {
					V.insert(make_pair(c,VU[c][0]));
					// cout << c->name << "\t" << VU[c][0] << endl;
				} else {
					numberOfPossibleStates = VU[c].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V.insert(make_pair(c,VU[c][pos]));
				}				
			} else {
				p = c->parent;
				if (find(VU[c].begin(),VU[c].end(),V[p])==VU[c].end()) {
					numberOfPossibleStates = VU[c].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V.insert(make_pair(c,VU[c][pos]));
					// cout << c->name << " ancestor \t" << VU[c][pos] << endl;
				} else {
					V.insert(make_pair(c,V[p]));
					// cout << c->name << " ancestor \t" << V[p] << endl;
				}				
			}				
			c->DNAcompressed[site] = V[c];
		}				
	}	
}

void SEM::ComputeMPEstimateOfAncestralSequences() {
	SEM_vertex * p;	
	map <SEM_vertex * , int> V;
	map <SEM_vertex * , vector<int>> VU;					
	map <int, int> dnaCount;
	int pos;
	int maxCount; int numberOfPossibleStates;
	this->ResetAncestralSequences();
	if (this->preOrderVerticesWithoutLeaves.size() == 0) {
		this->SetVerticesForPreOrderTraversalWithoutLeaves();
	}
		
	for (int site = 0; site < this->num_dna_patterns; site++) {		
		V.clear();
		VU.clear();		
	//	Compute V and VU for leaves		
		for (SEM_vertex * c : this->leaves) {			
			V.insert(make_pair(c,c->DNAcompressed[site]));
			vector <int> vectorToAdd;
			vectorToAdd.push_back(c->DNAcompressed[site]);			
			VU.insert(make_pair(c,vectorToAdd));			
		}		
	//	Set VU for ancestors
		for (SEM_vertex* c : this->preOrderVerticesWithoutLeaves) {
			vector <int> vectorToAdd;
			VU.insert(make_pair(c,vectorToAdd));
		}				
		for (int p_ind = this->preOrderVerticesWithoutLeaves.size()-1; p_ind > -1; p_ind--) {
			p = preOrderVerticesWithoutLeaves[p_ind];			
			dnaCount.clear();
			for (int dna = 0; dna < 5; dna++) {
				dnaCount[dna] = 0;
			}
			for (SEM_vertex * c : p->children) {
				for (int dna: VU[c]) {
					dnaCount[dna] += 1;
				}
			}
			maxCount = 0;
			for (pair <int, int> dnaCountPair: dnaCount) {
				if (dnaCountPair.second > maxCount) {
					maxCount = dnaCountPair.second;
				}
			}			
			for (pair <int, int> dnaCountPair: dnaCount) { 
				if (dnaCountPair.second == maxCount) {
					VU[p].push_back(dnaCountPair.first);					
				}
			}			
		}			
		
	// Set V for ancestors
		for (SEM_vertex * c : preOrderVerticesWithoutLeaves) {
			if (c->parent == c) {			
			// Set V for root
				if (VU[c].size()==1) {
					V.insert(make_pair(c,VU[c][0]));
				} else {
					numberOfPossibleStates = VU[c].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V.insert(make_pair(c,VU[c][pos]));
				}				
			} else {
				p = c->parent;
				if (find(VU[c].begin(),VU[c].end(),V[p])==VU[c].end()) {
					numberOfPossibleStates = VU[c].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V.insert(make_pair(c,VU[c][pos]));
				} else {
					V.insert(make_pair(c,V[p]));
				}				
			}				
			c->DNAcompressed[site] = V[c];
		}				
	}	
}


void SEM::ComputeMAPEstimateOfAncestralSequences() {
	if (this->root->DNArecoded.size() > 0) {
		this->ResetAncestralSequences();
	}	
	this->logLikelihood = 0;
	double currentProbability;
	map <SEM_vertex*, array<double,4>> conditionalLikelihoodMap;
	array <double,4> conditionalLikelihood;
	double maxProbability;
	double stateWithMaxProbability;
	double partialLikelihood;
	double siteLikelihood;
	double largestConditionalLikelihood = 0;
	double currentProb;
	int dna_ind_p; int dna_ind_c;
	vector <SEM_vertex *> verticesToVisit;
	SEM_vertex * p;
	SEM_vertex * c;
	emtr::Md P;
	for (int site = 0 ; site < this->num_dna_patterns; site++){
		conditionalLikelihoodMap.clear();
		this->ResetLogScalingFactors();
		for (pair<SEM_vertex *,SEM_vertex *> edge : this->edgesForPostOrderTreeTraversal){
			tie (p, c) = edge;					
			P = c->transitionMatrix;
			p->logScalingFactors += c->logScalingFactors;				
			// Initialize conditional likelihood for leaves
			if (c->observed) {
				for (int dna_c = 0; dna_c < 4; dna_c ++){
					conditionalLikelihood[dna_c] = 0;
				}
				conditionalLikelihood[c->DNArecoded[site]] = 1;
				conditionalLikelihoodMap.insert(pair<SEM_vertex *,array<double,4>>(c,conditionalLikelihood));
			}
			// Initialize conditional likelihood for ancestors
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()){
				for (int dna_c = 0; dna_c < 4; dna_c++){
				conditionalLikelihood[dna_c] = 1;
				}
				conditionalLikelihoodMap.insert(pair<SEM_vertex *,array<double,4>>(p,conditionalLikelihood));
			}
			largestConditionalLikelihood = 0;
			for (int dna_p = 0; dna_p < 4; dna_p++) {
				partialLikelihood = 0;
				for (int dna_c = 0; dna_c < 4; dna_c++) {
					partialLikelihood += P[dna_p][dna_c]*conditionalLikelihoodMap[c][dna_c];
				}
				conditionalLikelihoodMap[p][dna_p] *= partialLikelihood;
				if (conditionalLikelihoodMap[p][dna_p] > largestConditionalLikelihood) {
					largestConditionalLikelihood = conditionalLikelihoodMap[p][dna_p];
				}
			}
			if (largestConditionalLikelihood != 0){
				for (int dna_p = 0; dna_p < 4; dna_p++) {
					conditionalLikelihoodMap[p][dna_p] /= largestConditionalLikelihood;
				}
				p->logScalingFactors += log(largestConditionalLikelihood);
			} else {
				cout << "Largest conditional likelihood value is zero" << endl;
                throw mt_error("Largest conditional likelihood value is zero");                
			}					
		}
		maxProbability = -1; stateWithMaxProbability = 10;	
		for (dna_ind_c = 0; dna_ind_c < 4; dna_ind_c ++) {
			currentProbability = this->rootProbability[dna_ind_c];
			currentProbability *= conditionalLikelihoodMap[this->root][dna_ind_c];
			if (currentProbability > maxProbability) {
				maxProbability = currentProbability;
				stateWithMaxProbability = dna_ind_c;
			}
		}
		if (stateWithMaxProbability > 3) {
			cout << maxProbability << "\tError in computing maximum a posterior estimate for ancestor vertex\n";
		} else {
			this->root->DNArecoded.push_back(stateWithMaxProbability);
		}
//		Compute MAP estimate for each ancestral sequence
		for (pair <SEM_vertex *,SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {			
			tie (p, c) = edge;
			P = c->transitionMatrix;
			if (!c->observed) {
				maxProbability = -1; stateWithMaxProbability = 10;
				dna_ind_p = p->DNArecoded[site];
				for (dna_ind_c = 0; dna_ind_c < 4; dna_ind_c ++){ 
					currentProbability = P[dna_ind_p][dna_ind_c];
					currentProbability *= conditionalLikelihoodMap[c][dna_ind_c];
					if (currentProbability > maxProbability) {
						maxProbability = currentProbability;
						stateWithMaxProbability = dna_ind_c;
					}
				}
				if (stateWithMaxProbability > 3) {
//					cout << "Error in computing maximum a posterior estimate for ancestor vertex";
				} else {
					c->DNArecoded.push_back(stateWithMaxProbability);
				}
			}
		}		
		siteLikelihood = 0; 							
		for (int dna = 0; dna < 4; dna++) {
			currentProb = this->rootProbability[dna]*conditionalLikelihoodMap[this->root][dna];
			siteLikelihood += currentProb;					
		}
		this->logLikelihood += (this->root->logScalingFactors + log(siteLikelihood)) * this->DNAPatternWeights[site];				
	}
}

void SEM::ComputePosteriorProbabilitiesUsingMAPEstimates() {
	this->posteriorProbabilityForVertexPair.clear();
	emtr::Md P;
	SEM_vertex * u;
	SEM_vertex * v;
	double sum;
	int dna_u; int dna_v;	
	for (unsigned int u_id = 0; u_id < this->vertexMap->size()-1; u_id ++) {
		u = (*this->vertexMap)[u_id];		
		// Posterior probability for vertex u
		u->posteriorProbability = this->GetBaseComposition(u);
		// Posterior probabilies for vertex pair (u,v)
		for (unsigned int v_id = u_id + 1 ; v_id < this->vertexMap->size()-1; v_id ++) {			
			v = (*this->vertexMap)[v_id];
			P = emtr::Md{};
			for (int site = 0; site < this->num_dna_patterns; site++ ) {		
				dna_u = u->DNArecoded[site];
				dna_v = v->DNArecoded[site];
				P[dna_u][dna_v] += this->DNAPatternWeights[site];
			}
			sum = 0;
			for (dna_u = 0; dna_u < 4; dna_u ++) {				
				for (dna_v = 0; dna_v < 4; dna_v ++) {
					sum += P[dna_u][dna_v];
				}				
			}
			for (dna_u = 0; dna_u < 4; dna_u ++) {				
				for (dna_v = 0; dna_v < 4; dna_v ++) {
					P[dna_u][dna_v] /= sum;
				}				
			}			
			this->posteriorProbabilityForVertexPair.insert(make_pair(make_pair(u,v),P));
		}
		
	}	
}

double SEM::GetExpectedMutualInformation(SEM_vertex * x, SEM_vertex* y) {
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (x->id < y->id) {
		vertexPair = pair <SEM_vertex *, SEM_vertex *>(x,y);
	} else {
		vertexPair = pair <SEM_vertex *, SEM_vertex *>(y,x);
	}
	
	emtr::Md P = this->posteriorProbabilityForVertexPair[vertexPair];
//	cout << "Joint probability for vertex pair " << x->name << "\t" << y->name << " is " << endl;
//	cout << P << endl;
	std::array <double, 4> P_x;
	std::array <double, 4> P_y;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		P_x[dna_x] = 0;
		P_y[dna_x] = 0;
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			P_x[dna_x] += P[dna_x][dna_y];
			P_y[dna_x] += P[dna_y][dna_x];
		}		
	}

//	cout << endl;
	double mutualInformation = 0;
//	double inc;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			if (P[dna_x][dna_y] > 0) {
				mutualInformation += P[dna_x][dna_y] * log(P[dna_x][dna_y]/(P_x[dna_x] * P_y[dna_y]));
			}
		}
	}
//	cout << "P_XY is " << endl << P << endl;
//	cout << "mutual information is " << mutualInformation << endl;
	return (mutualInformation);
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

void SEM::RootedTreeAlongAnEdgeIncidentToCentralVertex() {	
	// Identify a central vertex
	vector <SEM_vertex*> verticesToVisit;
	vector <SEM_vertex*> verticesVisited;
	SEM_vertex * u; SEM_vertex * v;
	int n_ind; int u_ind;
	for (pair <int, SEM_vertex *> idPtrPair : *this->vertexMap) {
		v = idPtrPair.second;
		v->timesVisited = 0;
		if (v->observed) {
			verticesToVisit.push_back(v);
		}
	}
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		v = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(v);		
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex* n: v->neighbors) {
			if (find(verticesVisited.begin(),verticesVisited.end(),n)==verticesVisited.end()) {
				n->timesVisited += 1;
				if ((n->degree - n->timesVisited) == 1) {
					verticesToVisit.push_back(n);
					numberOfVerticesToVisit += 1;
				}
			} else {				
				n_ind = find(verticesVisited.begin(),verticesVisited.end(),n) - verticesVisited.begin();
				verticesVisited.erase(verticesVisited.begin()+n_ind);
			}
		}
	}
	// v is a central vertex	
	// Root tree at a randomly selected neighbor u of v
	uniform_int_distribution <int> distribution(0,v->neighbors.size()-1);
	u_ind = distribution(generator);
	u = v->neighbors[u_ind];
	this->root->AddChild(u);
	this->root->AddChild(v);
	u->AddParent(this->root);
	v->AddParent(this->root);
	verticesToVisit.clear();
	verticesVisited.clear();
	verticesToVisit.push_back(u);
	verticesToVisit.push_back(v);
	verticesVisited.push_back(u);
	verticesVisited.push_back(v);
	numberOfVerticesToVisit = verticesToVisit.size() - 1;
	while (numberOfVerticesToVisit > 0) {
		v = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(v);
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex* n: v->neighbors) {
			if (find(verticesVisited.begin(),verticesVisited.end(),n)==verticesVisited.end()) {
				verticesToVisit.push_back(n);
				numberOfVerticesToVisit += 1;
				v->AddChild(n);
				n->AddParent(v);
			}
		}
	}
	this->SetLeaves();	
	this->SetEdgesForPreOrderTraversal();
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
	this->SetEdgesForPostOrderTraversal();
}


void SEM::SetIdsOfExternalVertices() {	
	this->idsOfExternalVertices.clear();
	SEM_vertex * v;
	for (int i = this->numberOfVerticesInSubtree; i < this->numberOfObservedVertices; i++) {		
		v = (*this->vertexMap)[i];
		this->idsOfExternalVertices.push_back(v->global_id);
	}
}

void SEM::AddSitePatternWeights(vector <int> sitePatternWeightsToAdd) {
	this->DNAPatternWeights = sitePatternWeightsToAdd;
	this->num_dna_patterns = this->DNAPatternWeights.size();
	this->sequenceLength = 0;
	for (int sitePatternWeight : this->DNAPatternWeights) {
		this->sequenceLength += sitePatternWeight;
	}
}

void SEM::AddSitePatternRepeats(vector <vector <int> > sitePatternRepetitionsToAdd) {
	this->sitePatternRepetitions = sitePatternRepetitionsToAdd;
}


void SEM::AddNames(vector <string> namesToAdd) {
	if (this->numberOfObservedVertices == 0) {
		this->numberOfObservedVertices = namesToAdd.size();
	}	
	for (int i = 0; i < this->numberOfObservedVertices; i++) {
		(*this->vertexMap)[i]->name = namesToAdd[i];
		this->nameToIdMap.insert(make_pair(namesToAdd[i],i));
	}
	this->externalVertex = (*this->vertexMap)[this->numberOfObservedVertices-1];
}

void SEM::AddSequences(vector <vector <int>> sequencesToAdd) {
	this->numberOfObservedVertices = sequencesToAdd.size();
	this->node_ind = this->numberOfObservedVertices;
	for (int i = 0 ; i < this->numberOfObservedVertices; i++) {
		SEM_vertex * v = new SEM_vertex(i,sequencesToAdd[i]);
		v->observed = 1;
		this->vertexMap->insert(make_pair(i,v));
	}	
}

void SEM::AddRootVertex() {
	int n = this->numberOfObservedVertices;
	vector <int> emptySequence;	
	this->root = new SEM_vertex (-1,emptySequence);
	this->root->name = "h_root";	
	this->root->id = ( 2 * n ) - 2;
	this->vertexMap->insert(pair<int,SEM_vertex*>((( 2 * n ) - 2 ),this->root));
	this->nameToIdMap.insert(make_pair(this->root->name,this->root->id));
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

void SEM::AddEdgeLogLikelihoods(vector<tuple<string,string,double>> edgeLogLikelihoodsToAdd) {
	SEM_vertex * u; SEM_vertex * v; double edgeLogLikelihood;
	string u_name; string v_name;
	pair<SEM_vertex *, SEM_vertex *> vertexPair;
	for (tuple<string,string,double> edgeLogLikelihoodTuple : edgeLogLikelihoodsToAdd) {
		tie (u_name, v_name, edgeLogLikelihood) = edgeLogLikelihoodTuple;
		u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		vertexPair = pair <SEM_vertex *, SEM_vertex *> (u,v);
		this->edgeLogLikelihoodsMap.insert(pair<pair <SEM_vertex *, SEM_vertex *>,double>(vertexPair, edgeLogLikelihood));
	}	
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

///...///...///...///...///...///...///... EMBH ///...///...///...///...///...///...///...///...///

EMBH::EMBH(const string edge_list_file_name,
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
		
		this->P->SetVertexVector();
		this->P->SetVertexVectorExceptRoot();
				
		cout << "Number of unique site patterns is " << this->P->num_dna_patterns << endl;

		this->P->SetEdgesFromTopologyFile(edge_list_file_name);
		// set root probability as sample base composition
		this->P->SetRootProbabilityAsSampleBaseComposition(base_composition_file_name);
		// define F81 model using base composition file
		SEM_vertex * root_optim = this->P->GetVertex(root_optim_name);
		this->P->RootTreeAtVertex(root_optim);
		// [] set F81 model
		this->P->SetF81Model(base_composition_file_name); // assign transition probabilities
		// [] compute log-likelihood score using pruning algorithm with fasta input
		cout << "\n=== Computing log-likelihood using compressed sequences ===" << endl;
		this->P->ComputeLogLikelihood();
		cout << "log-likelihood using pruning algorithm and compressed sequences is " << this->P->logLikelihood << endl;		

		if (!pattern_file_name.empty() && !taxon_order_file_name.empty()) {
			cout << "\n=== Computing log-likelihood using packed patterns ===" << endl;
			this->P->ReadPatternsFromFile(pattern_file_name, taxon_order_file_name);
			this->P->ComputeLogLikelihoodUsingPatterns();
			cout << "log-likelihood using pruning algorithm and packed patterns is " 
			     << this->P->logLikelihood << endl;
		}
		// [] compute log-likelihood score using pruning algorithm with pattern input
		// [] compute log-likelihood score using propagation algorithm with pattern input
		/* [] reuse messages for branch patterns and compute log-likelihood
		 using propagation algorithm with */
		// the following is for BH modelgre
		SEM_vertex * root_check = this->P->GetVertex(root_check_name);
		// set root estimate and root test
		}

		EMBH::~EMBH() {			
			delete this->P;		
		}

	void EMBH::EM_main() {
	this->num_repetitions = this->P->num_repetitions;
	this->P->max_log_likelihood_best = -1 * std::pow(10.0, 10.0);

	const int total_nodes = max(0, int(this->P->vertexMap->size()) - this->P->numberOfObservedVertices);
		
	for (int rep = 0; rep < this->num_repetitions; ++rep) {	
		this->P->EMTrifle_DNA_for_replicate(rep + 1);		
	}
}


void EMBH::EMparsimony() {
    cout << "Starting EM with initial parameters set using parsimony" << endl;
	this->P->probabilityFileName_pars = this->prefix_for_output_files + ".pars_prob";
	this->probabilityFileName_pars = this->prefix_for_output_files + ".pars_prob";
    this->P->EM_rooted_at_each_internal_vertex_started_with_parsimony(this->num_repetitions);
}


void EMBH::EMdirichlet() {	
	cout << "Starting EM with initial parameters set using Dirichlet" << endl;
	this->P->probabilityFileName_diri = this->prefix_for_output_files + ".diri_prob";
	this->probabilityFileName_diri = this->prefix_for_output_files + ".diri_prob";
	this->P->EM_rooted_at_each_internal_vertex_started_with_dirichlet(this->num_repetitions);
}

void EMBH::SetprobFileforHSS() {
	if (this->max_log_lik_pars > this->max_log_lik_diri) {
		cout << "Initializing with Parsimony yielded higher log likelihood score" << endl;
		this->probabilityFileName_best = this->probabilityFileName_pars;
	} else {
		this->probabilityFileName_best = this->probabilityFileName_diri;
		cout << "Initializing with Dirichlet yielded higher log likelihood score" << endl;
	}
}

void EMBH::EMhss() {
	this->SetprobFileforHSS();
	cout << "Starting EM with initial parameters set using Bayes rule as described in HSS paper" << endl;
	this->P->probabilityFileName_best = this->probabilityFileName_best;	
    this->P->SetBHparameters();
	this->P->ReparameterizeBH();    
    this->P->EM_rooted_at_each_internal_vertex_started_with_HSS_par(this->num_repetitions);
}

void EMBH::SetDNAMap() {
	this->mapDNAtoInteger["A"] = 0;
	this->mapDNAtoInteger["C"] = 1;
	this->mapDNAtoInteger["G"] = 2;
	this->mapDNAtoInteger["T"] = 3;
}

