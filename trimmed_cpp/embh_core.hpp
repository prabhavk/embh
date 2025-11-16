#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <tuple>
#include <vector>
#include <unordered_set>

class SEM_vertex;
class clique;
class cliqueTree;
class SEM;
class PackedPatternStorage;

using namespace std;

struct mt_error : runtime_error {
    using runtime_error::runtime_error;
};

class manager
{
private:
    vector <string> sequenceNames;
    map <string,unsigned char> mapDNAtoInteger;
    int numberOfHiddenVertices = 0;
    chrono::steady_clock::time_point start_time;
    chrono::steady_clock::time_point current_time;
    chrono::steady_clock::time_point t_start_time;
    chrono::steady_clock::time_point t_end_time;
    chrono::steady_clock::time_point m_start_time;
    chrono::steady_clock::time_point m_end_time;
    chrono::duration<double> timeTakenToComputeEdgeAndVertexLogLikelihoods;
    chrono::duration<double> timeTakenToComputeGlobalUnrootedPhylogeneticTree;
    chrono::duration<double> timeTakenToRootViaEdgeLoglikelihoods;
    chrono::duration<double> timeTakenToRootViaRestrictedSEM;
    string edge_list_file_name;
    string fasta_file_name;
    string pattern_file_name;
    string base_comp_file_name;
    string root_train;
    string root_test;
    string fastaFileName;
    string topologyFileName;
    string prefix_for_output_files;
    string loglikelihood_node_rep_file_name;
    string probabilityFileName_pars;
    string probabilityFileName_diri;
    string probabilityFileName_pars_root;
    string probabilityFileName_diri_root;
    string probabilityFileName_best;
    double max_ll_pars;
    double max_ll_diri;
    string MSTFileName;
    string distance_measure_for_NJ = "Hamming";
    SEM * P;
    SEM * p;
    bool debug;
    bool verbose;
    bool modelSelection;
    int num_repetitions;
    int max_iter;
    int max_EM_iter;
    double conv_thresh;
    public:

    manager(const string EdgeListFileNameToSet,
         const string PatternListFileNameToSet,
         const string TaxonOrderFileNameToSet,
         const string BaseCompositionFileNameToSet,
         const string RootEstimateToSet,
         const string RootTestToSet
        );
    ~manager();

    double max_log_lik;
    double max_log_lik_pars;
    double max_log_lik_diri;
    double max_log_lik_ssh;
    string parameters_json;
    void SetDNAMap();
};
