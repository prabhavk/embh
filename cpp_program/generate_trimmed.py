#!/usr/bin/env python3
"""
Generate the complete trimmed embh_core.cpp and embh_core.hpp files
by extracting only the code needed for the Manager constructor.
"""

import re
from pathlib import Path

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def extract_lines(lines, start, end):
    """Extract lines from start to end (1-indexed, inclusive)."""
    return '\n'.join(lines[start-1:end])

def main():
    code_path = Path(__file__).parent / "embh_core.cpp"
    code = read_file(code_path)
    lines = code.split('\n')

    print("Generating trimmed embh_core.cpp...")

    # Build the output file piece by piece
    output = []

    # 1. Includes and namespace
    output.append("""#include "embh_core.hpp"

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
""")

    # 2. emtr namespace (lines 29-70)
    output.append(extract_lines(lines, 29, 70))
    output.append("\nusing namespace emtr;\n\nint ll_precision = 14;\n")

    # 3. pattern class (lines 79-87)
    output.append("\n" + extract_lines(lines, 79, 87))

    # 4. PackedPatternStorage class (lines 94-188)
    output.append("\n" + extract_lines(lines, 94, 188))

    # 5. SEM_vertex class and its method implementations (lines 192-284)
    output.append("\n" + extract_lines(lines, 192, 284))

    # 6. clique class (lines 288-335) and its methods
    output.append("\n" + extract_lines(lines, 288, 536))

    # 7. cliqueTree class (lines 540-581)
    output.append("\n" + extract_lines(lines, 540, 581))

    # 8. cliqueTree method implementations
    # GetXYZ (584-668), GetCommonVariable (715-723), SetMarginalProbabilitesForEachEdgeAsBelief (807-817)
    # SetRoot (856-862), InitializePotentialAndBeliefs (864-868), SetSite (870-872)
    # AddClique (874-876), AddEdge (878-881), SetEdgesForTreeTraversalOperations (883-912)
    # SetLeaves (914-921), SendMessage (924-1110), CalibrateTree (1112-1136)
    output.append("\n" + extract_lines(lines, 584, 1136))

    # 9. EM_struct and EMTrifle_struct (lines 1140-1186)
    output.append("\n" + extract_lines(lines, 1140, 1186))

    # 10. SEM class definition (lines 1191-1554)
    output.append("\n" + extract_lines(lines, 1191, 1554))

    # 11. SEM utility methods (lines 1556-1624) - gap_proportion_in_pattern, addToZeroEntries, etc.
    output.append("\n" + extract_lines(lines, 1556, 1624))

    # 12. SEM method implementations - extract each needed method
    # GetVertex, GetVertexId (1695-1722)
    output.append("\n" + extract_lines(lines, 1695, 1722))

    # ConvertDNAtoIndex (1789-1810)
    output.append("\n" + extract_lines(lines, 1789, 1810))

    # SetVertexVector, SetVertexVectorExceptRoot (1812-1828)
    output.append("\n" + extract_lines(lines, 1812, 1828))

    # ReparameterizeBH (1834-1881)
    output.append("\n" + extract_lines(lines, 1834, 1881))

    # GetLengthOfSubtendingBranch (2157-2168)
    output.append("\n" + extract_lines(lines, 2157, 2168))

    # AddToExpectedCountsForEachVariable (2622-2642)
    output.append("\n" + extract_lines(lines, 2622, 2642))

    # AddToExpectedCountsForEachEdge (2711-2730)
    output.append("\n" + extract_lines(lines, 2711, 2730))

    # ComputeExpectedCounts (2826-2860)
    output.append("\n" + extract_lines(lines, 2826, 2860))

    # ComputeMarginalProbabilitiesUsingExpectedCounts (2900-2957)
    output.append("\n" + extract_lines(lines, 2900, 2957))

    # ConstructCliqueTree (2959-3019)
    output.append("\n" + extract_lines(lines, 2959, 3019))

    # ResetExpectedCounts (3021-3044)
    output.append("\n" + extract_lines(lines, 3021, 3044))

    # InitializeExpectedCountsForEachVariable (3046-3089) - includes helper
    output.append("\n" + extract_lines(lines, 3046, 3089))

    # InitializeExpectedCountsForEachEdge (3092-3108)
    output.append("\n" + extract_lines(lines, 3092, 3108))

    # ReadPatternsFromFile, StorePatternIndex (3359-3461)
    output.append("\n" + extract_lines(lines, 3359, 3461))

    # ComputeLogLikelihood (3467-3567)
    output.append("\n" + extract_lines(lines, 3467, 3567))

    # ComputeLogLikelihoodUsingPatterns (3569-3736)
    output.append("\n" + extract_lines(lines, 3569, 3736))

    # ComputeLogLikelihoodUsingPatternsWithPropagation (3738-3813)
    output.append("\n" + extract_lines(lines, 3738, 3813))

    # embh_aitken (4101-4260)
    output.append("\n" + extract_lines(lines, 4101, 4260))

    # GetP_yGivenx (5170-5196)
    output.append("\n" + extract_lines(lines, 5170, 5196))

    # ComputeMLEstimateOfBHGivenExpectedDataCompletion (5208-5237)
    output.append("\n" + extract_lines(lines, 5208, 5237))

    # RootTreeAtVertex (5264-5287)
    output.append("\n" + extract_lines(lines, 5264, 5287))

    # ClearDirectedEdges (5289-5300)
    output.append("\n" + extract_lines(lines, 5289, 5300))

    # ResetTimesVisited (5325-5331)
    output.append("\n" + extract_lines(lines, 5325, 5331))

    # GetObservedCountsForVariable (5377-5388)
    output.append("\n" + extract_lines(lines, 5377, 5388))

    # EvaluateBHModelWithRootAtCheck (5491-5499)
    output.append("\n" + extract_lines(lines, 5491, 5499))

    # SetModelParametersUsingHSS (5501-5519)
    output.append("\n" + extract_lines(lines, 5501, 5519))

    # ResetLogScalingFactors (5624-5628)
    output.append("\n" + extract_lines(lines, 5624, 5628))

    # SetEdgesForTreeTraversalOperations through SetVerticesForPreOrderTraversalWithoutLeaves (5981-6052)
    output.append("\n" + extract_lines(lines, 5981, 6052))

    # SetEdgesFromTopologyFile (6177-6227)
    output.append("\n" + extract_lines(lines, 6177, 6227))

    # SetRootProbabilityAsSampleBaseComposition (6229-6255)
    output.append("\n" + extract_lines(lines, 6229, 6255))

    # SetF81_mu (6257-6270)
    output.append("\n" + extract_lines(lines, 6257, 6270))

    # SetF81Matrix (6272-6291)
    output.append("\n" + extract_lines(lines, 6272, 6291))

    # SetF81Model (6293-6298)
    output.append("\n" + extract_lines(lines, 6293, 6298))

    # CompressDNASequences (6300-6437)
    output.append("\n" + extract_lines(lines, 6300, 6437))

    # SetDNASequencesFromFile (6439-6496)
    output.append("\n" + extract_lines(lines, 6439, 6496))

    # ContainsVertex (6511-6517)
    output.append("\n" + extract_lines(lines, 6511, 6517))

    # AddVertex (6519-6526)
    output.append("\n" + extract_lines(lines, 6519, 6526))

    # manager constructor (6530-6580)
    output.append("\n" + extract_lines(lines, 6530, 6580))
    # manager destructor (6582-6584)
    output.append("\n" + extract_lines(lines, 6582, 6584))
    # manager::SetDNAMap (6632-6637)
    output.append("\n" + extract_lines(lines, 6632, 6637))

    # Write the output
    output_path = Path(__file__).parent.parent / "trimmed_cpp" / "embh_core.cpp"
    with open(output_path, 'w') as f:
        f.write('\n'.join(output))

    print(f"Generated: {output_path}")

    # Count lines
    total_lines = sum(len(section.split('\n')) for section in output)
    print(f"Total lines: {total_lines}")

    # Generate header file
    print("\nGenerating trimmed embh_core.hpp...")

    header = """#pragma once

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
         const string FastaFileNameToSet,
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
"""

    header_path = Path(__file__).parent.parent / "trimmed_cpp" / "embh_core.hpp"
    with open(header_path, 'w') as f:
        f.write(header)

    print(f"Generated: {header_path}")

if __name__ == "__main__":
    main()
