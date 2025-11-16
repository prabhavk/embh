#!/usr/bin/env python3
"""
Extract all code needed for the Manager constructor into trimmed_cpp folder.
This script:
1. Uses the call graph analysis results
2. Extracts all required methods
3. Extracts all required classes (SEM_vertex, clique, cliqueTree, PackedPatternStorage)
4. Builds a complete embh_core.cpp with only necessary code
"""

import re
from pathlib import Path

def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def extract_lines(code, start, end):
    """Extract lines from start to end (1-indexed, inclusive)."""
    lines = code.split('\n')
    return '\n'.join(lines[start-1:end])

def find_method_bounds(code, class_name, method_name):
    """Find exact start and end lines of a method implementation."""
    lines = code.split('\n')

    # Different patterns for constructor vs regular method
    if method_name == class_name:
        # Constructor
        patterns = [
            rf'^{class_name}::{method_name}\s*\(',
            rf'^{class_name}::{method_name}\s*\([^)]*\)\s*\{{',
        ]
    else:
        # Regular method
        patterns = [
            rf'^\w+[\s\*&:<>,]*\s+{class_name}::{method_name}\s*\(',
            rf'^{class_name}::{method_name}\s*\(',
        ]

    start_line = None

    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.match(pattern, line.lstrip()):
                start_line = i + 1
                break
        if start_line:
            break

    if not start_line:
        return None, None

    # Find matching closing brace
    brace_count = 0
    in_body = False

    for i in range(start_line - 1, len(lines)):
        line = lines[i]
        if not in_body:
            if '{' in line:
                brace_count = line.count('{') - line.count('}')
                in_body = True
                if brace_count == 0:
                    return start_line, i + 1
        else:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                return start_line, i + 1

    return start_line, None

def find_class_definition(code, class_name):
    """Find complete class definition including all members."""
    lines = code.split('\n')

    start_line = None
    brace_count = 0
    in_class = False

    for i, line in enumerate(lines):
        if start_line is None:
            if re.match(rf'^class\s+{class_name}\s*(?::\s*|$|\{{)', line):
                start_line = i + 1
                if '{' in line:
                    brace_count = line.count('{') - line.count('}')
                    in_class = True
        elif not in_class:
            if '{' in line:
                brace_count = line.count('{') - line.count('}')
                in_class = True
        else:
            brace_count += line.count('{') - line.count('}')
            if brace_count == 0:
                return start_line, i + 1

    return start_line, None

# List of all SEM methods needed (from call graph analysis)
SEM_METHODS = [
    ("GetVertex", 1695),
    ("GetVertexId", 1709),
    ("ConvertDNAtoIndex", 1789),
    ("SetVertexVector", 1812),
    ("SetVertexVectorExceptRoot", 1821),
    ("ReparameterizeBH", 1834),
    ("AddToExpectedCountsForEachVariable", 2622),
    ("AddToExpectedCountsForEachEdge", 2711),
    ("ComputeExpectedCounts", 2826),
    ("ComputeMarginalProbabilitiesUsingExpectedCounts", 2900),
    ("ConstructCliqueTree", 2959),
    ("ResetExpectedCounts", 3021),
    ("InitializeExpectedCountsForEachVariable", 3046),
    ("InitializeExpectedCountsForEachEdge", 3092),
    ("ReadPatternsFromFile", 3359),
    ("StorePatternIndex", 3436),
    ("ComputeLogLikelihood", 3467),
    ("ComputeLogLikelihoodUsingPatterns", 3569),
    ("ComputeLogLikelihoodUsingPatternsWithPropagation", 3738),
    ("embh_aitken", 4101),
    ("GetP_yGivenx", 5170),
    ("ComputeMLEstimateOfBHGivenExpectedDataCompletion", 5208),
    ("RootTreeAtVertex", 5264),
    ("ClearDirectedEdges", 5289),
    ("ResetTimesVisited", 5325),
    ("GetObservedCountsForVariable", 5377),
    ("EvaluateBHModelWithRootAtCheck", 5491),
    ("SetModelParametersUsingHSS", 5501),
    ("ResetLogScalingFactors", 5624),
    ("SetEdgesForTreeTraversalOperations", 5981),
    ("SetEdgesForPreOrderTraversal", 5988),
    ("SetLeaves", 6006),
    ("SetEdgesForPostOrderTraversal", 6015),
    ("SetVerticesForPreOrderTraversalWithoutLeaves", 6045),
    ("SetEdgesFromTopologyFile", 6177),
    ("SetRootProbabilityAsSampleBaseComposition", 6229),
    ("SetF81_mu", 6257),
    ("SetF81Matrix", 6272),
    ("SetF81Model", 6293),
    ("CompressDNASequences", 6300),
    ("SetDNASequencesFromFile", 6439),
    ("ContainsVertex", 6511),
    ("AddVertex", 6519),
    ("GetLengthOfSubtendingBranch", 2157),  # Needed by SetF81Matrix
    ("gap_proportion_in_pattern", 1573),
    ("unique_non_gap_count_in_pattern", 1580),
]

def main():
    code_path = Path(__file__).parent / "embh_core.cpp"
    code = read_file(code_path)

    print("Extracting code for trimmed_cpp...")
    print("=" * 70)

    # Extract all method implementations
    print("\n1. Extracting SEM method implementations...")
    methods_code = {}

    for method_name, approx_line in SEM_METHODS:
        start, end = find_method_bounds(code, "SEM", method_name)
        if start and end:
            methods_code[method_name] = {
                'code': extract_lines(code, start, end),
                'start': start,
                'end': end
            }
            print(f"   {method_name}: lines {start}-{end} ({end-start+1} lines)")
        else:
            print(f"   WARNING: {method_name} not found!")

    # Also extract manager methods
    print("\n2. Extracting manager methods...")
    manager_methods = ["manager", "~manager", "SetDNAMap"]

    for method in manager_methods:
        if method == "~manager":
            start, end = find_method_bounds(code, "manager", "~manager")
        else:
            start, end = find_method_bounds(code, "manager", method)
        if start and end:
            methods_code[f"manager::{method}"] = {
                'code': extract_lines(code, start, end),
                'start': start,
                'end': end
            }
            print(f"   manager::{method}: lines {start}-{end}")
        else:
            print(f"   WARNING: manager::{method} not found!")

    # Extract class definitions
    print("\n3. Extracting class definitions...")

    classes_to_extract = [
        ("SEM_vertex", 192),  # approx line
        ("clique", 288),
        ("cliqueTree", 540),
        ("PackedPatternStorage", 94),
    ]

    class_defs = {}
    for class_name, approx_line in classes_to_extract:
        start, end = find_class_definition(code, class_name)
        if start and end:
            class_defs[class_name] = extract_lines(code, start, end)
            print(f"   {class_name}: lines {start}-{end} ({end-start+1} lines)")
        else:
            print(f"   WARNING: {class_name} not found!")

    # Extract SEM_vertex method implementations
    print("\n4. Extracting SEM_vertex methods...")
    sem_vertex_methods = [
        "AddParent", "RemoveParent", "AddChild", "RemoveChild",
        "AddNeighbor", "RemoveNeighbor", "SetVertexLogLikelihood"
    ]

    for method in sem_vertex_methods:
        start, end = find_method_bounds(code, "SEM_vertex", method)
        if start and end:
            methods_code[f"SEM_vertex::{method}"] = {
                'code': extract_lines(code, start, end),
                'start': start,
                'end': end
            }
            print(f"   SEM_vertex::{method}: lines {start}-{end}")

    # Extract clique methods
    print("\n5. Extracting clique methods...")
    clique_methods = [
        "MarginalizeOverVariable", "ComputeBelief", "SetInitialPotentialAndBelief",
        "AddNeighbor", "AddParent", "AddChild"
    ]

    for method in clique_methods:
        start, end = find_method_bounds(code, "clique", method)
        if start and end:
            methods_code[f"clique::{method}"] = {
                'code': extract_lines(code, start, end),
                'start': start,
                'end': end
            }
            print(f"   clique::{method}: lines {start}-{end}")

    # Extract cliqueTree methods
    print("\n6. Extracting cliqueTree methods...")
    clique_tree_methods = [
        "CalibrateTree", "SetMarginalProbabilitesForEachEdgeAsBelief",
        "SendMessage", "AddClique", "AddEdge", "SetSite",
        "InitializePotentialAndBeliefs", "SetEdgesForTreeTraversalOperations",
        "SetRoot", "SetLeaves", "GetXYZ", "GetCommonVariable"
    ]

    for method in clique_tree_methods:
        start, end = find_method_bounds(code, "cliqueTree", method)
        if start and end:
            methods_code[f"cliqueTree::{method}"] = {
                'code': extract_lines(code, start, end),
                'start': start,
                'end': end
            }
            print(f"   cliqueTree::{method}: lines {start}-{end}")

    # Extract structs
    print("\n7. Extracting struct definitions...")
    struct_defs = {}

    # EM_struct
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^struct\s+EM_struct', line):
            brace_count = 0
            for j in range(i, len(lines)):
                brace_count += lines[j].count('{') - lines[j].count('}')
                if brace_count == 0 and j > i:
                    struct_defs['EM_struct'] = '\n'.join(lines[i:j+1])
                    print(f"   EM_struct: lines {i+1}-{j+1}")
                    break
            break

    # EMTrifle_struct
    for i, line in enumerate(lines):
        if re.match(r'^struct\s+EMTrifle_struct', line):
            brace_count = 0
            for j in range(i, len(lines)):
                brace_count += lines[j].count('{') - lines[j].count('}')
                if brace_count == 0 and j > i:
                    struct_defs['EMTrifle_struct'] = '\n'.join(lines[i:j+1])
                    print(f"   EMTrifle_struct: lines {i+1}-{j+1}")
                    break
            break

    # Extract emtr namespace
    print("\n8. Extracting emtr namespace...")
    emtr_start = None
    for i, line in enumerate(lines):
        if re.match(r'^namespace\s+emtr', line):
            emtr_start = i
            break

    if emtr_start:
        brace_count = 0
        for j in range(emtr_start, len(lines)):
            brace_count += lines[j].count('{') - lines[j].count('}')
            if brace_count == 0 and j > emtr_start:
                emtr_namespace = '\n'.join(lines[emtr_start:j+1])
                print(f"   emtr namespace: lines {emtr_start+1}-{j+1}")
                break

    # Calculate total lines
    total_lines = 0
    for name, data in methods_code.items():
        total_lines += data['end'] - data['start'] + 1

    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  - SEM methods: {len([k for k in methods_code if 'SEM::' not in k and 'manager' not in k and 'SEM_vertex' not in k and 'clique' not in k])}")
    print(f"  - Total method implementations: {len(methods_code)}")
    print(f"  - Approximate total lines: {total_lines}")
    print(f"{'='*70}")

    # Write a report
    report_file = Path(__file__).parent / "extraction_report.txt"
    with open(report_file, 'w') as f:
        f.write("CODE EXTRACTION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("SEM METHODS TO EXTRACT:\n")
        f.write("-" * 70 + "\n")
        for method_name, approx_line in sorted(SEM_METHODS, key=lambda x: x[1]):
            if method_name in methods_code:
                d = methods_code[method_name]
                f.write(f"{method_name}: lines {d['start']}-{d['end']} ({d['end']-d['start']+1} lines)\n")

        f.write(f"\n\nTotal methods to extract: {len(methods_code)}\n")
        f.write(f"Total lines: ~{total_lines}\n")

    print(f"\nReport written to: {report_file}")

    return methods_code, class_defs, struct_defs, emtr_namespace

if __name__ == "__main__":
    main()
