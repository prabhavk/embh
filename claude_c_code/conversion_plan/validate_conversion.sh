#!/bin/bash
# validate_conversion.sh
# Comprehensive validation script for C++ to C conversion
# Compares log-likelihood values and other key metrics between C++ and C implementations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CPP_BINARY="./bin/embh"
C_BINARY="./embh_c"
DATA_DIR="data"
EDGE_LIST="${DATA_DIR}/RAxML_bipartitions.CDS_FcC_partition.edgelist"
PATTERNS="${DATA_DIR}/patterns_1000.pat"
TAXON_ORDER="${DATA_DIR}/patterns_1000.taxon_order"
BASE_COMP="${DATA_DIR}/patterns_1000.basecomp"
ROOT_OPT="h_0"
ROOT_CHECK="h_5"

CPP_OUTPUT="cpp_output.txt"
C_OUTPUT="c_output.txt"
VALIDATION_REPORT="validation_report.txt"

TOLERANCE=1e-10  # Maximum acceptable difference in log-likelihoods

# Function to print section headers
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Function to extract numeric value from output
extract_value() {
    local file=$1
    local pattern=$2
    grep "$pattern" "$file" | grep -oE '[+-]?[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?' | head -1
}

# Function to compare two floating point numbers
compare_floats() {
    local val1=$1
    local val2=$2
    local tolerance=$3
    local description=$4
    
    if [ -z "$val1" ] || [ -z "$val2" ]; then
        echo -e "${RED}✗ FAIL${NC}: $description - Missing value (C++: $val1, C: $val2)"
        return 1
    fi
    
    # Use awk for floating point comparison
    local diff=$(awk -v a="$val1" -v b="$val2" 'BEGIN {print (a-b < 0 ? b-a : a-b)}')
    local passes=$(awk -v diff="$diff" -v tol="$tolerance" 'BEGIN {print (diff < tol ? 1 : 0)}')
    
    if [ "$passes" -eq 1 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $description"
        echo "  C++:        $val1"
        echo "  C:          $val2"
        echo "  Difference: $diff (< $tolerance)"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}: $description"
        echo "  C++:        $val1"
        echo "  C:          $val2"
        echo "  Difference: $diff (>= $tolerance)"
        return 1
    fi
}

# Function to check if value exists in output
check_value_exists() {
    local file=$1
    local pattern=$2
    local description=$3
    
    if grep -q "$pattern" "$file"; then
        echo -e "${GREEN}✓ PASS${NC}: $description found in output"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}: $description NOT found in output"
        return 1
    fi
}

# Initialize report
echo "EMBH C++ to C Conversion Validation Report" > "$VALIDATION_REPORT"
echo "==========================================" >> "$VALIDATION_REPORT"
echo "Date: $(date)" >> "$VALIDATION_REPORT"
echo "" >> "$VALIDATION_REPORT"

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Main validation
print_header "EMBH C++ to C Conversion Validation"
echo "Comparing outputs from C++ and C implementations"
echo ""

# Check if binaries exist
if [ ! -f "$CPP_BINARY" ]; then
    echo -e "${RED}ERROR${NC}: C++ binary not found at $CPP_BINARY"
    echo "Please run 'make' first to build the C++ version"
    exit 1
fi

if [ ! -f "$C_BINARY" ]; then
    echo -e "${YELLOW}WARNING${NC}: C binary not found at $C_BINARY"
    echo "C implementation not yet built. This is expected during early conversion stages."
    echo "This validation will run only the C++ baseline for now."
    
    # Run C++ only and save as baseline
    print_header "Running C++ Implementation (Baseline)"
    "$CPP_BINARY" -e "$EDGE_LIST" -p "$PATTERNS" -x "$TAXON_ORDER" -b "$BASE_COMP" -o "$ROOT_OPT" -c "$ROOT_CHECK" > "$CPP_OUTPUT" 2>&1
    
    echo "C++ output saved to: $CPP_OUTPUT"
    echo "Baseline metrics:"
    echo ""
    
    # Extract and display key metrics
    echo "Number of patterns:"
    grep -i "unique site patterns" "$CPP_OUTPUT" || echo "  (not found in output)"
    echo ""
    
    echo "Log-likelihood values:"
    grep -i "log-likelihood" "$CPP_OUTPUT" || echo "  (not found in output)"
    echo ""
    
    echo -e "${YELLOW}Create the C implementation and re-run this script to validate against baseline.${NC}"
    exit 0
fi

# Both binaries exist - run full validation
print_header "Running C++ Implementation (Baseline)"
echo "Command: $CPP_BINARY -e ... > $CPP_OUTPUT"
"$CPP_BINARY" -e "$EDGE_LIST" -p "$PATTERNS" -x "$TAXON_ORDER" -b "$BASE_COMP" -o "$ROOT_OPT" -c "$ROOT_CHECK" > "$CPP_OUTPUT" 2>&1
echo "✓ C++ execution completed"
echo ""

print_header "Running C Implementation"
echo "Command: $C_BINARY -e ... > $C_OUTPUT"
"$C_BINARY" -e "$EDGE_LIST" -p "$PATTERNS" -x "$TAXON_ORDER" -b "$BASE_COMP" -o "$ROOT_OPT" -c "$ROOT_CHECK" > "$C_OUTPUT" 2>&1
echo "✓ C execution completed"
echo ""

print_header "Validating Basic Outputs"
echo ""

# Test 1: Check for number of patterns
((TOTAL_TESTS++))
PATTERNS_CPP=$(extract_value "$CPP_OUTPUT" "unique site patterns")
PATTERNS_C=$(extract_value "$C_OUTPUT" "unique site patterns")
if [ "$PATTERNS_CPP" = "$PATTERNS_C" ] && [ -n "$PATTERNS_CPP" ]; then
    echo -e "${GREEN}✓ PASS${NC}: Number of patterns matches (${PATTERNS_CPP})"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ FAIL${NC}: Number of patterns mismatch (C++: ${PATTERNS_CPP}, C: ${PATTERNS_C})"
    ((FAILED_TESTS++))
fi
echo "" >> "$VALIDATION_REPORT"
echo "Pattern Count: C++=$PATTERNS_CPP, C=$PATTERNS_C" >> "$VALIDATION_REPORT"

# Test 2: Check for non-root vertices
((TOTAL_TESTS++))
VERTICES_CPP=$(extract_value "$CPP_OUTPUT" "Number of non-root vertices")
VERTICES_C=$(extract_value "$C_OUTPUT" "Number of non-root vertices")
if [ "$VERTICES_CPP" = "$VERTICES_C" ] && [ -n "$VERTICES_CPP" ]; then
    echo -e "${GREEN}✓ PASS${NC}: Number of non-root vertices matches (${VERTICES_CPP})"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ FAIL${NC}: Number of vertices mismatch (C++: ${VERTICES_CPP}, C: ${VERTICES_C})"
    ((FAILED_TESTS++))
fi
echo "Vertex Count: C++=$VERTICES_CPP, C=$VERTICES_C" >> "$VALIDATION_REPORT"
echo ""

print_header "Validating Log-Likelihood Values (CRITICAL)"
echo ""

# Test 3: Pruning algorithm log-likelihood
((TOTAL_TESTS++))
LL_PRUNING_CPP=$(extract_value "$CPP_OUTPUT" "log-likelihood using pruning")
LL_PRUNING_C=$(extract_value "$C_OUTPUT" "log-likelihood using pruning")

if compare_floats "$LL_PRUNING_CPP" "$LL_PRUNING_C" "$TOLERANCE" "Log-likelihood (Pruning)"; then
    ((PASSED_TESTS++))
    echo "PASS: Pruning LL" >> "$VALIDATION_REPORT"
else
    ((FAILED_TESTS++))
    echo "FAIL: Pruning LL (C++=$LL_PRUNING_CPP, C=$LL_PRUNING_C)" >> "$VALIDATION_REPORT"
fi
echo ""

# Test 4: Optimized propagation log-likelihood
((TOTAL_TESTS++))
LL_PROP_OPT_CPP=$(extract_value "$CPP_OUTPUT" "log-likelihood using OPTIMIZED propagation")
LL_PROP_OPT_C=$(extract_value "$C_OUTPUT" "log-likelihood using OPTIMIZED propagation")

if compare_floats "$LL_PROP_OPT_CPP" "$LL_PROP_OPT_C" "$TOLERANCE" "Log-likelihood (Optimized Propagation)"; then
    ((PASSED_TESTS++))
    echo "PASS: Optimized Propagation LL" >> "$VALIDATION_REPORT"
else
    ((FAILED_TESTS++))
    echo "FAIL: Optimized Propagation LL (C++=$LL_PROP_OPT_CPP, C=$LL_PROP_OPT_C)" >> "$VALIDATION_REPORT"
fi
echo ""

# Test 5: Memoized propagation log-likelihood
((TOTAL_TESTS++))
LL_PROP_MEM_CPP=$(extract_value "$CPP_OUTPUT" "log-likelihood using MEMOIZED propagation")
LL_PROP_MEM_C=$(extract_value "$C_OUTPUT" "log-likelihood using MEMOIZED propagation")

if compare_floats "$LL_PROP_MEM_CPP" "$LL_PROP_MEM_C" "$TOLERANCE" "Log-likelihood (Memoized Propagation)"; then
    ((PASSED_TESTS++))
    echo "PASS: Memoized Propagation LL" >> "$VALIDATION_REPORT"
else
    ((FAILED_TESTS++))
    echo "FAIL: Memoized Propagation LL (C++=$LL_PROP_MEM_CPP, C=$LL_PROP_MEM_C)" >> "$VALIDATION_REPORT"
fi
echo ""

# Test 6: Verify propagation matches pruning (internal consistency check)
print_header "Internal Consistency Checks"
echo ""

((TOTAL_TESTS++))
if [ -n "$LL_PROP_OPT_C" ] && [ -n "$LL_PRUNING_C" ]; then
    if compare_floats "$LL_PRUNING_C" "$LL_PROP_OPT_C" "$TOLERANCE" "C Implementation: Pruning vs Propagation consistency"; then
        ((PASSED_TESTS++))
        echo "PASS: C internal consistency" >> "$VALIDATION_REPORT"
    else
        ((FAILED_TESTS++))
        echo "FAIL: C internal consistency (Pruning=$LL_PRUNING_C, Propagation=$LL_PROP_OPT_C)" >> "$VALIDATION_REPORT"
    fi
else
    echo -e "${YELLOW}⊘ SKIP${NC}: Internal consistency check (values not available)"
    ((TOTAL_TESTS--))
fi
echo ""

# Test 7: Check for EM convergence (if available)
print_header "EM Algorithm Validation"
echo ""

((TOTAL_TESTS++))
check_value_exists "$CPP_OUTPUT" "EM" "EM algorithm output (C++)"
CPP_EM_EXISTS=$?

check_value_exists "$C_OUTPUT" "EM" "EM algorithm output (C)"
C_EM_EXISTS=$?

if [ $CPP_EM_EXISTS -eq 0 ] && [ $C_EM_EXISTS -eq 0 ]; then
    # Both have EM output - compare final log-likelihood
    LL_EM_CPP=$(extract_value "$CPP_OUTPUT" "final.*log-likelihood\|EM.*iteration.*log-likelihood" | tail -1)
    LL_EM_C=$(extract_value "$C_OUTPUT" "final.*log-likelihood\|EM.*iteration.*log-likelihood" | tail -1)
    
    if compare_floats "$LL_EM_CPP" "$LL_EM_C" "$TOLERANCE" "Final EM Log-likelihood"; then
        ((PASSED_TESTS++))
        echo "PASS: EM final LL" >> "$VALIDATION_REPORT"
    else
        ((FAILED_TESTS++))
        echo "FAIL: EM final LL (C++=$LL_EM_CPP, C=$LL_EM_C)" >> "$VALIDATION_REPORT"
    fi
else
    echo -e "${YELLOW}⊘ SKIP${NC}: EM validation (not yet implemented in one or both versions)"
    ((TOTAL_TESTS--))
fi
echo ""

# Test 8: Memory leak check (if valgrind is available)
if command -v valgrind &> /dev/null; then
    print_header "Memory Leak Detection (C Implementation)"
    echo ""
    
    ((TOTAL_TESTS++))
    VALGRIND_OUTPUT="valgrind_output.txt"
    echo "Running valgrind on C implementation..."
    valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
             "$C_BINARY" -e "$EDGE_LIST" -p "$PATTERNS" -x "$TAXON_ORDER" \
             -b "$BASE_COMP" -o "$ROOT_OPT" -c "$ROOT_CHECK" > /dev/null 2> "$VALGRIND_OUTPUT"
    
    DEFINITELY_LOST=$(grep "definitely lost:" "$VALGRIND_OUTPUT" | grep -oE '[0-9]+' | head -1)
    INDIRECTLY_LOST=$(grep "indirectly lost:" "$VALGRIND_OUTPUT" | grep -oE '[0-9]+' | head -1)
    
    if [ "$DEFINITELY_LOST" = "0" ] && [ "$INDIRECTLY_LOST" = "0" ]; then
        echo -e "${GREEN}✓ PASS${NC}: No memory leaks detected"
        ((PASSED_TESTS++))
        echo "PASS: No memory leaks" >> "$VALIDATION_REPORT"
    else
        echo -e "${RED}✗ FAIL${NC}: Memory leaks detected"
        echo "  Definitely lost: $DEFINITELY_LOST bytes"
        echo "  Indirectly lost: $INDIRECTLY_LOST bytes"
        echo "  See $VALGRIND_OUTPUT for details"
        ((FAILED_TESTS++))
        echo "FAIL: Memory leaks (definitely=$DEFINITELY_LOST, indirectly=$INDIRECTLY_LOST)" >> "$VALIDATION_REPORT"
    fi
    echo ""
else
    echo -e "${YELLOW}⊘ SKIP${NC}: valgrind not available for memory leak detection"
fi

# Final summary
print_header "Validation Summary"
echo ""

echo "Total Tests:  $TOTAL_TESTS"
echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
echo ""

# Calculate percentage
if [ $TOTAL_TESTS -gt 0 ]; then
    PASS_PERCENT=$(awk -v p=$PASSED_TESTS -v t=$TOTAL_TESTS 'BEGIN {printf "%.1f", (p/t)*100}')
    echo "Pass Rate:    ${PASS_PERCENT}%"
fi
echo ""

# Write summary to report
echo "" >> "$VALIDATION_REPORT"
echo "Summary: $PASSED_TESTS/$TOTAL_TESTS tests passed ($PASS_PERCENT%)" >> "$VALIDATION_REPORT"

# Save full outputs
echo "" >> "$VALIDATION_REPORT"
echo "=== Full C++ Output ===" >> "$VALIDATION_REPORT"
cat "$CPP_OUTPUT" >> "$VALIDATION_REPORT"
echo "" >> "$VALIDATION_REPORT"
echo "=== Full C Output ===" >> "$VALIDATION_REPORT"
cat "$C_OUTPUT" >> "$VALIDATION_REPORT"

echo "Full validation report saved to: $VALIDATION_REPORT"
echo ""

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}   ✓ ALL TESTS PASSED - VALIDATION SUCCESSFUL${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "The C implementation produces identical results to the C++ baseline."
    echo "You may proceed to the next stage of conversion."
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}   ✗ VALIDATION FAILED - $FAILED_TESTS TEST(S) FAILED${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Please review the failures above and fix issues before proceeding."
    echo "Check $VALIDATION_REPORT for detailed comparison."
    exit 1
fi
