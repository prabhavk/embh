================================================================================
PROPAGATION ALGORITHM COMPARISON FOR EMBH
================================================================================

This package adds functionality to compare log-likelihood computation using
two different algorithms: Pruning (Felsenstein) and Propagation (Belief Propagation).

FILES INCLUDED:
---------------
1. CODE_SNIPPETS.cpp         - Ready-to-copy code snippets
2. INTEGRATION_GUIDE.md      - Step-by-step integration instructions  
3. propagation_implementation.cpp - Full documented implementation
4. SUMMARY.md                - Quick reference and overview
5. README_PROPAGATION.txt    - This file

QUICK START:
-----------
1. Open CODE_SNIPPETS.cpp
2. Copy the 5 code sections into your files as marked
3. Compile: make clean && make all
4. Run: make test

WHAT YOU GET:
------------
- Automatic comparison of pruning vs propagation algorithms
- Validation that both produce identical results
- Performance timing for both methods
- Clear pass/fail output

EXPECTED OUTPUT:
---------------
==================================================================
COMPARING PRUNING AND PROPAGATION ALGORITHMS ON PATTERNS
==================================================================

[1/2] Running PRUNING ALGORITHM...
  ✓ Complete
  Log-likelihood: -19310.0633113404
  Time:           0.0234 seconds

[2/2] Running PROPAGATION ALGORITHM...
  ✓ Complete  
  Log-likelihood: -19310.0633113404
  Time:           0.0456 seconds

✓ PASSED - Algorithms agree within tolerance
==================================================================

SUPPORT:
-------
For detailed instructions, see INTEGRATION_GUIDE.md
For code reference, see CODE_SNIPPETS.cpp
For technical details, see SUMMARY.md

================================================================================
