# EMBH C++ to C Conversion - Master Documentation Index

## 📚 Documentation Overview

This directory contains a complete, systematic plan for converting the EMBH (EM Barry-Hartigan) phylogenetic analysis software from C++ to C, with the goal of enabling CUDA parallelization.

---

## 🎯 Start Here

### If you want to understand the overall approach:
**Read:** `CPP_TO_C_CONVERSION_PLAN.md`
- Comprehensive 10-stage conversion plan
- Detailed technical specifications for each stage
- Validation checkpoints throughout
- CUDA parallelization strategy
- ~15-23 day timeline estimate

### If you want to start converting RIGHT NOW:
**Read:** `QUICK_START_GUIDE.md`
- Get baseline running in 5 minutes
- First Claude Code session prompt ready to copy-paste
- Common pitfalls and solutions
- Testing strategy
- Success indicators

### If you're using Claude Code in VS Code:
**Read:** `CLAUDE_CODE_INSTRUCTIONS.md`
- Stage-by-stage prompts for Claude Code
- Specific validation commands
- Troubleshooting guide
- Quick reference commands

### If you want to track your progress:
**Use:** `CONVERSION_VALIDATION_CHECKLIST.md`
- Checkbox for every deliverable
- Validation criteria for each stage
- Space to record metrics and blockers
- Progress tracking template

---

## 📖 Document Descriptions

### 1. CPP_TO_C_CONVERSION_PLAN.md
**Purpose:** Master technical document
**Contains:**
- 10 conversion stages with detailed specifications
- Data structure conversions (C++ classes → C structs)
- Algorithm implementations (pruning, propagation, EM)
- Build system setup
- CUDA preparation refactoring
- Quality metrics and success criteria

**Use this when:**
- Planning the overall conversion
- Understanding technical requirements
- Designing data structures
- Estimating effort

**Key sections:**
- Stage 1: Core Data Structures
- Stage 4: Pruning Algorithm (first critical milestone)
- Stage 5: Belief Propagation (most complex)
- Stage 9: CUDA Preparation
- CUDA Parallelization Strategy

### 2. CLAUDE_CODE_INSTRUCTIONS.md
**Purpose:** AI-assisted conversion guide
**Contains:**
- Ready-to-use prompts for each conversion stage
- Specific file references from the C++ codebase
- Validation commands to run after each step
- Troubleshooting common issues
- Quick reference for validation

**Use this when:**
- Working with Claude Code in VS Code
- Need specific prompts for each stage
- Want validation commands ready
- Debugging issues

**Key sections:**
- Stage-by-stage prompts
- Validation commands
- Troubleshooting guide
- Success criteria summary

### 3. CONVERSION_VALIDATION_CHECKLIST.md
**Purpose:** Progress tracking and validation
**Contains:**
- Checkbox for every deliverable
- Spaces to record actual values and metrics
- Validation criteria that must be met
- Blocker tracking
- Time tracking

**Use this when:**
- Tracking conversion progress
- Ensuring no steps are skipped
- Recording validation results
- Identifying blockers

**Key sections:**
- Stage-by-stage checklists
- Validation checkboxes
- Metrics recording spaces
- Milestone markers
- Final validation summary

### 4. QUICK_START_GUIDE.md
**Purpose:** Get started immediately
**Contains:**
- 5-minute baseline setup
- First Claude Code session prompt
- Critical success criteria
- Common pitfalls and solutions
- Testing strategies
- Success indicators

**Use this when:**
- Starting the conversion for the first time
- Need a quick overview
- Want to validate approach early
- Looking for practical advice

**Key sections:**
- 5-minute getting started
- First Claude Code session
- Common pitfalls
- Testing strategy
- Success indicators

---

## 🎓 Recommended Reading Order

### For First-Time Readers:
1. **QUICK_START_GUIDE.md** (15 min)
   - Get oriented
   - Understand the goal
   - Run baseline test

2. **CPP_TO_C_CONVERSION_PLAN.md** - Skim (30 min)
   - Understand the 10 stages
   - See the big picture
   - Appreciate the scope

3. **CLAUDE_CODE_INSTRUCTIONS.md** - Stage 0 (15 min)
   - Establish baseline
   - Extract metrics
   - Ready for Stage 1

4. **CONVERSION_VALIDATION_CHECKLIST.md** (10 min)
   - Print or keep open
   - Check off Stage 0
   - Ready to track progress

**Total reading time:** ~70 minutes
**Ready to convert!** 🚀

### For Project Managers:
1. **CPP_TO_C_CONVERSION_PLAN.md** - Executive Summary
   - Timeline: 15-23 days
   - Key risks and mitigations
   - Success criteria

2. **CONVERSION_VALIDATION_CHECKLIST.md** - Milestones
   - 6 major milestones
   - Validation gates
   - Progress tracking

### For Developers:
1. **QUICK_START_GUIDE.md** - Full read
2. **CLAUDE_CODE_INSTRUCTIONS.md** - Stage by stage
3. **CPP_TO_C_CONVERSION_PLAN.md** - Deep dive on current stage
4. **CONVERSION_VALIDATION_CHECKLIST.md** - After each stage

---

## 🗺️ Conversion Roadmap

```
Stage 0: Baseline (1 hour)
    └─> Run C++ test, extract metrics
         └─> CHECKPOINT: Baseline established

Stage 1: Data Structures (1-2 days)
    ├─> 1.1: Pattern & PackedPatternStorage
    ├─> 1.2: SEM_vertex  
    ├─> 1.3: Clique & CliqueTree
    └─> 1.4: Main SEM structure
         └─> MILESTONE 1: All structs defined ✓

Stage 2: Utilities (0.5 days)
    └─> Matrix ops, string utils, DNA conversion
         └─> CHECKPOINT: Utils validated

Stage 3: File I/O (1 day)
    └─> Read patterns, topology, base composition
         └─> MILESTONE 2: Can load all data ✓

Stage 4: Pruning Algorithm (2-3 days)
    └─> Felsenstein's algorithm
         └─> MILESTONE 3: First algorithm works! ✓
              └─> CRITICAL: Log-likelihood matches baseline

Stage 5: Belief Propagation (3-4 days)
    └─> Pearl's algorithm, optimized & memoized
         └─> MILESTONE 4: Both algorithms work ✓
              └─> CRITICAL: Both match each other & baseline

Stage 6: EM Algorithm (2-3 days)
    └─> Expectation-Maximization with Aitken
         └─> MILESTONE 5: Complete C implementation ✓

Stage 7: Manager & Main (1 day)
    └─> Top-level workflow
         └─> CHECKPOINT: Full pipeline runs

Stage 8: Build & Test (1-2 days)
    └─> Makefile, test suite, validation
         └─> CHECKPOINT: All tests pass

Stage 9: CUDA Prep (2-3 days)
    └─> SoA conversion, parallel regions
         └─> MILESTONE 6: CUDA-ready! ✓

Stage 10: Final Validation (1 day)
    └─> Multi-dataset, memory, docs
         └─> PROJECT COMPLETE ✓
              └─> Ready for CUDA implementation!
```

---

## ✅ Success Criteria

### Technical Validation
- [ ] All log-likelihoods match C++ baseline (< 1e-10 difference)
- [ ] EM converges identically
- [ ] Zero memory leaks (valgrind clean)
- [ ] Performance within 10% of C++

### Quality Metrics
- [ ] All code documented
- [ ] Build system robust
- [ ] Test coverage comprehensive
- [ ] Error handling implemented

### CUDA Readiness
- [ ] Data structures optimized (SoA where beneficial)
- [ ] Parallel regions identified and documented
- [ ] Memory layout optimized for GPU
- [ ] Parallelization strategy clear

---

## 🎯 Key Milestones

### Milestone 1: Data Structures Complete
**What:** All C++ classes converted to C structs
**Validation:** Unit tests pass, no memory leaks
**Time:** ~1-2 days
**Ready for:** File I/O implementation

### Milestone 2: Data Loading Works
**What:** Can read all input files and build tree
**Validation:** Same tree structure as C++, pattern count matches
**Time:** ~3-4 days total
**Ready for:** Algorithm implementation

### Milestone 3: First Algorithm Works
**What:** Pruning algorithm computes correct log-likelihood
**Validation:** Matches C++ baseline to 10+ decimal places
**Time:** ~6-7 days total
**Ready for:** Second algorithm
**CRITICAL:** This proves the approach works!

### Milestone 4: Both Algorithms Work
**What:** Pruning and propagation both correct
**Validation:** Both match each other and baseline
**Time:** ~10-11 days total
**Ready for:** EM optimization

### Milestone 5: Complete Implementation
**What:** Full pipeline including EM works
**Validation:** All outputs match C++ baseline
**Time:** ~13-14 days total
**Ready for:** CUDA preparation

### Milestone 6: CUDA-Ready Code
**What:** Refactored for optimal GPU parallelization
**Validation:** Still correct, structure optimized
**Time:** ~16-17 days total
**Ready for:** CUDA kernel development!

---

## 🚨 Critical Checkpoints

These are MUST-PASS validation points. Do not proceed if these fail:

### Checkpoint 1: Baseline Established
- C++ version runs successfully
- Output captured completely
- Key metrics extracted
- **If this fails:** Fix C++ build issues first

### Checkpoint 2: Pattern Storage Correct
- 3-bit packing matches C++ bit-for-bit
- Memory usage exactly as expected
- No leaks
- **If this fails:** Review C++ PackedPatternStorage carefully

### Checkpoint 3: Tree Structure Matches
- Same number of vertices
- Same topology
- Same edges
- **If this fails:** Debug file reading carefully

### Checkpoint 4: First Log-Likelihood Matches
- Difference < 1e-10 from baseline
- Works on full 1000-pattern dataset
- **If this fails:** STOP and debug thoroughly
  - This proves the approach works
  - Don't proceed until this passes

### Checkpoint 5: Propagation Matches Pruning
- Two independent algorithms give same result
- Both match baseline
- **If this fails:** Review clique tree construction

### Checkpoint 6: EM Converges
- Monotonically increasing log-likelihood
- Final value matches baseline
- **If this fails:** Check expected statistics computation

---

## 📊 Expected Timeline

### Conservative Estimate (23 days)
- Stage 0: 1 hour
- Stage 1: 2 days
- Stage 2: 1 day
- Stage 3: 1 day
- Stage 4: 3 days
- Stage 5: 4 days
- Stage 6: 3 days
- Stage 7: 1 day
- Stage 8: 2 days
- Stage 9: 3 days
- Stage 10: 1 day
- **Buffer:** 2 days
- **Total:** 23 days

### Aggressive Estimate (15 days)
- With AI assistance
- Experienced C programmer
- Minimal debugging needed
- **Total:** 15 days

### Realistic Estimate (18-20 days)
- Some debugging time
- Learning as you go
- Good AI assistance
- **Total:** 18-20 days

---

## 🛠️ Tools Required

### Essential
- GCC or Clang (C99 support)
- Make
- Git (for version control)
- Text editor / IDE

### Highly Recommended
- Valgrind (memory debugging)
- GDB (debugging)
- Python (for validation scripts)

### Optional
- gprof (profiling)
- massif (memory profiling)
- cppcheck (static analysis)

---

## 📚 Additional Resources

### In This Directory
- `CPP_TO_C_CONVERSION_PLAN.md` - Master plan
- `CLAUDE_CODE_INSTRUCTIONS.md` - AI-assisted conversion
- `CONVERSION_VALIDATION_CHECKLIST.md` - Progress tracking
- `QUICK_START_GUIDE.md` - Getting started
- `README_CONVERSION.md` - This file!

### External References
- C99 Standard
- CUDA Programming Guide
- Valgrind Documentation
- RAxML-NG source code (reference phylogenetic implementation)

---

## 💡 Pro Tips

### 1. Validate Early and Often
Don't write 1000 lines then test. Test after every function.

### 2. Match C++ Exactly
When in doubt, do exactly what C++ does. Optimize later.

### 3. Use Intermediate Prints
Compare intermediate values with C++ during debugging.

### 4. Document as You Go
Future you will thank present you.

### 5. Version Control Everything
Commit after each working stage.

### 6. Don't Skip Validation
Each checkpoint exists for a reason.

---

## 🎉 You're Ready!

With these documents, you have:
- ✅ Complete conversion plan
- ✅ Stage-by-stage instructions
- ✅ Validation criteria
- ✅ Progress tracking
- ✅ Quick start guide
- ✅ Troubleshooting help

**Next steps:**
1. Read QUICK_START_GUIDE.md (15 minutes)
2. Run baseline test (5 minutes)
3. Start Stage 1.1 with Claude Code
4. Check off each item in the validation checklist

**Good luck with your conversion!** 🚀

Remember: The goal isn't just working C code - it's **correct, validated, CUDA-ready C code** that will enable massive speedups through GPU parallelization.

You've got this! 💪

---

## 📞 Support

If you get stuck:
1. Check the troubleshooting sections in QUICK_START_GUIDE.md
2. Review the relevant stage in CPP_TO_C_CONVERSION_PLAN.md
3. Use the debugging prompts in CLAUDE_CODE_INSTRUCTIONS.md
4. Compare with C++ code carefully

**Most important:** Don't skip validation steps. If something doesn't match the baseline, debug it before moving on!

---

**Last updated:** November 2024
**Version:** 1.0
**Status:** Ready for conversion
