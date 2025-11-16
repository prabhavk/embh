# Makefile for EMBH C Implementation
# Stage-by-stage validation

CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra -pedantic -I./include
LDFLAGS = -lm

# CUDA settings
NVCC = nvcc
CUDA_FLAGS = -O2 -I./include
CUDA_LDFLAGS = -lcudart -lm

SRCDIR = src
TESTDIR = test
OBJDIR = obj
BINDIR = bin

# Source files (Stage 1-7)
SOURCES = $(SRCDIR)/embh_pattern.c \
          $(SRCDIR)/embh_vertex.c \
          $(SRCDIR)/embh_clique.c \
          $(SRCDIR)/embh_sem.c \
          $(SRCDIR)/embh_utils.c \
          $(SRCDIR)/embh_io.c \
          $(SRCDIR)/embh_pruning.c \
          $(SRCDIR)/embh_propagation.c \
          $(SRCDIR)/embh_em.c \
          $(SRCDIR)/embh_manager.c

OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# CUDA source
CUDA_SOURCES = $(SRCDIR)/embh_cuda.cu
CUDA_OBJECTS = $(OBJDIR)/embh_cuda.o

.PHONY: all clean directories test_pruning test_propagation test_em embh embh_cuda

all: directories embh

# Main EMBH executable (CPU only, no CUDA)
embh: directories $(OBJECTS) $(SRCDIR)/embh_main.c
	$(CC) $(CFLAGS) $(SRCDIR)/embh_main.c $(OBJECTS) -o $(BINDIR)/embh $(LDFLAGS)
	@echo "Built EMBH executable (CPU only): $(BINDIR)/embh"

# EMBH with CUDA support
# Note: We need to recompile embh_manager.c and embh_em.c with USE_CUDA defined
embh_cuda: directories $(filter-out $(OBJDIR)/embh_manager.o $(OBJDIR)/embh_em.o,$(OBJECTS)) $(OBJDIR)/embh_manager_cuda.o $(OBJDIR)/embh_em_cuda.o $(CUDA_OBJECTS) $(SRCDIR)/embh_main.c
	$(NVCC) $(CUDA_FLAGS) -DUSE_CUDA $(SRCDIR)/embh_main.c $(filter-out $(OBJDIR)/embh_manager.o $(OBJDIR)/embh_em.o,$(OBJECTS)) $(OBJDIR)/embh_manager_cuda.o $(OBJDIR)/embh_em_cuda.o $(CUDA_OBJECTS) -o $(BINDIR)/embh_cuda $(CUDA_LDFLAGS)
	@echo "Built EMBH executable (with CUDA): $(BINDIR)/embh_cuda"

# Compile CUDA source
$(OBJDIR)/embh_cuda.o: $(SRCDIR)/embh_cuda.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

# Compile embh_manager with CUDA support
$(OBJDIR)/embh_manager_cuda.o: $(SRCDIR)/embh_manager.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -DUSE_CUDA -c $< -o $@

# Compile embh_em with CUDA support
$(OBJDIR)/embh_em_cuda.o: $(SRCDIR)/embh_em.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -DUSE_CUDA -c $< -o $@

directories:
	@mkdir -p $(OBJDIR) $(BINDIR)

# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Test pruning algorithm (Stage 4 validation)
test_pruning: directories $(OBJECTS) $(TESTDIR)/test_pruning.c
	$(CC) $(CFLAGS) $(TESTDIR)/test_pruning.c $(OBJECTS) -o $(BINDIR)/test_pruning $(LDFLAGS)
	@echo "Running pruning algorithm test..."
	@echo ""
	./$(BINDIR)/test_pruning

# Test belief propagation algorithm (Stage 5 validation)
test_propagation: directories $(OBJECTS) $(TESTDIR)/test_propagation.c
	$(CC) $(CFLAGS) $(TESTDIR)/test_propagation.c $(OBJECTS) -o $(BINDIR)/test_propagation $(LDFLAGS)
	@echo "Running belief propagation test..."
	@echo ""
	./$(BINDIR)/test_propagation

# Test EM algorithm (Stage 6 validation)
test_em: directories $(OBJECTS) $(TESTDIR)/test_em.c
	$(CC) $(CFLAGS) $(TESTDIR)/test_em.c $(OBJECTS) -o $(BINDIR)/test_em $(LDFLAGS)
	@echo "Running EM algorithm test..."
	@echo ""
	./$(BINDIR)/test_em

# Clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Valgrind check for memory leaks
valgrind: directories $(OBJECTS) $(TESTDIR)/test_em.c
	$(CC) $(CFLAGS) $(TESTDIR)/test_em.c $(OBJECTS) -o $(BINDIR)/test_em $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
	         ./$(BINDIR)/test_em 2>&1 | tee valgrind_output.txt
	@echo ""
	@echo "Checking for memory leaks..."
	@grep "definitely lost:" valgrind_output.txt
	@grep "indirectly lost:" valgrind_output.txt

# All unit tests
test_all: test_pruning test_propagation test_em
	@echo "All tests completed"

# Comprehensive test suite (Stage 8)
test_suite: directories $(OBJECTS) $(TESTDIR)/test_suite.c
	$(CC) $(CFLAGS) $(TESTDIR)/test_suite.c $(OBJECTS) -o $(BINDIR)/test_suite $(LDFLAGS)
	@echo "Running comprehensive test suite..."
	@echo ""
	./$(BINDIR)/test_suite

# Valgrind on comprehensive test suite
valgrind_suite: directories $(OBJECTS) $(TESTDIR)/test_suite.c
	$(CC) $(CFLAGS) $(TESTDIR)/test_suite.c $(OBJECTS) -o $(BINDIR)/test_suite $(LDFLAGS)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
		./$(BINDIR)/test_suite 2>&1 | tee valgrind_suite_output.txt
	@echo ""
	@echo "Checking for memory leaks..."
	@grep "definitely lost:" valgrind_suite_output.txt || true
	@grep "indirectly lost:" valgrind_suite_output.txt || true
	@grep "ERROR SUMMARY:" valgrind_suite_output.txt

# Run full EMBH pipeline with test data
run: embh
	./$(BINDIR)/embh \
		-e data/RAxML_bipartitions.CDS_FcC_partition.edgelist \
		-p data/patterns_1000.pat \
		-x data/patterns_1000.taxon_order \
		-b data/patterns_1000.basecomp \
		-o h_0 -c h_5

# Memory check on full pipeline
valgrind_full: embh
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
		./$(BINDIR)/embh \
		-e data/RAxML_bipartitions.CDS_FcC_partition.edgelist \
		-p data/patterns_1000.pat \
		-x data/patterns_1000.taxon_order \
		-b data/patterns_1000.basecomp \
		-o h_0 -c h_5 2>&1 | tee valgrind_full_output.txt
	@echo ""
	@echo "Checking for memory leaks..."
	@grep "definitely lost:" valgrind_full_output.txt
	@grep "indirectly lost:" valgrind_full_output.txt
