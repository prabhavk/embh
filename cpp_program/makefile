# Makefile (no embh_em.cpp dependency)

# Compiler and flags
CXX = g++

DEV ?= 1

ifeq ($(DEV),1)
  BASE_CXXFLAGS = -std=c++11 -O0 -g3 -fPIC -fno-omit-frame-pointer # Fast compilation
else
  BASE_CXXFLAGS = -std=c++11 -O2 -fPIC     # Optimized
endif

# Toggle warnings (set to 0 to disable warnings)
WARNINGS ?= 0

# Warning flags
ifeq ($(WARNINGS),1)
  WARN_FLAGS = -Wall
else
  WARN_FLAGS = -w
endif

CXXFLAGS = $(BASE_CXXFLAGS) $(WARN_FLAGS)
INCLUDES = -I. 
LDFLAGS = -shared
LIBS =

# Directories
OBJDIR = obj
BINDIR = bin
LIBDIR = lib
DATADIR = data

# Targets
TARGET     = $(BINDIR)/embh
SHARED_LIB = $(LIBDIR)/libembh.so

# Sources (library has ONLY embh_core.cpp)
LIB_SOURCES  = embh_core.cpp
MAIN_SOURCES = embh.cpp

# Objects
LIB_OBJECTS  = $(LIB_SOURCES:%.cpp=$(OBJDIR)/%.o)
MAIN_OBJECTS = $(MAIN_SOURCES:%.cpp=$(OBJDIR)/%.o)
ALL_OBJECTS  = $(LIB_OBJECTS) $(MAIN_OBJECTS)

# Default
.PHONY: all
all: directories $(SHARED_LIB) $(TARGET)

# Dirs
.PHONY: directories
directories:
	@mkdir -p $(OBJDIR) $(BINDIR) $(LIBDIR)

# Shared library (from embh_core only)
$(SHARED_LIB): $(LIB_OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Shared library built: $(SHARED_LIB)"

# Executable (links to libembh)
$(TARGET): $(MAIN_OBJECTS) $(SHARED_LIB)
	$(CXX) -o $@ $(MAIN_OBJECTS) -L$(LIBDIR) -lembh -Wl,-rpath,'$$ORIGIN/../$(LIBDIR)'
	@echo "Build successful: $(TARGET)"

# Compile objects
$(OBJDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(BINDIR) $(LIBDIR)
	@echo "Cleaned build artifacts"

# Rebuild
.PHONY: rebuild
rebuild: clean all

# Fast rebuild of main
.PHONY: main
main: $(OBJDIR)/embh.o $(SHARED_LIB)
	$(CXX) -o $(TARGET) $(OBJDIR)/embh.o -L$(LIBDIR) -lembh -Wl,-rpath,'$$ORIGIN/../$(LIBDIR)'
	@echo "Rebuilt main: $(TARGET)"

# Library-only rebuild
.PHONY: lib
lib: directories
	@$(MAKE) $(SHARED_LIB)

# Run
.PHONY: run
run: $(TARGET)
	cd $(BINDIR) && ./embh

.Phony: test
test: 
	$(TARGET) -e $(DATADIR)/RAxML_bipartitions.CDS_FcC_partition.edgelist -f $(DATADIR)/CDS_FcC_1000.fas -p $(DATADIR)/patterns_1000.pat -x $(DATADIR)/patterns_1000.taxon_order -b $(DATADIR)/patterns_1000.basecomp -o h_0 -c h_5

.Phony: test_2k
test_2k: 
	$(TARGET) -e $(DATADIR)/RAxML_bipartitions.CDS_FcC_partition.edgelist -f $(DATADIR)/CDS_FcC_2000.fas -p $(DATADIR)/patterns_2000.pat -x $(DATADIR)/patterns_2000.taxon_order -b $(DATADIR)/patterns_2000.basecomp -o h_0 -c h_15

# Debug build
.PHONY: debug
debug: BASE_CXXFLAGS = -std=c++11 -g -fPIC -DDEBUG
debug: clean all

# Build without warnings
.PHONY: nowarn
nowarn: WARNINGS = 0
nowarn: clean all

# ldd helper
.PHONY: ldd
ldd: $(TARGET)
	ldd $(TARGET)

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all        - Build shared library (embh_core) and executable (embh)"
	@echo "  main       - Rebuild only main executable"
	@echo "  lib        - Rebuild only shared library"
	@echo "  clean      - Remove build artifacts"
	@echo "  rebuild    - Clean and build"
	@echo "  debug      - Build with debug symbols"
	@echo "  nowarn     - Build without warnings"
	@echo "  run        - Build and run"
	@echo "  ldd        - Show library dependencies"
	@echo ""
	@echo "Toggle warnings:"
	@echo "  make WARNINGS=0  # Build without warnings (default: 0)"
	@echo "  make WARNINGS=1  # Build with warnings"

# Dependency tracking
-include $(ALL_OBJECTS:.o=.d)

$(OBJDIR)/%.d: %.cpp
	@mkdir -p $(OBJDIR)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -MM -MT '$(OBJDIR)/$*.o' $< > $@
