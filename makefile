# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
LDFLAGS = 

# Target executable name
TARGET = embh

# Source files
SRCS = embh.cpp embh_classes.cpp

# Object files (automatically derived from source files)
OBJS = $(SRCS:.cpp=.o)

# Header files (for dependency tracking)
HEADERS = embh_classes.hpp

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Rebuild everything from scratch
rebuild: clean all

# Run the program
run: $(TARGET)
	./$(TARGET)

# Test the program
test: $(TARGET)
	./$(TARGET) -e ../data/RAxML_bipartitions.CDS_FcC_partition.edgelist -f ../data/CDS_FcC_1000.fas -p ../data/patterns_1000.pat -b ../data/patterns_1000.basecomp -r h_0 -t h_5

# Phony targets (not actual files)
.PHONY: all clean rebuild run test