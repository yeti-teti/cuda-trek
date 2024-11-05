# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3

# Source files and executable name
SRC = train.cpp
EXEC = lstm_model

# Build the executable
$(EXEC): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(EXEC) $(SRC)

# Clean up the build
clean:
	rm -f $(EXEC)

