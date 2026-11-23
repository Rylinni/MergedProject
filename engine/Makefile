CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
BUILD := build
HEADERS := merged.hpp agents.hpp

all: $(BUILD)/merged_eval $(BUILD)/libmerged.so

$(BUILD)/merged_eval: eval.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread -o $@ eval.cpp

$(BUILD)/libmerged.so: capi.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ capi.cpp

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
