CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra
BUILD := build
HEADERS := merged.hpp agents.hpp ntuple.hpp

all: $(BUILD)/merged_eval $(BUILD)/merged_train $(BUILD)/maxmerge $(BUILD)/libmerged.so

$(BUILD)/maxmerge: maxmerge.cpp merged.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ maxmerge.cpp

$(BUILD)/merged_train: train.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread -o $@ train.cpp

$(BUILD)/merged_eval: eval.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -pthread -o $@ eval.cpp

$(BUILD)/libmerged.so: capi.cpp $(HEADERS) | $(BUILD)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ capi.cpp

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
