CXX := clang++
PYTHON ?= python3
CXXFLAGS := -std=c++17 -Wall -Wextra

TARGET := main
CORE_SOURCES := Tensor.cpp Operation.cpp Ops.cpp Optimizer.cpp Graph.cpp Codegen.cpp
CORE_OBJECTS := $(CORE_SOURCES:.cpp=.o)
OBJECTS := main.o $(CORE_OBJECTS)
TEST_TARGET := tests
TEST_OBJECTS := tests.o $(CORE_OBJECTS)
DEPS := $(sort $(OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d) linear_runner.d graph_import.d)

.PHONY: all run test test-ubsan test-asan test-pytorch test-import test-fx test-codegen benchmark clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	./$(TARGET)

$(TEST_TARGET): $(TEST_OBJECTS)
	$(CXX) $(LDFLAGS) $(TEST_OBJECTS) $(LDLIBS) -o $@

test: $(TEST_TARGET)
	./$(TEST_TARGET)
	@echo "All tests passed."

linear-runner: linear_runner.o $(CORE_OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

graph-import: graph_import.o $(CORE_OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

test-import: graph-import
	$(PYTHON) scripts/test_graph_import.py ./graph-import

test-fx: graph-import
	$(PYTHON) scripts/validate_fx.py ./graph-import

test-codegen: graph-import
	$(PYTHON) scripts/validate_codegen.py --runner ./graph-import --cxx "$(CXX)"

test-pytorch: linear-runner
	$(PYTHON) scripts/validate_pytorch.py --runner ./linear-runner

# Build separately so sanitizer flags cannot mix with normal object files.
tests-ubsan: tests.cpp $(CORE_SOURCES) $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer tests.cpp $(CORE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test-ubsan: tests-ubsan
	./tests-ubsan

# Linux CI exercises this target; the macOS ASan runtime may fail to initialize.
tests-asan: tests.cpp $(CORE_SOURCES) $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -fsanitize=address -fno-omit-frame-pointer tests.cpp $(CORE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test-asan: tests-asan
	./tests-asan

# Compile every source with optimization; do not reuse debug/default objects.
benchmark-bin: benchmark.cpp $(CORE_SOURCES) $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O3 benchmark.cpp $(CORE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

benchmark: benchmark-bin
	./benchmark-bin

clean:
	rm -f $(TARGET) $(TEST_TARGET) tests-ubsan tests-asan benchmark-bin linear-runner graph-import linear_runner.o graph_import.o $(OBJECTS) tests.o $(DEPS)

-include $(DEPS)
