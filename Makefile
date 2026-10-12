CXX := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra

TARGET := main
CORE_SOURCES := Tensor.cpp Operation.cpp Ops.cpp Optimizer.cpp Graph.cpp
CORE_OBJECTS := $(CORE_SOURCES:.cpp=.o)
OBJECTS := main.o $(CORE_OBJECTS)
TEST_TARGET := tests
TEST_OBJECTS := tests.o $(CORE_OBJECTS)
DEPS := $(sort $(OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d))

.PHONY: all run test test-ubsan benchmark clean

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

# Build separately so sanitizer flags cannot mix with normal object files.
tests-ubsan: tests.cpp $(CORE_SOURCES) $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -g -fsanitize=undefined -fno-sanitize-recover=all -fno-omit-frame-pointer tests.cpp $(CORE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test-ubsan: tests-ubsan
	./tests-ubsan

# Compile every source with optimization; do not reuse debug/default objects.
benchmark-bin: benchmark.cpp $(CORE_SOURCES) $(wildcard *.h) Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -O3 benchmark.cpp $(CORE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

benchmark: benchmark-bin
	./benchmark-bin

clean:
	rm -f $(TARGET) $(TEST_TARGET) tests-ubsan benchmark-bin $(OBJECTS) tests.o $(DEPS)

-include $(DEPS)
