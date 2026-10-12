CXX := clang++
CXXFLAGS := -std=c++17 -Wall -Wextra

TARGET := main
CORE_SOURCES := Tensor.cpp Operation.cpp Ops.cpp Optimizer.cpp Graph.cpp
CORE_OBJECTS := $(CORE_SOURCES:.cpp=.o)
OBJECTS := main.o $(CORE_OBJECTS)
TEST_TARGET := tests
TEST_OBJECTS := tests.o $(CORE_OBJECTS)
DEPS := $(sort $(OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d))

.PHONY: all run test clean

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

clean:
	rm -f $(TARGET) $(TEST_TARGET) $(OBJECTS) tests.o $(DEPS)

-include $(DEPS)
