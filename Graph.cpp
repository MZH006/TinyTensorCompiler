#include "Graph.h"
#include <utility>

Operation* Graph::addOperation(
    std::unique_ptr<Operation> operation) {
    // Save operation.get() in a raw pointer.
    // Move operation into the operations vector.
    // Return the saved pointer.

    Operation* pointer = operation.get();
    operations.push_back(std::move(operation));
    return pointer;
}