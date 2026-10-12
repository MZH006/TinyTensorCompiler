#include "Graph.h"
#include "Optimizer.h"

#include <stdexcept>
#include <unordered_set>
#include <utility>

Operation* Graph::addOperation(std::unique_ptr<Operation> operation) {
    if (!operation) {
        throw std::invalid_argument("Cannot add a null operation to Graph");
    }
    Operation* pointer = operation.get();
    operations.push_back(std::move(operation));
    return pointer;
}

std::vector<Operation*> Graph::getExecutionOrder() const {
    if (!output) {
        throw std::runtime_error("Graph has no output");
    }
    std::unordered_set<Operation*> visited;
    std::vector<Operation*> order;
    ::getExecutionOrder(output, visited, order);
    return order;
}

void Graph::execute() {
    for (Operation* operation : getExecutionOrder()) {
        operation->execute();
    }
}
