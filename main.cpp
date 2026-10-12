#include "Ops.h"
#include "Optimizer.h"

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>
#include "Graph.h"
#include <utility>


int main() {
    Tensor x("x", {1.0f, 2.0f}, {1, 2});
    Tensor weights("weights", {3.0f, 4.0f, 5.0f, 6.0f}, {2, 2});
    Tensor bias("bias", {1.0f, -20.0f}, {1, 2});

    Graph graph;

    Operation* matmul = graph.addOperation(
        std::make_unique<MatMul>(&x, &weights));

    Operation* add = graph.addOperation(
        std::make_unique<Add>(matmul->getOutput(), &bias));

    Operation* relu = graph.addOperation(
        std::make_unique<ReLU>(add->getOutput()));

    Operation* downstream = graph.addOperation(
        std::make_unique<ReLU>(relu->getOutput()));

    graph.setOutput(downstream->getOutput());

    std::unordered_set<Operation*> visited;
    std::vector<Operation*> order;
    getExecutionOrder(graph.getOutput(), visited, order);

    std::cout << "Original order: ";
    for (Operation* op : order) {
        std::cout << op->getName() << " ";
        op->execute();
    }
    std::cout << "\n";
    // Copy the original result before execution overwrites the final tensor.
    const std::vector<float> original = graph.getOutput()->getData();

    std::vector<FusionPattern> patterns{
        {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
        {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
    };
    // Own every replacement for as long as the optimized graph is used.
    std::vector<std::unique_ptr<Operation>> replacements;
    optimize(order, patterns, replacements);

    for (auto& replacement : replacements) {
        graph.addOperation(std::move(replacement));
    }
    replacements.clear();

    visited.clear();
    order.clear();
    getExecutionOrder(graph.getOutput(), visited, order);
    std::cout << "Optimized order: ";
    for (Operation* op : order) {
        std::cout << op->getName() << " ";
        op->execute();
    }
    std::cout << "\nOriginal result: ";
    for (float value : original) {
        std::cout << value << " ";
    }
    std::cout << "\nOptimized result: ";
    for (float value : graph.getOutput()->getData()) {
        std::cout << value << " ";
    }
    const bool outputs_match = original == graph.getOutput()->getData();
    std::cout << "\nOutputs match: " << outputs_match << "\n";
    return outputs_match ? 0 : 1;
}
