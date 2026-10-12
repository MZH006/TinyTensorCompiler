#include "Graph.h"
#include "Ops.h"
#include "Optimizer.h"

#include <iostream>
#include <memory>
#include <vector>

int main() {
    // Inputs outlive Graph; Graph owns all operations and their outputs.
    Tensor x("x", {1.0f, 2.0f}, {1, 2});
    Tensor weights("weights", {3.0f, 4.0f, 5.0f, 6.0f}, {2, 2});
    Tensor bias("bias", {1.0f, -20.0f}, {1, 2});
    Graph graph;
    Operation* matmul = graph.addOperation(std::make_unique<MatMul>(&x, &weights));
    Operation* add = graph.addOperation(std::make_unique<Add>(matmul->getOutput(), &bias));
    Operation* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    graph.setOutput(relu->getOutput());

    std::cout << "Original order: ";
    for (Operation* op : graph.getExecutionOrder()) {
        std::cout << op->getName() << " ";
    }
    graph.execute();
    const std::vector<float> original = graph.getOutput()->getData();

    const std::vector<FusionPattern> patterns{
        {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
        {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
    };
    optimize(graph, patterns);

    std::cout << "\nOptimized order: ";
    for (Operation* op : graph.getExecutionOrder()) {
        std::cout << op->getName() << " ";
    }
    graph.execute();
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
