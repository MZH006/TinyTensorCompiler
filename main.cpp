#include "Ops.h"
#include "Optimizer.h"

#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

int main() {
    Tensor x("x", {1.0f, 2.0f}, {1, 2});
    Tensor weights("weights", {3.0f, 4.0f, 5.0f, 6.0f}, {2, 2});
    Tensor bias("bias", {1.0f, -20.0f}, {1, 2});

    MatMul matmul(&x, &weights);
    Add add(matmul.getOutput(), &bias);
    ReLU relu(add.getOutput());
    // Keep a downstream user to demonstrate redirecting the fused output.
    ReLU downstream(relu.getOutput());

    std::unordered_set<Operation*> visited;
    std::vector<Operation*> order;
    getExecutionOrder(downstream.getOutput(), visited, order);

    std::cout << "Original order: ";
    for (Operation* op : order) {
        std::cout << op->getName() << " ";
        op->execute();
    }
    std::cout << "\n";
    // Copy the original result before execution overwrites the final tensor.
    const std::vector<float> original = downstream.getOutput()->getData();

    std::vector<FusionPattern> patterns{
        {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
        {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
    };
    // Own every replacement for as long as the optimized graph is used.
    std::vector<std::unique_ptr<Operation>> replacements;
    optimize(order, patterns, replacements);

    visited.clear();
    order.clear();
    getExecutionOrder(downstream.getOutput(), visited, order);
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
    for (float value : downstream.getOutput()->getData()) {
        std::cout << value << " ";
    }
    const bool outputs_match = original == downstream.getOutput()->getData();
    std::cout << "\nOutputs match: " << outputs_match << "\n";
    return outputs_match ? 0 : 1;
}
