#include "Graph.h"
#include "Ops.h"
#include "Optimizer.h"
#include <cassert>
#include <iostream>
#include <memory>
#include <vector> 


int main() {
    Tensor x("x", {1.0f, 2.0f}, {1, 2});
    Tensor bias("bias", {1.0f, -20.0f}, {1, 2});
    Graph graph;
    Operation* add = graph.addOperation(std::make_unique<Add>(&x, &bias));
    Operation* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    graph.setOutput(relu->getOutput());


    graph.execute();
    const std::vector<float> original = graph.getOutput()->getData();
    assert(original == std::vector<float>({2.0f, 0.0f}));

    const std::vector<FusionPattern> patterns{
        {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
        {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
    };

    optimize(graph, patterns);

    assert(graph.getOutput()->getProducer()->getName() == "FusedAddReLU");
    assert(graph.getExecutionOrder().size() == 1);

    graph.execute();
    assert(graph.getOutput()->getData() == original);

}