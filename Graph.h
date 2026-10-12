#pragma once

#include "Operation.h"

#include <memory>
#include <vector>

class Graph {
public:
    void setOutput(Tensor* tensor) {
        output = tensor;
    }

    Tensor* getOutput() const {
        return output;
    }

    Operation* addOperation(std::unique_ptr<Operation> operation);

private:
    std::vector<std::unique_ptr<Operation>> operations;
    Tensor* output = nullptr;
};