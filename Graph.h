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

    void replaceOutput(Tensor* old_output, Tensor* new_output) {
        if (output == old_output) {
            output = new_output;
        }
    }

    std::vector<Operation*> getExecutionOrder() const;
    void execute();

    Operation* addOperation(std::unique_ptr<Operation> operation);

private:
    std::vector<std::unique_ptr<Operation>> operations;
    Tensor* output = nullptr;
};
