#pragma once

#include "Operation.h"

#include <memory>
#include <vector>

class Graph {
public:
    void setOutput(Tensor* tensor) {
        output = tensor;
        planValid = false;
    }

    Tensor* getOutput() const {
        return output;
    }

    void replaceOutput(Tensor* old_output, Tensor* new_output) {
        if (output == old_output) {
            output = new_output;
            planValid = false;
        }
    }

    std::vector<Operation*> getExecutionOrder() const;
    void execute();

    std::size_t getPlanBuildCount() const { return planBuildCount; }

    Operation* addOperation(std::unique_ptr<Operation> operation);

private:
    std::vector<std::unique_ptr<Operation>> operations;
    Tensor* output = nullptr;
    void prepareExecutionPlan();
    std::vector<Operation*> executionPlan;
    bool planValid = false;
    std::size_t planRevision = 0;
    std::size_t planBuildCount = 0;
};
