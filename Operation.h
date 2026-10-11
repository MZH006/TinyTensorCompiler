#pragma once

#include "Tensor.h"

#include <string>
#include <vector>

class Operation{
public:
    Operation(std::string operation_name,
        std::vector<Tensor*> op_inputs,
        std::vector<std::size_t> output_shape);

    void print() const;

    Tensor* getOutput();

    const std::string& getName() const;

    const std::vector<Tensor*>& getInputs() const;

    void replaceInput(Tensor* old_input, Tensor* new_input);

    void disconnectInputs();

    virtual void execute() = 0;
    virtual ~Operation() = default;
private:
    std::string name;
    std::vector<Tensor*> inputs;
    Tensor output;
};
