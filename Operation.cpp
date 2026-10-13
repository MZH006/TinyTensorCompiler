#include "Operation.h"
#include "Topology.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

Operation::Operation(std::string operation_name,
    std::vector<Tensor*> op_inputs,
    std::vector<std::size_t> output_shape)
: output("output", output_shape) {
    name = operation_name;
    inputs = op_inputs;

    for (Tensor* input : inputs) {
        input->addConsumer(this);
    }

    output.setProducer(this);
}

void Operation::print() const {
    std::cout << name << "\n";
    std::cout << inputs.size() << "\n";
    output.print();
}

Tensor* Operation::getOutput() {
    return &output;
}

const std::string& Operation::getName() const {
    return name;
}

const std::vector<Tensor*>& Operation::getInputs() const {
    return inputs;
}

void Operation::replaceInput(Tensor* old_input, Tensor* new_input) {
    for (std::size_t i = 0; i < inputs.size(); i++) {
        if (new_input == old_input) {
            return;
        }
        if (inputs[i] == old_input) {
            new_input->addConsumer(this);
            old_input->removeConsumer(this);
            inputs[i] = new_input;
            topology::changed();
        }
    }
}

void Operation::disconnectInputs() {
    if (!inputs.empty()) {
        topology::changed();
    }
    for (Tensor* input : inputs) {
        input->removeConsumer(this);
    }
    inputs.clear();
}
