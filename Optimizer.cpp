#include "Optimizer.h"
#include "Ops.h"

#include <algorithm>
#include <utility>

bool FusionPattern::match(const std::vector<Operation*>& order, std::size_t i) const {
    if (chain.empty() || i > order.size() || chain.size() > order.size() - i) {
        return false;
    }

    for (std::size_t j = 0; j < chain.size(); j++) {
        Operation *op = order[i+j];

        if  ( (*op).getName() != chain[j] ) {
            return false;
        }

        if (j>0) {
            Operation *prev = order[i+j-1];
            if ( (std::find(op->getInputs().begin(), op->getInputs().end(), prev->getOutput())) == op->getInputs().end()) {
                return false;
            }
        }

        if (j < chain.size() - 1) {
            if (op->getOutput()->getConsumers().size() != 1) {
                return false;
            }
        }
    }

    return true;

}

std::unique_ptr<Operation> buildLinearReLUPattern(
    const std::vector<Operation*>& ops) {
    Operation* matmul = ops[0];
    Operation* add = ops[1];
    // Add is commutative: locate the input other than the MatMul output.
    Tensor* bias = add->getInputs()[0] == matmul->getOutput()
    ? add->getInputs()[1] : add->getInputs()[0];
    return std::make_unique<FusedLinearReLU>(
        matmul->getInputs()[0], matmul->getInputs()[1], bias);
}

std::unique_ptr<Operation> buildAddReLUPattern(
    const std::vector<Operation*>& ops) {
    return std::make_unique<FusedAddReLU>(
        ops[0]->getInputs()[0], ops[0]->getInputs()[1]);
}

void getExecutionOrder(
    Tensor* tensor,
    std::unordered_set<Operation*>& visited,
    std::vector<Operation*>& order) {
    if (tensor->getProducer() == nullptr) {
        return;
    }

    Operation* op = tensor->getProducer();
    if (visited.count(op)) {
        return;
    }
    visited.insert(op);

    for (Tensor* tensor : op->getInputs()){
        getExecutionOrder(tensor,visited,order);
    }
    order.push_back(op);
}

void redirectConsumers(Tensor* old_output, Tensor* new_output) {
    auto consumers = old_output->getConsumers();
    for (auto consumer: consumers) {
        consumer->replaceInput(old_output, new_output);
    }
}

std::unique_ptr<Operation> applyPattern(
    const FusionPattern& pattern,
    std::vector<Operation*>& order,
    std::size_t i) {
    if (!pattern.match(order, i)) {
        return nullptr;
    }

    std::vector<Operation*> matched;
    for (std::size_t j = 0; j < pattern.chain.size(); j++) {
        matched.push_back(order[i + j]);
    }

    auto replacement = pattern.build(matched);

    redirectConsumers(matched.back()->getOutput(), replacement->getOutput());

    for (Operation *op: matched) {
        op->disconnectInputs();
    }

    auto first = order.begin() + i;
    auto last = first + matched.size();

    auto position = order.erase(first, last);
    order.insert(position, replacement.get());

    return replacement;
}

void optimize(std::vector<Operation*>& order, const std::vector<FusionPattern>& patterns, std::vector<std::unique_ptr<Operation>>& replacements) {

    std::size_t i = 0;
    while (i < order.size()) {
        bool matched = false;

        for (const FusionPattern& pattern : patterns) {
            auto replacement = applyPattern(pattern, order, i);

            if (replacement) {
                replacements.push_back(std::move(replacement));
                matched = true;
                break;
            }
        }

        if (!matched) {
            i++;
        }
    }
}
