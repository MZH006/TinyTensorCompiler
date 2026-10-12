#pragma once

#include "Operation.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

class Graph;

struct FusionPattern {
    std::string name;
    std::vector<std::string> chain;
    std::function<std::unique_ptr<Operation>(const std::vector<Operation*>&)> build;

    bool match(const std::vector<Operation*>& order, std::size_t i) const;
};

std::unique_ptr<Operation> buildLinearReLUPattern(
    const std::vector<Operation*>& ops);

std::unique_ptr<Operation> buildAddReLUPattern(
    const std::vector<Operation*>& ops);

void getExecutionOrder(
    Tensor* tensor,
    std::unordered_set<Operation*>& visited,
    std::vector<Operation*>& order);

void redirectConsumers(Tensor* old_output, Tensor* new_output);

std::unique_ptr<Operation> applyPattern(
    Graph& graph,
    const FusionPattern& pattern,
    std::vector<Operation*>& order,
    std::size_t i);

void optimize(Graph& graph, const std::vector<FusionPattern>& patterns);
