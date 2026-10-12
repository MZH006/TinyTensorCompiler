#include "Graph.h"
#include "Ops.h"
#include "Optimizer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <cassert>

#ifdef NDEBUG
#error "Tests require assertions; build without -DNDEBUG"
#endif
#include <iostream>
#include <memory>
#include <vector> 


void testFinalOutputFusion() {
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

void testBranchPreventsFusion() {
    Tensor x("x", {-2.0f, 3.0f}, {1, 2});
    Tensor y("y", {1.0f, 2.0f}, {1, 2});
    Graph graph;
    Operation* add = graph.addOperation(std::make_unique<Add>(&x, &y));
    Operation* left = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    Operation* right = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    Operation* final = graph.addOperation(
        std::make_unique<Add>(left->getOutput(), right->getOutput()));
    graph.setOutput(final->getOutput());

    graph.execute();
    const std::vector<float> original = graph.getOutput()->getData();
    const std::vector<float> expected{0.0f, 10.0f};
    assert(original == expected);

    const std::vector<FusionPattern> patterns{
        {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
        {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
    };
    optimize(graph, patterns);

    // The shared Add must remain available to both ReLU branches.
    assert(graph.getExecutionOrder().size() == 4);
    assert(add->getOutput()->getConsumers().size() == 2);
    assert(left->getInputs()[0] == add->getOutput());
    assert(right->getInputs()[0] == add->getOutput());
    assert(graph.getOutput() == final->getOutput());

    graph.execute();
    assert(graph.getOutput()->getData() == original);
    assert(graph.getOutput()->getData() == expected);
}

void testLinearFusionRectangular() {
    Tensor x("x", {1, 2, 3, 4, 5, 6}, {2, 3});
    Tensor weights("weights", {1, 0, 0, 1, 1, 1}, {3, 2});
    Tensor bias("bias", {-5, 1, 0, -12}, {2, 2});
    Graph graph;
    Operation* matmul = graph.addOperation(std::make_unique<MatMul>(&x, &weights));
    Operation* add = graph.addOperation(
        std::make_unique<Add>(matmul->getOutput(), &bias));
    Operation* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    graph.setOutput(relu->getOutput());

    graph.execute();
    const std::vector<float> original = graph.getOutput()->getData();
    const std::vector<float> expected{0, 6, 10, 0};
    assert(original == expected);

    const std::vector<FusionPattern> patterns{
        {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
        {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
    };
    optimize(graph, patterns);

    assert(graph.getOutput()->getProducer()->getName() == "FusedLinearReLU");
    assert(graph.getOutput()->getShape() == std::vector<std::size_t>({2, 2}));
    assert(graph.getExecutionOrder().size() == 1);

    graph.execute();
    assert(graph.getOutput()->getData() == original);
    assert(graph.getOutput()->getData() == expected);
}


// Shared helpers keep the individual tests focused on their behavior.
std::vector<FusionPattern> fusionPatterns() {
    return {
        {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
        {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
    };
}

template<typename Exception, typename Action>
void expectThrows(Action action) {
    bool thrown = false;
    try {
        action();
    } catch (const Exception&) {
        thrown = true;
    }
    assert(thrown);
}

void assertClose(const std::vector<float>& actual,
                 const std::vector<float>& expected) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); i++) {
        const float tolerance = 1e-5f + 1e-4f * std::abs(expected[i]);
        assert(std::isfinite(actual[i]));
        assert(std::abs(actual[i] - expected[i]) <= tolerance);
    }
}

// Check edge multiplicity: Add(x, x) records two uses, not one.
void assertGraphLinks(Graph& graph) {
    for (Operation* op : graph.getExecutionOrder()) {
        assert(op->getOutput()->getProducer() == op);
        for (Tensor* input : op->getInputs()) {
            const auto uses = std::count(op->getInputs().begin(), op->getInputs().end(), input);
            const auto registrations = std::count(
                input->getConsumers().begin(), input->getConsumers().end(), op);
            assert(uses == registrations);
        }
        for (Operation* consumer : op->getOutput()->getConsumers()) {
            assert(std::find(consumer->getInputs().begin(), consumer->getInputs().end(),
                             op->getOutput()) != consumer->getInputs().end());
        }
    }
}

void testTensorValidationAndMissingData() {
    expectThrows<std::invalid_argument>([] {
        Tensor invalid("invalid", {1, 2}, {2, 2});
    });
    Tensor pending("pending", {1, 2});
    assert(pending.getProducer() == nullptr);
    assert(pending.getConsumers().empty());
    expectThrows<std::runtime_error>([&] { pending.getData(); });
    pending.setData({3, -4});
    expectThrows<std::invalid_argument>([&] { pending.setData({9}); });
    assert(pending.getData() == std::vector<float>({3, -4}));
}

void testInvalidShapesLeaveNoConsumers() {
    Tensor a("a", {1, 2});
    Tensor different("different", {2, 1});
    expectThrows<std::invalid_argument>([&] { Add invalid(&a, &different); });
    expectThrows<std::invalid_argument>([&] { FusedAddReLU invalid(&a, &different); });
    Tensor rank_one("rank_one", {2});
    Tensor rank_three("rank_three", {1, 2, 1});
    expectThrows<std::invalid_argument>([&] { MatMul invalid(&rank_one, &a); });
    expectThrows<std::invalid_argument>([&] { MatMul invalid(&a, &rank_three); });
    Tensor wrong_inner("wrong_inner", {3, 2});
    expectThrows<std::invalid_argument>([&] { MatMul invalid(&a, &wrong_inner); });
    Tensor weights("weights", {2, 2});
    expectThrows<std::invalid_argument>([&] { FusedLinearReLU invalid(&a, &weights, &different); });
    expectThrows<std::invalid_argument>([&] { FusedLinearReLU invalid(&rank_one, &weights, &a); });
    for (Tensor* tensor : {&a, &different, &rank_one, &rank_three, &wrong_inner, &weights}) {
        assert(tensor->getConsumers().empty());
    }
}

void testGraphErrorsAndInputOnlyOutput() {
    Graph graph;
    expectThrows<std::runtime_error>([&] { graph.getExecutionOrder(); });
    expectThrows<std::runtime_error>([&] { graph.execute(); });
    expectThrows<std::invalid_argument>([&] { graph.addOperation(nullptr); });
    Tensor input("input", {2, -3}, {1, 2});
    graph.setOutput(&input);
    assert(graph.getExecutionOrder().empty());
    optimize(graph, fusionPatterns());
    graph.execute();
    assert(graph.getOutput() == &input);
    assert(graph.getOutput()->getData() == std::vector<float>({2, -3}));
}

void testExecutionRejectsMissingInputData() {
    Tensor x("x", {1, 2});
    Tensor y("y", {1, 2}, {1, 2});
    Graph graph;
    auto* add = graph.addOperation(std::make_unique<Add>(&x, &y));
    graph.setOutput(add->getOutput());
    expectThrows<std::runtime_error>([&] { graph.execute(); });
    expectThrows<std::runtime_error>([&] { graph.getOutput()->getData(); });
    x.setData({3, 4});
    graph.execute();
    assert(graph.getOutput()->getData() == std::vector<float>({4, 6}));
}

void testFusionWithRepeatedInputs() {
    Tensor x("x", {-2, 3}, {1, 2});
    Graph graph;
    auto* add = graph.addOperation(std::make_unique<Add>(&x, &x));
    auto* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    graph.setOutput(relu->getOutput());
    assert(x.getConsumers().size() == 2);
    // Rewriting must work before any original operation executes.
    optimize(graph, fusionPatterns());
    auto* fused = graph.getOutput()->getProducer();
    assert(fused->getName() == "FusedAddReLU");
    assert(graph.getExecutionOrder().size() == 1);
    assert(fused->getInputs()[0] == &x && fused->getInputs()[1] == &x);
    assert(x.getConsumers().size() == 2);
    assert(x.getConsumers()[0] == fused && x.getConsumers()[1] == fused);
    assert(add->getInputs().empty() && relu->getInputs().empty());
    assertGraphLinks(graph);
    graph.execute();
    assert(graph.getOutput()->getData() == std::vector<float>({0, 6}));
}

void testInputReplacementAndDisconnection() {
    Tensor x("x", {1, 2}, {1, 2});
    Tensor y("y", {3, 4}, {1, 2});
    Add add(&x, &x);
    add.replaceInput(&x, &x); // Self-replacement is a no-op.
    assert(x.getConsumers().size() == 2);
    add.replaceInput(&y, &x); // Absent old input is a no-op.
    assert(x.getConsumers().size() == 2);
    add.replaceInput(&x, &y);
    assert(x.getConsumers().empty());
    assert(y.getConsumers().size() == 2);
    assert(add.getInputs() == std::vector<Tensor*>({&y, &y}));
    y.removeConsumer(nullptr); // Removing an absent registration is harmless.
    assert(y.getConsumers().size() == 2);
    add.execute();
    assert(add.getOutput()->getData() == std::vector<float>({6, 8}));
    add.disconnectInputs();
    add.disconnectInputs();
    assert(add.getInputs().empty() && y.getConsumers().empty());
}

void testMatcherBoundariesAndDisconnectedNodes() {
    Tensor x("x", {1, 2}, {1, 2});
    Graph graph;
    auto* add = graph.addOperation(std::make_unique<Add>(&x, &x));
    auto* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    auto* unrelated = graph.addOperation(std::make_unique<ReLU>(&x));
    graph.setOutput(relu->getOutput());
    auto pattern = fusionPatterns()[1];
    auto order = graph.getExecutionOrder();
    assert(pattern.match(order, 0));
    assert(!pattern.match(order, 1));
    assert(!pattern.match(order, order.size()));
    assert(!pattern.match(order, std::numeric_limits<std::size_t>::max()));
    assert(!pattern.match({}, 0));
    FusionPattern empty{"empty", {}, buildAddReLUPattern};
    assert(!empty.match(order, 0));
    assert(!pattern.match({add, unrelated}, 0));
    assert(!pattern.match({relu, add}, 0));
    Tensor* old_output = graph.getOutput();
    auto replacement = applyPattern(graph, pattern, order, 1);
    assert(!replacement && graph.getOutput() == old_output);
    assert(add->getInputs().size() == 2 && order.size() == 2);
    assertGraphLinks(graph);
}

void testRedirectAllConsumersAndRepeatedUses() {
    Tensor x("x", {-2, 3}, {1, 2});
    Graph graph;
    auto* add = graph.addOperation(std::make_unique<Add>(&x, &x));
    auto* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    auto* repeated = graph.addOperation(std::make_unique<Add>(relu->getOutput(), relu->getOutput()));
    auto* other = graph.addOperation(std::make_unique<ReLU>(relu->getOutput()));
    auto* final = graph.addOperation(std::make_unique<Add>(repeated->getOutput(), other->getOutput()));
    graph.setOutput(final->getOutput());
    graph.execute();
    const auto original = graph.getOutput()->getData();
    std::vector<Operation*> pair{add, relu};
    auto owner = applyPattern(graph, fusionPatterns()[1], pair, 0);
    assert(owner);
    auto* fused = graph.addOperation(std::move(owner));
    assert(relu->getOutput()->getConsumers().empty());
    assert(fused->getOutput()->getConsumers().size() == 3);
    assert(repeated->getInputs()[0] == fused->getOutput());
    assert(repeated->getInputs()[1] == fused->getOutput());
    assert(other->getInputs()[0] == fused->getOutput());
    assert(graph.getOutput() == final->getOutput());
    assertGraphLinks(graph);
    graph.execute();
    assert(graph.getOutput()->getData() == original);
    assert(original == std::vector<float>({0, 18}));
}

void testMultipleMatchesAndRepeatedOptimization() {
    Tensor x("x", {-2, 3}, {1, 2});
    Tensor y("y", {1, 2}, {1, 2});
    Graph graph;
    auto* first = graph.addOperation(std::make_unique<Add>(&x, &y));
    auto* first_relu = graph.addOperation(std::make_unique<ReLU>(first->getOutput()));
    auto* second = graph.addOperation(std::make_unique<Add>(first_relu->getOutput(), &y));
    auto* final = graph.addOperation(std::make_unique<ReLU>(second->getOutput()));
    graph.setOutput(final->getOutput());
    graph.execute();
    const auto original = graph.getOutput()->getData();
    optimize(graph, fusionPatterns());
    assert(graph.getExecutionOrder().size() == 2);
    for (auto* op : graph.getExecutionOrder()) assert(op->getName() == "FusedAddReLU");
    assertGraphLinks(graph);
    auto* output = graph.getOutput();
    optimize(graph, fusionPatterns());
    assert(graph.getOutput() == output && graph.getExecutionOrder().size() == 2);
    graph.execute();
    assert(graph.getOutput()->getData() == original);
    x.setData({10, -10});
    graph.execute();
    assert(graph.getOutput()->getData() == std::vector<float>({12, 2}));
}

void testNoPatternsAndNoMatchingPatterns() {
    Tensor x("x", {-2, 3}, {1, 2});
    Graph graph;
    auto* relu = graph.addOperation(std::make_unique<ReLU>(&x));
    graph.setOutput(relu->getOutput());
    optimize(graph, {});
    optimize(graph, fusionPatterns());
    assert(graph.getOutput() == relu->getOutput());
    assert(graph.getExecutionOrder().size() == 1);
    graph.execute();
    assert(graph.getOutput()->getData() == std::vector<float>({0, 3}));
}

void testGraphOwnershipKeepsAddressesStable() {
    Tensor x("x", {1, 2}, {1, 2});
    Graph graph;
    auto* first = graph.addOperation(std::make_unique<ReLU>(&x));
    Operation* last = first;
    for (int i = 0; i < 128; i++) {
        last = graph.addOperation(std::make_unique<ReLU>(last->getOutput()));
    }
    graph.setOutput(last->getOutput());
    assert(graph.getExecutionOrder().front() == first);
    assert(graph.getExecutionOrder().size() == 129);
    assertGraphLinks(graph);
    graph.execute();
    assert(graph.getOutput()->getData() == x.getData());
}

void testZeroSizedDimensions() {
    Tensor x("x", std::vector<float>{}, {2, 0});
    Tensor weights("weights", std::vector<float>{}, {0, 3});
    Tensor bias("bias", {-1, 0, 2, 3, -4, 5}, {2, 3});
    Graph graph;
    auto* mm = graph.addOperation(std::make_unique<MatMul>(&x, &weights));
    auto* add = graph.addOperation(std::make_unique<Add>(mm->getOutput(), &bias));
    auto* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
    graph.setOutput(relu->getOutput());
    graph.execute();
    const auto original = graph.getOutput()->getData();
    assert(original == std::vector<float>({0, 0, 2, 3, 0, 5}));
    optimize(graph, fusionPatterns());
    graph.execute();
    assert(graph.getOutput()->getData() == original);
}

void testLinearFusionAgainstReference() {
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    for (int trial = 0; trial < 32; trial++) {
        const std::size_t M = 1 + trial % 4;
        const std::size_t K = 1 + trial % 5;
        const std::size_t N = 1 + trial % 3;
        std::vector<float> a(M * K), b(K * N), bias_values(M * N);
        for (float& value : a) value = distribution(generator);
        for (float& value : b) value = distribution(generator);
        for (float& value : bias_values) value = distribution(generator);
        // Independent double-precision oracle with a different loop ordering.
        std::vector<double> sums(M * N, 0.0);
        for (std::size_t k = 0; k < K; k++) {
            for (std::size_t i = 0; i < M; i++) {
                for (std::size_t j = 0; j < N; j++) {
                    sums[i * N + j] += static_cast<double>(a[i * K + k]) * b[k * N + j];
                }
            }
        }
        std::vector<float> expected(M * N);
        for (std::size_t i = 0; i < expected.size(); i++) {
            expected[i] = static_cast<float>(std::max(0.0, sums[i] + bias_values[i]));
        }
        Tensor x("x", a, {M, K});
        Tensor weights("weights", b, {K, N});
        Tensor bias("bias", bias_values, {M, N});
        Graph graph;
        auto* mm = graph.addOperation(std::make_unique<MatMul>(&x, &weights));
        // Alternate operand order to verify commutative Add matching/building.
        auto* add = graph.addOperation(trial % 2
            ? std::make_unique<Add>(&bias, mm->getOutput())
            : std::make_unique<Add>(mm->getOutput(), &bias));
        auto* relu = graph.addOperation(std::make_unique<ReLU>(add->getOutput()));
        graph.setOutput(relu->getOutput());
        graph.execute();
        assertClose(graph.getOutput()->getData(), expected);
        const auto original = graph.getOutput()->getData();
        auto patterns = fusionPatterns();
        // Exercise both full linear fusion and shorter-rule fallback.
        if (trial % 3 == 0) patterns.erase(patterns.begin());
        optimize(graph, patterns);
        assert(graph.getExecutionOrder().size() == (trial % 3 == 0 ? 2u : 1u));
        assertGraphLinks(graph);
        graph.execute();
        assertClose(graph.getOutput()->getData(), expected);
        assertClose(graph.getOutput()->getData(), original);
    }
}

int main() {
    int passed = 0;
    auto run = [&](const char* name, void (*test)()) {
        test();
        std::cout << "PASS " << name << "\n";
        passed++;
    };
    run("final output fusion", testFinalOutputFusion);
    run("branch prevents fusion", testBranchPreventsFusion);
    run("rectangular linear fusion", testLinearFusionRectangular);
    run("tensor validation and missing data", testTensorValidationAndMissingData);
    run("invalid shapes preserve graph links", testInvalidShapesLeaveNoConsumers);
    run("graph errors and input-only output", testGraphErrorsAndInputOnlyOutput);
    run("execution rejects missing input data", testExecutionRejectsMissingInputData);
    run("fusion with repeated inputs", testFusionWithRepeatedInputs);
    run("input replacement and disconnection", testInputReplacementAndDisconnection);
    run("matcher boundaries and disconnected nodes", testMatcherBoundariesAndDisconnectedNodes);
    run("redirect all consumers and repeated uses", testRedirectAllConsumersAndRepeatedUses);
    run("multiple matches and repeated optimization", testMultipleMatchesAndRepeatedOptimization);
    run("no patterns and no matches", testNoPatternsAndNoMatchingPatterns);
    run("ownership keeps addresses stable", testGraphOwnershipKeepsAddressesStable);
    run("zero-sized dimensions", testZeroSizedDimensions);
    run("32 numerical reference cases", testLinearFusionAgainstReference);
    std::cout << passed << " tests passed.\n";
}
