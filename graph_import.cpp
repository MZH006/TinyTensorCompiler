#include "Graph.h"
#include "Ops.h"
#include "Optimizer.h"
#include "Codegen.h"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

int main(int argc, char* argv[]) {
    try {
        if (argc < 2){
            throw std::runtime_error("Usage: graph-import <graph-file> [input-file ...]");
        }
        const bool generate = std::string(argv[1]) == "--emit-cpp";
        if (generate && argc != 4) {
            throw std::runtime_error("Usage: graph-import --emit-cpp <graph-file> <output.cpp>");
        }
        const char* graphFilename = argv[generate ? 2 : 1];
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
        // Inputs outlive Graph; moving unique_ptrs preserves tensor addresses.
        std::vector<std::unique_ptr<Tensor>> tensors;
        Graph graph;

        std::unordered_map<std::string, Tensor*> values;
        std::unordered_map<std::string, Tensor*> inputs;

        auto lookup = [&](const std::string& name) -> Tensor* {
            auto it = values.find(name);
            if (it == values.end()) {
                throw std::runtime_error("Unknown tensor: " + name);
            }
            return it->second;
        };

        std::ifstream program(graphFilename);
        if (!program) {
            throw std::runtime_error("Could not open " + std::string(graphFilename));
        }

        std::size_t lineNumber = 0;
        std::string line;
        while (std::getline(program, line)) {
            ++lineNumber;
            try {
                std::istringstream stream(line);
                std::string opcode;
                if (!(stream >> opcode)) {
                    continue;
                }
                if (opcode == "input") {
                    std::string name;
                    long long rows, columns;
                    if (!(stream >> name >> rows >> columns) || rows <= 0 || columns <= 0) {
                        throw std::runtime_error("Expected: input name positive-rows positive-columns");
                    }
                    if (values.find(name) != values.end()) {
                        throw std::runtime_error("Duplicate input name: " + name);
                    }
                    const auto m = static_cast<std::size_t>(rows);
                    const auto n = static_cast<std::size_t>(columns);
                    if (m > std::numeric_limits<std::size_t>::max() / n) {
                        throw std::runtime_error("Input shape is too large");
                    }

                    std::string extra;
                    if (stream >> extra) {
                        throw std::runtime_error("Unexpected extra argument: " + extra);
                    }

                    auto tensor = std::make_unique<Tensor>(name, std::vector<std::size_t>{m, n});

                    values.emplace(name, tensor.get());
                    inputs.emplace(name, tensor.get());
                    tensors.push_back(std::move(tensor));

                }

                else if (opcode == "tensor") {
                    std::string name;
                    long long rows, columns;
                    if (!(stream >> name >> rows >> columns) || rows <= 0 || columns <= 0) {
                        throw std::runtime_error("Expected: tensor name positive-rows positive-columns data...");
                    }
                    if (values.find(name) != values.end()) {
                        throw std::runtime_error("Duplicate tensor name: " + name);
                    }
                    const auto m = static_cast<std::size_t>(rows);
                    const auto n = static_cast<std::size_t>(columns);
                    if (m > std::numeric_limits<std::size_t>::max() / n) {
                        throw std::runtime_error("Tensor shape is too large");
                    }
                    std::vector<float> data;
                    for (std::size_t i = 0; i < m * n; ++i) {
                        float value;
                        if (!(stream >> value) || !std::isfinite(value)) {
                            throw std::runtime_error("Expected rows * columns finite tensor values");
                        }
                        data.push_back(value);
                    }
                    std::string extra;
                    if (stream >> extra) {
                        throw std::runtime_error("Unexpected extra argument: " + extra);
                    }
                    auto tensor = std::make_unique<Tensor>(name, std::move(data),
                                                          std::vector<std::size_t>{m, n});
                    values.emplace(name, tensor.get());
                    tensors.push_back(std::move(tensor));
                } else if (opcode == "add") {
                    std::string outputName, lhsName, rhsName;
                    if (!(stream >> outputName >> lhsName >> rhsName)) {
                        throw std::runtime_error("Expected: add output lhs rhs");
                    }
                    std::string extra;
                    if (stream >> extra) {
                        throw std::runtime_error("Unexpected extra argument: " + extra);
                    }
                    if (values.find(outputName) != values.end()) {
                        throw std::runtime_error("Duplicate tensor name: " + outputName);
                    }
                    auto* add = graph.addOperation(
                        std::make_unique<Add>(lookup(lhsName), lookup(rhsName)));
                    values.emplace(outputName, add->getOutput());
                } else if (opcode == "matmul") {
                    std::string outputName, lhsName, rhsName;
                    if (!(stream >> outputName >> lhsName >> rhsName)) {
                        throw std::runtime_error("Expected: matmul output lhs rhs");
                    }
                    std::string extra;
                    if (stream >> extra) {
                        throw std::runtime_error("Unexpected extra argument: " + extra);
                    }
                    if (values.find(outputName) != values.end()) {
                        throw std::runtime_error("Duplicate tensor name: " + outputName);
                    }
                    auto* matmul = graph.addOperation(
                        std::make_unique<MatMul>(lookup(lhsName), lookup(rhsName)));
                    values.emplace(outputName, matmul->getOutput());
                } else if (opcode == "relu") {
                    std::string outputName, inputName;
                    if (!(stream >> outputName >> inputName)) {
                        throw std::runtime_error("Expected: relu output input");
                    }
                    std::string extra;
                    if (stream >> extra) {
                        throw std::runtime_error("Unexpected extra argument: " + extra);
                    }
                    if (values.find(outputName) != values.end()) {
                        throw std::runtime_error("Duplicate tensor name: " + outputName);
                    }
                    auto* relu = graph.addOperation(
                        std::make_unique<ReLU>(lookup(inputName)));
                    values.emplace(outputName, relu->getOutput());
                } else if (opcode == "output") {
                    if (graph.getOutput() != nullptr) {
                        throw std::runtime_error("Only one output instruction is allowed");
                    }
                    std::string outputName;
                    if (!(stream >> outputName)) {
                        throw std::runtime_error("Expected: output name");
                    }
                    std::string extra;
                    if (stream >> extra) {
                        throw std::runtime_error("Unexpected extra argument: " + extra);
                    }
                    graph.setOutput(lookup(outputName));
                } else {
                    throw std::runtime_error("Unknown opcode: " + opcode);
                }
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    "Line " + std::to_string(lineNumber) + ": " + error.what());
            }
        }

        if (program.bad()) {
            throw std::runtime_error("Failed to read graph file");
        }
        if (graph.getOutput() == nullptr) {
            throw std::runtime_error("Missing output instruction");
        }

        if (generate) {
            const std::vector<FusionPattern> patterns{
                {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern}
            };
            optimize(graph, patterns);
            const auto order = graph.getExecutionOrder();
            if (order.size() != 1 || inputs.size() != 1 ||
                order.front()->getInputs().empty() ||
                inputs.begin()->second != order.front()->getInputs()[0]) {
                throw std::runtime_error("Codegen requires exactly one runtime input: linear x");
            }
            std::ostringstream source;
            emitCpp(graph, source);
            std::error_code pathError;
            if (std::filesystem::weakly_canonical(argv[3]) ==
                    std::filesystem::weakly_canonical(graphFilename) ||
                std::filesystem::equivalent(argv[3], graphFilename, pathError)) {
                throw std::runtime_error("Source output must differ from model file");
            }
            std::ofstream destination(argv[3]);
            destination << source.str();
            destination.close();
            if (!destination) throw std::runtime_error("Failed to write generated source");
            std::cout << "Generated " << argv[3] << "\n";
            return 0;
        }

        auto loadInputs = [&](const char* filename) {
            std::unordered_set<std::string> suppliedInputs;
            std::ifstream inputFile(filename);
            if (!inputFile) {
                throw std::runtime_error("Could not open " + std::string(filename));
            }
            std::string inputLine;
            while (std::getline(inputFile, inputLine)) {
                std::istringstream stream(inputLine);
                std::string name;

                if (!(stream >> name)) {
                    continue;
                }

                auto it = inputs.find(name);
                if (it == inputs.end()) {
                    throw std::runtime_error("Unknown input: " + name);
                }
                if (suppliedInputs.find(name) != suppliedInputs.end()) {
                    throw std::runtime_error("Duplicate input data: " + name);
                }

                Tensor* tensor = it->second;
                const auto& shape = tensor->getShape();
                const std::size_t count = shape[0] * shape[1];

                std::vector<float> data;
                for (std::size_t i = 0; i < count; ++i) {
                    float value;
                    if (!(stream >> value) || !std::isfinite(value)) {
                        throw std::runtime_error(
                            "Expected " + std::to_string(count)
                            + " finite values for input: " + name);
                    }
                    data.push_back(value);
                }
                std::string extra;
                if (stream >> extra) {
                    throw std::runtime_error(
                        "Unexpected extra argument for input " + name + ": " + extra);
                }
                tensor->setData(std::move(data));
                suppliedInputs.insert(name);
            }
            if (inputFile.bad()) {
                throw std::runtime_error("Failed to read input file");
            }
            for (const auto& entry : inputs) {
                if (suppliedInputs.find(entry.first) == suppliedInputs.end()) {
                    throw std::runtime_error("Missing input data: " + entry.first);
                }
            }
        };

        if (!inputs.empty()) {
            if (argc == 2) {
                throw std::runtime_error("Missing input data: " + inputs.begin()->first);
            }
            const auto originalCount = graph.getExecutionOrder().size();
            const std::vector<FusionPattern> patterns{
                {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
                {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
            };
            optimize(graph, patterns);
            std::cout << "Operations: " << originalCount << " -> "
                      << graph.getExecutionOrder().size() << "\n";
            for (int fileIndex = 2; fileIndex < argc; ++fileIndex) {
                loadInputs(argv[fileIndex]);
                graph.execute();
                std::cout << "Input file: " << argv[fileIndex] << "\nResult: ";
                for (float value : graph.getOutput()->getData()) {
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("Nonfinite execution result");
                    }
                    std::cout << value << " ";
                }
                std::cout << "\n";
            }
            return 0;
        }
        if (argc > 2) {
            throw std::runtime_error("Input files supplied but model declares no inputs");
        }

        const auto originalCount = graph.getExecutionOrder().size();
        graph.execute();
        const std::vector<float> original = graph.getOutput()->getData();

        const std::vector<FusionPattern> patterns{
            {"linear_relu", {"MatMul", "Add", "ReLU"}, buildLinearReLUPattern},
            {"add_relu", {"Add", "ReLU"}, buildAddReLUPattern}
        };
        optimize(graph, patterns);
        graph.execute();
        const auto& optimized = graph.getOutput()->getData();

        std::cout << "Operations: " << originalCount << " -> "
                  << graph.getExecutionOrder().size() << "\n";
        std::cout << "Optimized order: ";
        for (Operation* op : graph.getExecutionOrder()) {
            std::cout << op->getName() << " ";
        }
        std::cout << "\nOriginal result: ";
        for (float value : original) {
            std::cout << value << " ";
        }
        std::cout << "\nOptimized result: ";
        for (float value : optimized) {
            std::cout << value << " ";
        }

        bool matches = original.size() == optimized.size();
        for (std::size_t i = 0; matches && i < original.size(); ++i) {
            matches = std::isfinite(original[i]) && std::isfinite(optimized[i])
                && std::fabs(original[i] - optimized[i])
                    <= 1e-5f + 1e-4f * std::fabs(original[i]);
        }
        std::cout << "\nOutputs match: " << std::boolalpha << matches << "\n";
        return matches ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
