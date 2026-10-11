#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

class Operation;

class Tensor {
public:
    Tensor(std::string tensor_name,
        std::vector<float> tensor_data,
        std::vector<std::size_t> tensor_shape);

    Tensor(
        std::string tensor_name,
        std::vector<std::size_t> tensor_shape);

    void print() const;

    const std::vector<std::size_t>& getShape() const;

    void addConsumer(Operation* consumer);

    void setProducer(Operation* operation);

    void setData(std::vector<float> new_data);

    const std::vector<float>& getData() const;

    Operation* getProducer() const;

    const std::vector<Operation*>& getConsumers() const;

    void removeConsumer(Operation* consumer);
private:
    std::string name;
    std::optional<std::vector<float>> data;
    std::vector<std::size_t> shape;
    Operation* producer = nullptr;
    std::vector<Operation*> consumers;
};
