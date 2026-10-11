#include "Tensor.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

Tensor::Tensor(std::string tensor_name,
    std::vector<float> tensor_data,
    std::vector<std::size_t> tensor_shape) {
    name = tensor_name;
    data = tensor_data;
    shape = tensor_shape;

    std::size_t temp = 1;
    for (std::size_t k=0; k < shape.size(); k++){
        temp *= shape[k];
    }

    if (temp != data->size()){
        throw std::invalid_argument("Shape does not match data size");
    }
}

Tensor::Tensor(
    std::string tensor_name,
    std::vector<std::size_t> tensor_shape) {
    name = tensor_name;
    shape = tensor_shape;
}

void Tensor::print() const {

    std::cout << "Tensor is " << name << "\n";

    if (data.has_value()){
        std::cout << "Data is ";
        for (std::size_t i=0; i<this->data->size(); i++ ){
            std::cout << (*data)[i] << " ";
        }
        std::cout << "\n";
    } else {
        std::cout << "No data in Tensor";
        std::cout << "\n";
    }

    std::cout << "Shape is: ";
    for (std::size_t j=0; j<this->shape.size(); j++ ){
        std::cout << this->shape[j] << " ";
    }
    std::cout << "\n";

    std::cout << ((producer == nullptr) ? "producer is null pointer" : "producer is not null pointer") << "\n";
    std::cout << (consumers.size()) << "\n";
}

const std::vector<std::size_t>& Tensor::getShape() const {
    return shape;
}

void Tensor::addConsumer(Operation* consumer) {
    consumers.push_back(consumer);
}

void Tensor::setProducer(Operation* operation) {
    producer = operation;
}

void Tensor::setData(std::vector<float> new_data) {
    // Validate the element count, then assign to data.
    std::size_t temp = 1;
    for (std::size_t k=0; k < shape.size(); k++){
        temp *= shape[k];
    }
    if (new_data.size() != temp){
        throw std::invalid_argument("Shape does not match data size");
    }
    data = new_data;
}

const std::vector<float>& Tensor::getData() const {
    if (!data.has_value()) {
        throw std::runtime_error("Tensor data is not available");
    }
    return *data;
}

Operation* Tensor::getProducer() const {
    return producer;
}

const std::vector<Operation*>& Tensor::getConsumers() const {
    return consumers;
}

void Tensor::removeConsumer(Operation* consumer) {
    std::vector<Operation*>::iterator it = std::find(consumers.begin(), consumers.end(), consumer);
    if (it != consumers.end()) {
        consumers.erase(it);
    }
}
