#include "Ops.h"

#include <stdexcept>

Add::Add(Tensor* a, Tensor* b)
: Operation("Add", {a, b}, checkedShape(a, b)) {
}

void Add::execute()  {
    const std::vector<float>& a = getInputs()[0]->getData();
    const std::vector<float>& b = getInputs()[1]->getData();

    std::vector<float>& result = getOutput()->getWritableData();
    for (std::size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }

}

std::vector<std::size_t> Add::checkedShape(Tensor* a, Tensor* b) {
    if (a->getShape() != b->getShape()){
        throw std::invalid_argument("Shape of inputs do not match");
    }
    return a->getShape();
}

MatMul::MatMul(Tensor* a, Tensor* b)
: Operation("MatMul", {a, b}, checkedShape(a, b)) {
}

void MatMul::execute()  {
    const std::vector<float>& a = getInputs()[0]->getData();
    const std::vector<float>& b = getInputs()[1]->getData();
    const std::vector<std::size_t>& a_shape = getInputs()[0]->getShape();
    const std::vector<std::size_t>& b_shape = getInputs()[1]->getShape();

    const std::size_t M = a_shape[0];
    const std::size_t K = a_shape[1];
    const std::size_t N = b_shape[1];
    std::vector<float>& result = getOutput()->getWritableData();

    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            result[i * N + j] = sum;
        }
    }
}

std::vector<std::size_t> MatMul::checkedShape(Tensor* a, Tensor* b) {
    if  (a->getShape().size() != 2 || b->getShape().size() != 2){
        throw std::invalid_argument("MatMul requires two-dimensional inputs");
    }
    if (a->getShape()[1] != b->getShape()[0]) {
        throw std::invalid_argument("Matrix Dimensions do not match");
    }
    return {a->getShape()[0], b->getShape()[1]};
}

ReLU::ReLU(Tensor* input)
: Operation("ReLU", {input}, input->getShape()) {
}

void ReLU::execute()  {
    const std::vector<float>& input = getInputs()[0]->getData();
    std::vector<float>& result = getOutput()->getWritableData();
    for (std::size_t i = 0; i < input.size(); i++) {
        result[i] = input[i] < 0.0f ? 0.0f : input[i];
    }
}

FusedAddReLU::FusedAddReLU(Tensor* a, Tensor* b)
: Operation("FusedAddReLU", {a, b}, checkedShape(a, b)) {
}

void FusedAddReLU::execute()  {
    const std::vector<float>& a = getInputs()[0]->getData();
    const std::vector<float>& b = getInputs()[1]->getData();
    std::vector<float>& result = getOutput()->getWritableData();
    for (std::size_t i = 0; i < a.size(); i++) {
        const float sum = a[i] + b[i];
        result[i] = sum < 0.0f ? 0.0f : sum;
    }
}

std::vector<std::size_t> FusedAddReLU::checkedShape(Tensor* a, Tensor* b) {
    if (a->getShape() != b->getShape()) {
        throw std::invalid_argument("Shape of inputs do not match");
    }
    return a->getShape();
}

FusedLinearReLU::FusedLinearReLU(Tensor* x, Tensor* weights, Tensor* bias)
: Operation("FusedLinearReLU", {x, weights, bias}, checkedShape(x, weights, bias)) {
}

void FusedLinearReLU::execute()  {
    const auto& x = getInputs()[0]->getData();
    const auto& weights = getInputs()[1]->getData();
    const auto& bias = getInputs()[2]->getData();
    const auto& x_shape = getInputs()[0]->getShape();
    const auto& weights_shape = getInputs()[1]->getShape();
    const std::size_t M = x_shape[0];
    const std::size_t K = x_shape[1];
    const std::size_t N = weights_shape[1];
    std::vector<float>& result = getOutput()->getWritableData();

    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < K; k++) {
                sum += x[i * K + k] * weights[k * N + j];
            }
            sum += bias[i * N + j];
            result[i * N + j] = sum < 0.0f ? 0.0f : sum;
        }
    }
}

std::vector<std::size_t> FusedLinearReLU::checkedShape(
    Tensor* x, Tensor* weights, Tensor* bias) {
    const auto& x_shape = x->getShape();
    const auto& weights_shape = weights->getShape();
    if (x_shape.size() != 2 || weights_shape.size() != 2 ||
        x_shape[1] != weights_shape[0]) {
        throw std::invalid_argument("Invalid matrix shapes for FusedLinearReLU");
    }
    std::vector<std::size_t> output_shape{x_shape[0], weights_shape[1]};
    if (bias->getShape() != output_shape) {
        throw std::invalid_argument("Bias shape must equal the matrix output shape");
    }
    return output_shape;
}
