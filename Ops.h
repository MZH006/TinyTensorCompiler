#pragma once

#include "Operation.h"

class Add : public Operation {
public:
    Add(Tensor* a, Tensor* b);
    void execute() override;

private:
    static std::vector<std::size_t> checkedShape(Tensor* a, Tensor* b);
};

class MatMul : public Operation {
public:
    MatMul(Tensor* a, Tensor* b);
    void execute() override;

private:
    static std::vector<std::size_t> checkedShape(Tensor* a, Tensor* b);
};

class ReLU : public Operation {
public:
    ReLU(Tensor* input);
    void execute() override;
};

class FusedAddReLU : public Operation {
public:
    FusedAddReLU(Tensor* a, Tensor* b);
    void execute() override;

private:
    static std::vector<std::size_t> checkedShape(Tensor* a, Tensor* b);
};

class FusedLinearReLU : public Operation {
public:
    FusedLinearReLU(Tensor* x, Tensor* weights, Tensor* bias);
    void execute() override;

private:
    static std::vector<std::size_t> checkedShape(Tensor* x, Tensor* weights, Tensor* bias);
};
