#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <optional>
#include <unordered_set>

class Operation;

class Tensor {
    public:
        Tensor(std::string tensor_name,
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
        };

        Tensor(
            std::string tensor_name,
            std::vector<std::size_t> tensor_shape) {
                name = tensor_name;
                shape = tensor_shape;
            };

        void print() const {

            
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


        const std::vector<std::size_t>& getShape() const {
            return shape;
        }

        void addConsumer(Operation* consumer) {
           consumers.push_back(consumer);
        }

        void setProducer(Operation* operation) {
            producer = operation;
        }

        void setData(std::vector<float> new_data) {
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

        const std::vector<float>& getData() const {
            if (!data.has_value()) {
                throw std::runtime_error("Tensor data is not available");
            }
            return *data;
        }

        Operation* getProducer() const {
            return producer;
        }

    private:
        std::string name;
        std::optional<std::vector<float>> data;
        std::vector<std::size_t> shape;
        Operation* producer = nullptr;
        std::vector<Operation*> consumers;
} ;




class Operation{
    public:
        Operation(std::string operation_name,
                std::vector<Tensor*> op_inputs,
                std::vector<std::size_t> output_shape)
            : output("output", output_shape) {
            name = operation_name;
            inputs = op_inputs;

        for (Tensor* input : inputs) {
             input->addConsumer(this);
        }

        output.setProducer(this);
        };

        void print() const {
            std::cout << name << "\n";
            std::cout << inputs.size() << "\n"; 
            output.print();
        }
        
        Tensor* getOutput() {
            return &output;
        }

        const std::string& getName() const {
            return name;
        }

        const std::vector<Tensor*>& getInputs() const {
            return inputs;
        }

        virtual void execute() = 0;
        virtual ~Operation() = default;
    
        private:
            std::string name;
            std::vector<Tensor*> inputs;
            Tensor output;
};




class Add : public Operation {
    public:
        Add(Tensor* a, Tensor* b)
            : Operation("Add", {a, b}, checkedShape(a, b)) {
        }
    
    void execute() override {
        const std::vector<float>& a = getInputs()[0]->getData();
        const std::vector<float>& b = getInputs()[1]->getData();

        std::vector<float> result(a.size());
        for (std::size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] + b[i];
        }

        getOutput()->setData(result);
    }
    
    private:
        static std::vector<std::size_t> checkedShape(Tensor* a, Tensor* b) {
            if (a->getShape() != b->getShape()){
                    throw std::invalid_argument("Shape of inputs do not match");
                }
            return a->getShape();
        }   
};

class MatMul : public Operation {
    public:
        MatMul(Tensor* a, Tensor* b)
            : Operation("MatMul", {a, b}, checkedShape(a,b)) {
        }

        void execute() override {
            const std::vector<float>& a = getInputs()[0]->getData();
            const std::vector<float>& b = getInputs()[1]->getData();
            const std::vector<std::size_t>& a_shape = getInputs()[0]->getShape();
            const std::vector<std::size_t>& b_shape = getInputs()[1]->getShape();

            const std::size_t M = a_shape[0];
            const std::size_t K = a_shape[1];
            const std::size_t N = b_shape[1];
            std::vector<float> result(M * N, 0.0f);

            for (std::size_t i = 0; i < M; i++) {
                for (std::size_t j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (std::size_t k = 0; k < K; k++) {
                        sum += a[i * K + k] * b[k * N + j];
                    }
                    result[i * N + j] = sum;
                }
            }
            getOutput()->setData(result);
        }

    private:
        static std::vector<std::size_t> checkedShape(Tensor* a, Tensor* b) {
            if  (a->getShape().size() != 2 || b->getShape().size() != 2){
                throw std::invalid_argument("MatMul requires two-dimensional inputs");
            }
            if (a->getShape()[1] != b->getShape()[0]) {
                throw std::invalid_argument("Matrix Dimensions do not match");
            }
            return {a->getShape()[0], b->getShape()[1]};
        }   
    
};



class ReLU : public Operation {
    public:
        ReLU(Tensor* input)
            : Operation("ReLU", {input}, input->getShape()) {
        }

        void execute() override {
            const std::vector<float>& input = getInputs()[0]->getData();
            std::vector<float> result(input.size());
            for (std::size_t i = 0; i < input.size(); i++) {
                result[i] = input[i] < 0.0f ? 0.0f : input[i];
            }
            getOutput()->setData(result);
        }
};

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

int main() {
    Tensor x("x", {1.0f, 2.0f}, {1, 2});
    Tensor weights("weights", {3.0f, 4.0f, 5.0f, 6.0f}, {2, 2});
    Tensor bias("bias", {1.0f, -20.0f}, {1, 2});

    MatMul matmul(&x, &weights);
    Add add(matmul.getOutput(), &bias);
    ReLU relu(add.getOutput());

    std::unordered_set<Operation*> visited;
    std::vector<Operation*> order;
    getExecutionOrder(relu.getOutput(), visited, order);

    for (Operation* op : order) {
        op->execute();
        std::cout << op->getName() << " output: ";
        for (float value : op->getOutput()->getData()) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }
}
