#ifndef SIMPLENN_NEURALNETWORK_H
#define SIMPLENN_NEURALNETWORK_H

#include <cassert>
#include <vector>
#include <map>
#include <deque>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <random>
#include <memory>
#include <set>
#include <iostream>
#include <algorithm>
#include <chrono>

enum class ActivationType {
    ID, BIN, SIGM, TANH, RELU
};

class ActivationFunction {
public:
    virtual double value(double x) const = 0;
    virtual double derivative(double x) const = 0;
};

class Identity : public ActivationFunction {
public:
    double value(double x) const override {
        return x;
    }

    double derivative(double x) const override {
        return 1;
    }
};

class BinaryStep : public ActivationFunction {
public:
    double value(double x) const override {
        return x < 0 ? 0 : 1;
    }

    double derivative(double x) const override {
        return x != 0 ? 0 : -1; // todo:
    }
};

class Sigmoid : public ActivationFunction {
public:
    double value(double x) const override {
        return 1 / (1 + std::exp(-x));
    }

    double derivative(double x) const override {
        double val = value(x);
        return val * (1 - val);
    }
};

class TanH : public ActivationFunction {
public:
    double value(double x) const override {
        return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
    }

    double derivative(double x) const override {
        return 1 - std::pow(value(x), 2);
    }
};

class ReLU : public ActivationFunction {
public:
    double value(double x) const override {
        return x < 0 ? 0 : x;
    }

    double derivative(double x) const override {
        return x < 0 ? 0 : 1;
    }
};

// todo: arctan and other


// todo: should I make this into some factory/constructor/builder?
class ActivationFunctionBuilder {
public:
    static ActivationFunction* get(ActivationType type) {
        switch (type) {
            case ActivationType::ID:
                return new Identity();
            case ActivationType::BIN:
                return new BinaryStep();
            case ActivationType::SIGM:
                return new Sigmoid();
            case ActivationType::TANH:
                return new TanH();
            case ActivationType::RELU:
                return new ReLU();
        }
    }
};

class CostFunction {
public:
    virtual double derivative(double y, double yHat) const = 0;
    virtual ~CostFunction() = default;
};

class MSE : public CostFunction {
public:
    double derivative(double y, double yHat) const override {
        return yHat - y;
    }
    ~MSE() override = default;
};

class BinaryCrossEntropy: public CostFunction {
public:
    double derivative(double y, double yHat) const override {
        return (yHat - y) / ((1 - yHat) * (yHat));
    }
    ~BinaryCrossEntropy() override = default;
};

typedef std::tuple<unsigned, double, std::shared_ptr<ActivationFunction>, std::map<int, std::vector<int>>> LayerDef;

typedef std::vector<std::pair<std::vector<double>, double>> Data;

class NeuralNetwork {

    // todo: make it so we can substitute whole networks into computation nodes (additional abstraction layer)
    class Node;
    typedef std::shared_ptr<Node> NodePtr;
    class Layer;
    typedef std::shared_ptr<Layer> LayerPtr;

    std::vector<Layer> layers;
    // todo: probably a vector as vals instead of a map would be better
    std::shared_ptr<CostFunction> func;
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution;

    class Node {
        double bias;
        double input;
//        double output;
//        std::vector<NodePtr> inputNodes;
        std::map<NodePtr, double> outgoingEdges;
        std::shared_ptr<ActivationFunction> func;

    public:
        Node(double bias, std::shared_ptr<ActivationFunction> func) : bias(bias), func(std::move(func)) { }

        void putInput(double x) { input += x; }

        void putOutputNode(NodePtr nPtr, double weight) { outgoingEdges.emplace(nPtr, weight); }

        void print() {
            std::cout << "[b:{" << bias << "}";
            if(!outgoingEdges.empty()) {
                std::cout << ",w:{";
                for(auto const& [node, weight] : outgoingEdges) {
                    std::cout << weight << ",";
                }
                std::cout << "}";
            }
            std::cout << "]";
        }

        void feedForward(bool verbose) {
            for(auto& [node, weight]: outgoingEdges) {
                node.get()->putInput(compute(weight));

                if(verbose)
                    node.get()->print();
            }
        }

        double compute(double weight) {
            return func->value(weight*input + bias);
        }

        void backpropagate() {

        };
    };

    class Layer {
        unsigned layerNum;
        std::vector<NodePtr> nodes;

    public:
        Layer() : layerNum(0) { };

        std::vector<NodePtr> getNodes() const { return nodes; };

        NodePtr getNode(unsigned n) const { if(nodes.size() <= n ) return nullptr; else return nodes.at(n); };

        bool isEmpty() const { return nodes.empty(); }

        Layer(unsigned layerNum, const LayerDef& layerDef, const Layer& nextLayer) : layerNum(layerNum) {
            auto& [size, bias, activationFunc, wiring] = layerDef;
            bool emptyWiring = wiring.empty();
            for(unsigned i = 0; i < size; ++i) {
                auto n = Node(bias, activationFunc);
                auto nPtr = std::make_shared<Node>(n);
                if(!nextLayer.isEmpty()) {
                    if(emptyWiring) {
                        for (auto &nPtr2 : nextLayer.getNodes())
                            nPtr.get()->putOutputNode(nPtr2, distribution(generator));
                    } else {
                        if(wiring.find(i) != wiring.end()) {
                            auto& iVec = wiring.at(i);
                            for(auto& i2 : iVec) {
                                auto nPtr2 = nextLayer.getNode(i2);
                                if(nPtr2 != nullptr)
                                    nPtr.get()->putOutputNode(nPtr2, distribution(generator));
                            }
                        }

                    }
                }
                nodes.push_back(nPtr);
            }
        }

        void putInput(std::vector<double>& input) {
            assert(input.size() == nodes.size());

            for(unsigned i = 0; i < input.size(); ++i)
                nodes.at(i).get()->putInput(input.at(i));
        }

        void feedForward(bool verbose) {
            if(verbose)
                std::cout << "LAYER " << layerNum << ": ";
            for(auto& node : nodes)
                node.get()->feedForward(verbose);
        }
    };

public:
    NeuralNetwork(std::vector<LayerDef> &layerDefs,
                  std::shared_ptr<CostFunction> func) : func(std::move(func)) {
        // todo: better generation of randomized weights with better seeding + should this be here or in builder?
        unsigned long seed = (unsigned long) std::chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);
        distribution = std::uniform_real_distribution<double>(-1.0,1.0);

        Layer nextLayer; // defaults to an empty layer with dft constructor
        unsigned layerCount = 0;
        std::reverse(layerDefs.begin(), layerDefs.end()); // we go from end to beginning
        for(auto const& layerDef: layerDefs) {
            // Create new node layer
            Layer currLayer(layerCount, layerDef, nextLayer);
            layers.push_back(currLayer);

            nextLayer = currLayer;
            layerCount++;
        }
    }

    ~NeuralNetwork() = default;

    void train(Data trainingData, unsigned epochs, unsigned miniBatchSize, double learningRate, bool verbose) {
        for(unsigned i = 0; i < epochs; ++i) {

            // shuffle training data
            std::shuffle(trainingData.begin(), trainingData.end(), generator);

            // create mini batches
            std::vector<Data> miniBatches = prepareBatches(trainingData, miniBatchSize);

            // train on mini batches
            for(auto& miniBatch: miniBatches) {
                train(miniBatch, learningRate, verbose);
            }

            if(verbose)
                std::cout << "Epoch " << i << " finished\n\n";
        }
    }

    std::vector<double> predict(std::vector<double> inputs, bool verbose) {

    }

private:
    std::vector<Data> prepareBatches(const Data& trainingData, unsigned miniBatchSize) {
        std::vector<Data> miniBatches;
        unsigned currBatchSize = 0, currBatchNum = 0;
        for(auto& data : trainingData) {
            // todo: this double checking of curr batch size isnt too elegant
            if(currBatchSize == 0) {
                miniBatches.emplace_back(Data());
            }
            miniBatches[currBatchNum].push_back(data);
            currBatchNum = currBatchSize == 0 ? currBatchNum + 1 : currBatchNum;
            currBatchSize++;
            currBatchSize %= miniBatchSize;
        }
        return miniBatches;
    }

    void train(const Data& miniBatch, double learningRate, bool verbose) {
        std::vector<double> batchOutput;
        for(const auto& [input, output] : miniBatch) {
            feedForward(input, verbose);
            batchOutput.push_back(output);
        }
        backpropagate(batchOutput, learningRate, verbose);
    }

    void feedForward(std::vector<double> inputs, bool verbose) {
        layers.at(0).putInput(inputs);
        for(auto& layer : layers)
            layer.feedForward(verbose);
    }

    void backpropagate(std::vector<double> expected, double learningRate, bool verbose) {

    }
};

class ConcurrentNeuralNetwork: public NeuralNetwork {
public:
    ConcurrentNeuralNetwork() = default;
};

// for now, lets assume full connection between layers and only MSE cost function
class NNBuilder {
    std::vector<LayerDef> layerDefs;
public:
    NNBuilder& start() {
        layerDefs.clear();
        return *this;
    }
    NNBuilder& addLayer(unsigned size, double bias, ActivationType type) {
        layerDefs.emplace_back(std::make_tuple(size, bias, std::shared_ptr<ActivationFunction>(ActivationFunctionBuilder::get(type))));
        return *this;
    }
    NNBuilder& addInputLayer(unsigned size) {
        if(layerDefs.size())
            return *this;
        return addLayer(size, 0, ActivationType::ID);
    }
    NeuralNetwork* build() {
        auto ret = new NeuralNetwork(layerDefs, std::shared_ptr<CostFunction>(new MSE()));
        layerDefs.clear();
        return ret;
    }
};

#endif //SIMPLENN_NEURALNETWORK_H
