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

enum class ActiFuncType {
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
class ActiFuncBuilder {
public:
    static ActivationFunction* get(ActiFuncType type) {
        switch (type) {
            case ActiFuncType::ID:
                return new Identity();
            case ActiFuncType::BIN:
                return new BinaryStep();
            case ActiFuncType::SIGM:
                return new Sigmoid();
            case ActiFuncType::TANH:
                return new TanH();
            case ActiFuncType::RELU:
                return new ReLU();
        }
    }
};

// todo: make it so we can substitute whole networks into computation nodes (additional abstraction layer)

class Node {
    unsigned layer;
    double bias;
    std::vector<double> inputs;
    // todo: move weights into the node too
    std::shared_ptr<ActivationFunction> func;

public:
    Node(unsigned layer, double bias, std::shared_ptr<ActivationFunction> func) : layer(layer), bias(bias), func(std::move(func)) { }
    unsigned getLayer() const { return layer; }
    double getBias() const { return bias; }
    void putInput(double x) { inputs.push_back(x); }
    void print(std::vector<double> weights) {
        std::cout << "[b:{" << bias << "},w:{";
        for(unsigned i = 0; i < weights.size(); ++i) {
            if(i != weights.size() - 1)
                std::cout << weights[i] << ",";
            else
                std::cout << weights[i] << "}]";
        }
    }
    double compute() {
        double ret = 0;
        for(auto const& input: inputs)
            ret += func->value(input);
        inputs.clear();
        return ret + bias;
    }
};

typedef std::shared_ptr<Node> NodePtr;

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

class NeuralNetwork {
    std::vector<NodePtr> inputLayer;
    // todo: probably a vector as vals instead of a map would be better
    std::unordered_map<NodePtr, std::map<NodePtr, double>> nodes;
    std::shared_ptr<CostFunction> func;

public:
    NeuralNetwork(std::vector<std::tuple<unsigned, double, std::shared_ptr<ActivationFunction>>>& layers, std::shared_ptr<CostFunction> func) : func(std::move(func)) {
        // todo: better generation of randomized weights with better seeding + should this be here or in builder?
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        std::vector<NodePtr> prevLayer, currLayer;
        unsigned layerCount = 0;

        for(auto const& [size, bias, acti]: layers) {
            // nice c++-17 structured binding :3

            // Create new node layer
            for(unsigned i = 0; i < size; ++i) {
                auto n = Node(layerCount, bias, acti);
                auto nPtr = std::make_shared<Node>(n);
                currLayer.push_back(nPtr);
            }

            if(inputLayer.empty()) {
                inputLayer = std::vector<NodePtr>(currLayer);
            } else {
                // let's generate the weights
                for(auto const& prevNode: prevLayer) {
                    nodes[prevNode] = std::map<NodePtr, double>();
                    for(auto const& currNode: currLayer) {
                        nodes[prevNode][currNode] = distribution(generator);
                    }
                }
            }
            prevLayer = currLayer;
            currLayer.clear();
            layerCount++;
        }
    }
    ~NeuralNetwork() = default;

    std::vector<double> feedForward(std::vector<double> inputs, bool print) /*const*/ {
        assert(inputs.size() == inputLayer.size());

        for(unsigned i = 0; i < inputs.size(); ++i)
            inputLayer[i].get()->putInput(inputs[i]);

        std::deque<NodePtr> toProcess(inputLayer.begin(), inputLayer.end());
        std::set<NodePtr> processed;

        std::vector<double> results;
        std::vector<double> weights;

        // todo: this queue based implementation is really underwhelming with that set of processed nodes
        unsigned layerCount = 0;
        if(print)
            std::cout << "LAYER " << layerCount << ": ";
        while(!toProcess.empty()) {
            auto currentNode = toProcess.front();
            toProcess.pop_front();
            // todo: this is very unoptimal
            if(processed.find(currentNode) != processed.end())
                continue;
            double val = currentNode->compute();
            std::map<NodePtr, double> adjacentNodes = nodes[currentNode];

            if(adjacentNodes.empty()) {
                results.push_back(val);
            } else {
                for(auto const& edge: adjacentNodes) {
                    auto adjacentNode = edge.first;
                    double weight = edge.second;
                    weights.push_back(weight);
                    adjacentNode.get()->putInput(weight * val);
                    toProcess.push_back(adjacentNode);
                }
            }
            processed.insert(currentNode);
            if(print) {
                if(currentNode.get()->getLayer() == layerCount) {
                    currentNode.get()->print(weights);
                } else {
                    layerCount++;
                    std::cout << std::endl;
                    std::cout << "LAYER " << layerCount << ": ";
                    currentNode.get()->print(weights);
                }
                weights.clear();
            }

        }

        return results;
    }

    void backpropagate(std::vector<double> expected, std::vector<double> computed) {

    }
};

// for now, lets assume full connection between layers and only MSE cost function
class NNBuilder {
    std::vector<std::tuple<unsigned, double, std::shared_ptr<ActivationFunction>>> layers;
public:
    NNBuilder& start() {
        layers.clear();
        return *this;
    }
    NNBuilder& addLayer(unsigned size, double bias, ActiFuncType type) {
        layers.emplace_back(std::make_tuple(size, bias, std::shared_ptr<ActivationFunction>(ActiFuncBuilder::get(type))));
        return *this;
    }
    NeuralNetwork* build() {
        auto ret = new NeuralNetwork(layers, std::shared_ptr<CostFunction>(new MSE()));
        layers.clear();
        return ret;
    }
};

#endif //SIMPLENN_NEURALNETWORK_H
