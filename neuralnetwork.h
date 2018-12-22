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
    double value(double x) const {
        return x;
    }

    double derivative(double x) const {
        return 1;
    }
};

class BinaryStep : public ActivationFunction {
public:
    double value(double x) const {
        return x < 0 ? 0 : 1;
    }

    double derivative(double x) const {
        return x != 0 ? 0 : -1; // todo:
    }
};

class Sigmoid : public ActivationFunction {
public:
    double value(double x) const {
        return 1 / (1 + std::exp(-x));
    }

    double derivative(double x) const {
        double val = value(x);
        return val * (1 - val);
    }
};

class TanH : public ActivationFunction {
public:
    double value(double x) const {
        return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
    }

    double derivative(double x) const {
        return 1 - std::pow(value(x), 2);
    }
};

class ReLU : public ActivationFunction {
public:
    double value(double x) const {
        return x < 0 ? 0 : x;
    }

    double derivative(double x) const {
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

class Node {
    double bias;
    std::shared_ptr<ActivationFunction> func;

public:
    Node(double bias, std::shared_ptr<ActivationFunction> func) : bias(bias), func(func) { }
    double compute(std::vector<double> inputs) {
        double ret = 0;
        for(auto const& input: inputs)
            ret += func->value(input);
        return ret + bias;
    }
};

class CostFunction {
public:
    virtual double derivative(double y, double yHat) const = 0;
    virtual ~CostFunction() { }
};

class MSE : public CostFunction {
    double derivative(double y, double yHat) const {
        return yHat - y;
    }
    ~MSE() { }
};

class NeuralNetwork {
    std::vector<std::shared_ptr<Node>> inputLayer;
    std::unordered_map<std::shared_ptr<Node>, std::map<std::shared_ptr<Node>, double>> nodes;
    CostFunction* func;

public:
    NeuralNetwork(std::vector<std::tuple<unsigned, double, std::shared_ptr<ActivationFunction>>>& layers, CostFunction* func) : func(func) {
        // todo: better generation of randomized weights with better seeding + should this be here or in builder?
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        std::vector<std::shared_ptr<Node>> prevLayer, currLayer;

        for(auto const& [size, bias, acti]: layers) {
            // nice c++-17 structured binding :3

            // Create new node layer
            for(unsigned i = 0; i < size; ++i) {
                auto n = Node(bias, acti);
                auto nPtr = std::make_shared<Node>(n);
                currLayer.push_back(nPtr);
            }

            if(inputLayer.empty()) {
                inputLayer = std::vector<std::shared_ptr<Node>>(currLayer);
            } else {
                // let's generate the weights
                for(auto prevNode: prevLayer) {
                    nodes[prevNode] = std::map<std::shared_ptr<Node>, double>();
                    for(auto currNode: currLayer) {
                        nodes[prevNode][currNode] = distribution(generator);
                    }
                }
            }
            prevLayer = currLayer;
            currLayer.clear();
        }
    }
    ~NeuralNetwork() {
        delete func;
        // no need to take care of nodes due to shared_ptr doing the job
    }

    std::vector<double> feedForward(std::vector<double> inputs) /*const*/ {
        assert(inputs.size() == inputLayer.size());

        std::map<std::shared_ptr<Node>, std::vector<double> > nodeInputs;
        for(unsigned i = 0; i < inputs.size(); ++i) {
            nodeInputs[inputLayer[i]] = std::vector<double>();
            nodeInputs[inputLayer[i]].push_back(inputs[i]);
        }

        // Does this initialize correctly? This is so ugly too...
        std::deque<std::shared_ptr<Node>> toProcess(inputLayer.begin(), inputLayer.end());
        std::set<std::shared_ptr<Node>> processed;

        std::vector<double> results;

        // todo: this queue based implementation is really underwhelming with all the checks
        while(!toProcess.empty()) {
            auto currentNode = toProcess.front();
            toProcess.pop_front();
            if(processed.find(currentNode) != processed.end())
                continue;
            double val = currentNode->compute(nodeInputs[currentNode]);
            std::map<std::shared_ptr<Node>, double> adjacentNodes = nodes[currentNode];

            if(adjacentNodes.empty()) {
                results.push_back(val);
            } else {
                for(auto const& edge: adjacentNodes) {
                    auto adjacentNode = edge.first;
                    double weight = edge.second;
                    // lol, amazing this only came out in c++-17
                    nodeInputs.try_emplace(adjacentNode, std::vector<double>());
                    nodeInputs[adjacentNode].push_back(weight * val);
                    toProcess.push_back(adjacentNode);
                }
            }
            processed.insert(currentNode);
        }

        return results;
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
    NeuralNetwork* finish() {
        return new NeuralNetwork(layers, new MSE());
    }
};

#endif //SIMPLENN_NEURALNETWORK_H
