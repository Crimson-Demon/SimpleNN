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
class ActiFuncBuilder {
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

// todo: make it so we can substitute whole networks into computation nodes (additional abstraction layer)
class Node;

typedef std::shared_ptr<Node> NodePtr;

class Edge {
    // todo: do we need both sides?
    NodePtr in;
    NodePtr out;
    double weight;
public:
    Edge(NodePtr in, NodePtr out, double weight): in(std::move(in)), out(std::move(out)), weight(weight) { }
    double getWeight() const { return weight; }
    NodePtr otherEnd(NodePtr const& n) const {
        if(n.get() == in.get())
            return out;
        else if(n.get() == out.get())
            return in;
        else
            throw std::runtime_error("Vertex does not belong to given edge\n");
    }
    NodePtr getInNode() const { return in; }
    NodePtr getOutNode() const { return out; }
};

typedef std::shared_ptr<Edge> EdgePtr;

class Node {
    unsigned layer;
    double bias;
    std::vector<double> inputs;
    double output;
    std::vector<EdgePtr> inputNodes;
    std::vector<EdgePtr> outputNodes;
    std::shared_ptr<ActivationFunction> func;

public:
    Node(unsigned layer, double bias, std::shared_ptr<ActivationFunction> func) : layer(layer), bias(bias), func(std::move(func)) { }

    unsigned getLayer() const { return layer; }

    double getBias() const { return bias; }

    double getOutput() const { return output; }

    void putInput(double x) { inputs.push_back(x); }

    void putInputNode(EdgePtr const& edge) { inputNodes.push_back(edge); }

    std::vector<NodePtr> getInputNodes() const {
        std::vector<NodePtr> res;
        for(auto const& edge: inputNodes)
            res.push_back(edge.get()->getInNode());
        return res;
    }

    std::vector<NodePtr> getOutputNodes() const {
        std::vector<NodePtr> res;
        for(auto const& edge: outputNodes)
            res.push_back(edge.get()->getOutNode());
        return res;
    }

    void putOutputNode(EdgePtr const& edge) { outputNodes.push_back(edge); }

    void print() {
        std::cout << "[b:{" << bias << "}";
        if(!outputNodes.empty()) {
            std::cout << ",w:{";
            for (unsigned i = 0; i < outputNodes.size(); ++i) {
                if (i != outputNodes.size() - 1)
                    std::cout << outputNodes[i].get()->getWeight() << ",";
                else
                    std::cout << outputNodes[i].get()->getWeight() << "}";
            }
        }
        std::cout << "]";
    }

    void feedforward(double val) {
        for(auto& node: outputNodes)
            node.get()->getOutNode().get()->putInput(val);
    }

    double compute() {
        if(isInput()) {
            double ret = 0;
            for(unsigned i = 0; i < inputs.size(); ++i)
                ret += inputs[i] * 1;
            inputs.clear();
            output = func->value(ret + bias);
            return output;
        } else {
            assert(inputs.size() == inputNodes.size());
            double ret = 0;
            for(unsigned i = 0; i < inputs.size(); ++i)
                ret += inputs[i] * inputNodes[i].get()->getWeight();
            inputs.clear();
            output = func->value(ret + bias);
            return output;
        }
    }

    bool isInput() const { return inputNodes.empty(); }

    bool isOutput() const { return outputNodes.empty(); }

    bool isHidden() const { return !isInput() && !isOutput(); }
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

class NeuralNetwork {
    std::vector<NodePtr> inputLayer;
    // todo: probably a vector as vals instead of a map would be better
    std::vector<NodePtr> nodes;
    std::shared_ptr<CostFunction> func;
    std::default_random_engine generator;

public:
    NeuralNetwork(std::vector<std::tuple<unsigned, double, std::shared_ptr<ActivationFunction>>>& layers, std::shared_ptr<CostFunction> func) : func(std::move(func)) {
        // todo: better generation of randomized weights with better seeding + should this be here or in builder?
        unsigned long seed = (unsigned long) std::chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);

        std::uniform_real_distribution<double> distribution(-1.0,1.0);
        std::vector<NodePtr> prevLayer, currLayer;
        unsigned layerCount = 0;

        for(auto const& [size, bias, acti]: layers) {
            // nice c++-17 structured binding :3

            // Create new node layer
            for(unsigned i = 0; i < size; ++i) {
                auto n = Node(layerCount, bias, acti);
                auto nPtr = std::make_shared<Node>(n);
                if(!prevLayer.empty()) {
                    for(auto& prevNPtr: prevLayer) {
                        auto e = Edge(prevNPtr, nPtr, distribution(generator));
                        auto ePtr = std::make_shared<Edge>(e);
                        nPtr.get()->putInputNode(ePtr);
                        prevNPtr.get()->putOutputNode(ePtr);
                    }
                }
                currLayer.push_back(nPtr);
                nodes.push_back(nPtr);
            }

            if(!layerCount)
                inputLayer = std::vector<NodePtr>(currLayer);

            prevLayer = currLayer;
            currLayer.clear();
            layerCount++;
        }
    }

    ~NeuralNetwork() = default;

    void train(std::vector<std::pair<double, double>> trainingData, unsigned epochs, unsigned miniBatchSize, double learningRate, bool verbose) {
        for(unsigned i = 0; i < epochs; ++i) {
            // shuffle training data
            std::shuffle(trainingData.begin(), trainingData.end(), generator);

            // create mini batches
            std::vector<std::vector<std::pair<double, double>>> miniBatches; // typedef vec<pair<double, double>> into data
            unsigned currBatchSize = 0, currBatchNum = 0;
            for(unsigned j = 0; j < trainingData.size(); ++j) {
                // todo: this double checking of curr batch size isnt too elegant
                if(currBatchSize == 0) {
                    miniBatches.emplace_back(std::vector<std::pair<double, double>>());
                }
                miniBatches[currBatchNum].push_back(trainingData[j]);
                currBatchNum = currBatchSize == 0 ? currBatchNum + 1 : currBatchNum;
                currBatchSize++;
                currBatchSize %= miniBatchSize;
            }

            for(auto& miniBatch: miniBatches) {
                train(miniBatch, learningRate, verbose);
            }

            if(verbose)
                std::cout << "Epoch " << i << " finished\n\n";
        }
    }

//private:
    void train(std::vector<std::pair<double, double>> const& miniBatch, double learningRate, bool verbose) {

    }


    std::vector<double> feedForward(std::vector<double> inputs, bool verbose) /*const*/ {
        assert(inputs.size() == inputLayer.size());

        for(unsigned i = 0; i < inputs.size(); ++i)
            inputLayer[i].get()->putInput(inputs[i]);

        std::vector<double> results;

        unsigned layerCount = 0;

        if(verbose)
            std::cout << "LAYER " << layerCount << ": ";

        for(auto &currentNode: nodes) {
            double val = currentNode.get()->compute();

            if(currentNode.get()->isOutput())
                results.push_back(val);
            else
                currentNode.get()->feedforward(val);

            if(verbose) {
                if(currentNode.get()->getLayer() == layerCount) {
                    currentNode.get()->print();
                } else {
                    layerCount++;
                    std::cout << std::endl;
                    std::cout << "LAYER " << layerCount << ": ";
                    currentNode.get()->print();
                }
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
    NNBuilder& addLayer(unsigned size, double bias, ActivationType type) {
        layers.emplace_back(std::make_tuple(size, bias, std::shared_ptr<ActivationFunction>(ActiFuncBuilder::get(type))));
        return *this;
    }
    NNBuilder& addInputLayer(unsigned size) {
        if(layers.size())
            return *this;
        return addLayer(size, 0, ActivationType::ID);
    }
    NeuralNetwork* build() {
        auto ret = new NeuralNetwork(layers, std::shared_ptr<CostFunction>(new MSE()));
        layers.clear();
        return ret;
    }
};

#endif //SIMPLENN_NEURALNETWORK_H
