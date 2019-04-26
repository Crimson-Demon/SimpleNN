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

class NeuralNetwork {

    // todo: make it so we can substitute whole networks into computation nodes (additional abstraction layer)
    class Node;
    typedef std::shared_ptr<Node> NodePtr;
    class Edge;
    typedef std::shared_ptr<Edge> EdgePtr;
    class Layer;
    typedef std::shared_ptr<Layer> LayerPtr;

//    std::vector<NodePtr> inputLayer;
    std::vector<Layer> layers;
    // todo: probably a vector as vals instead of a map would be better
    //std::vector<NodePtr> nodes;
    std::shared_ptr<CostFunction> func;
    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution;

    class Edge {
        // todo: do we need both sides?
        NodePtr in;
        NodePtr out;
        double weight;
    public:
        Edge(NodePtr in, NodePtr out, double weight): in(std::move(in)), out(std::move(out)), weight(weight) { }
        double getWeight() const { return weight; }
        /*NodePtr otherEnd(NodePtr const& n) const {
            if(n.get() == in.get())
                return out;
            else if(n.get() == out.get())
                return in;
            else
                throw std::runtime_error("Vertex does not belong to given edge\n");
        }*/
        NodePtr getInNode() const { return in; }
        NodePtr getOutNode() const { return out; }
    };

    class Node {
        double bias;
//        std::vector<double> inputs;
        double input;
        double output;
        std::vector<NodePtr> inputNodes;
//        std::vector<EdgePtr> inputNodes;
        std::map<NodePtr, double> outgoingEdges;
//        std::vector<EdgePtr> outputNodes;
        std::shared_ptr<ActivationFunction> func;

    public:
        Node(double bias, std::shared_ptr<ActivationFunction> func) : bias(bias), func(std::move(func)) { }

        double getBias() const { return bias; }

        double getOutput() const { return output; }

//        void putInput(double x) { inputs.push_back(x); }

//        void putInputNode(EdgePtr const& edge) { inputNodes.push_back(edge); }

        void putOutputNode(NodePtr nPtr, double weight) { outgoingEdges.emplace(nPtr, weight); }

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

        void backpropagate();

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

    void train(std::vector<std::pair<double, double>> trainingData, unsigned epochs, unsigned miniBatchSize, double learningRate, bool verbose) {
        for(unsigned i = 0; i < epochs; ++i) {

            // shuffle training data
            std::shuffle(trainingData.begin(), trainingData.end(), generator);

            // create mini batches
            std::vector<std::vector<std::pair<double, double>>> miniBatches = prepareBatches(trainingData, miniBatchSize); // typedef vec<pair<double, double>> into data

            // train on mini batches
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

    std::vector<std::vector<std::pair<double, double>>> prepareBatches(std::vector<std::pair<double, double>> trainingData, unsigned miniBatchSize) {
        std::vector<std::vector<std::pair<double, double>>> miniBatches;
        unsigned currBatchSize = 0, currBatchNum = 0;
        for(auto& data : trainingData)
        for(unsigned j = 0; j < trainingData.size(); ++j) {
            // todo: this double checking of curr batch size isnt too elegant
            if(currBatchSize == 0) {
                miniBatches.emplace_back(std::vector<std::pair<double, double>>());
            }
            miniBatches[currBatchNum].push_back(data);
            currBatchNum = currBatchSize == 0 ? currBatchNum + 1 : currBatchNum;
            currBatchSize++;
            currBatchSize %= miniBatchSize;
        }
        return miniBatches;
    }

    std::vector<double> feedForward(std::vector<double> inputs, bool verbose) {
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

class ConcurrentNeuralNetwork: public NeuralNetwork {
public:
    ConcurrentNeuralNetwork() = default;
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
        layers.emplace_back(std::make_tuple(size, bias, std::shared_ptr<ActivationFunction>(ActivationFunctionBuilder::get(type))));
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
