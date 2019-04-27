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
    // wrt yHat
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

    std::vector<Layer> layers;
    std::shared_ptr<CostFunction> costFunc;
    unsigned long seed;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution;

    class Node {
        std::string nodeId;
        double bias;
        double output;
        double outputDerivative;
        double dErrordActi; // the derivative of the cost function wrt the
        std::map<NodePtr, double> incomingEdges;
        std::shared_ptr<ActivationFunction> activationFunc;

    public:
        Node(std::string nodeId, double bias, std::shared_ptr<ActivationFunction> func) : nodeId(std::move(nodeId)), bias(bias), output(0), outputDerivative(0), dErrordActi(0), activationFunc(std::move(func)) { }

        void putInputNode(NodePtr nPtr, double weight) {
            if(incomingEdges.find(nPtr) == incomingEdges.end())
                incomingEdges.emplace(nPtr, weight);
            else
                incomingEdges[nPtr] = weight;
        }

        double getOutput() const { return output; }

        void setOutput(double output2) { output = output2; }

        bool isInputNode() const { return incomingEdges.empty(); }

        void print() {
            std::cout << "[n:{" << nodeId << "},b:{" << bias << "}";
            if(!incomingEdges.empty()) {
                std::cout << ",w:{";
                for(auto const& [node, weight] : incomingEdges) {
                    std::cout << weight << ",";
                }
                std::cout << "}";
            }
            std::cout << "]";
        }

        double feedForward(bool verbose) {
            double result = 0;
            for(auto& [node, weight]: incomingEdges) {
                result += weight*node.get()->getOutput();
            }
            if(verbose)
                print();
            output = activationFunc->value(result);
            outputDerivative = activationFunc->derivative(result);
            return result;
        }

        void updateErrorDerivative(double val) { dErrordActi += val; }

        void updateWeights(double learningRate) {
            for(auto [node, weight] : incomingEdges) {
                double incomingVal = node.get()->getOutput();
                node.get()->updateErrorDerivative(weight*outputDerivative*dErrordActi);
                weight -= learningRate*(incomingVal*outputDerivative*dErrordActi); // + alpha*weight
                incomingEdges[node] = weight;
            }
            dErrordActi = 0;
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

        Layer(unsigned layerNum, const LayerDef& layerDef, const Layer& prevLayer, std::default_random_engine generator,
                std::uniform_real_distribution<double> distribution) : layerNum(layerNum) {
            auto& [size, bias, activationFunc, wiring] = layerDef;
            bool emptyWiring = wiring.empty();
            for(unsigned i = 0; i < size; ++i) {
                std::string nodeId = "n(" + std::to_string(layerNum) + "," + std::to_string(i) + ")";
                auto n = Node(nodeId , bias, activationFunc);
                auto nPtr = std::make_shared<Node>(n);
                if(!prevLayer.isEmpty()) {
                    if(emptyWiring) {
                        auto nodes2 = prevLayer.getNodes();
                        for(auto &nPtr2 : nodes2)
                            nPtr.get()->putInputNode(nPtr2, distribution(generator));
                    } else {
                        if(wiring.find(i) != wiring.end()) {
                            auto& iVec = wiring.at(i);
                            for(auto& i2 : iVec) {
                                auto nPtr2 = prevLayer.getNode(i2);
                                if(nPtr2 != nullptr)
                                    nPtr.get()->putInputNode(nPtr2, distribution(generator));
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
                if(nodes.at(i).get()->isInputNode())
                    nodes.at(i).get()->setOutput(input.at(i));
        }

        void print() const {
            std::cout << "LAYER " << layerNum << ": ";
            for(auto& node : nodes)
                node.get()->print();
        }

        void feedForward(bool verbose) {
            if(verbose)
                std::cout << "LAYER " << layerNum << ": ";
            for(auto& node : nodes)
                node.get()->feedForward(verbose);
        }

        std::vector<double> getOutput() const {
            std::vector<double> result;
            for(auto& node : nodes)
                result.push_back(node.get()->getOutput());
            return result;
        }

        void backpropagate(double learningRate, bool verbose) {
            for(auto& node : nodes)
                node.get()->updateWeights(learningRate);
        }

        void updateErrorDerivatives(const std::vector<double>& expected, const std::shared_ptr<CostFunction>& localCostFunc) {
            assert(expected.size() == nodes.size());
            for(unsigned i = 0; i < expected.size(); ++i) {
                double expectedVal = expected.at(i);
                double predictedVal = nodes.at(i).get()->getOutput();
                nodes.at(i)->updateErrorDerivative(localCostFunc.get()->derivative(expectedVal, predictedVal));
            }
        }
    };

public:
    NeuralNetwork(std::vector<LayerDef> &layerDefs,
                  std::shared_ptr<CostFunction> func) : costFunc(std::move(func)) {
        // todo: better generation of randomized weights with better seeding + should this be here or in builder?
        seed = (unsigned long) std::chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);
        distribution = std::uniform_real_distribution<double>(-1.0,1.0);

        Layer prevLayer; // defaults to an empty layer with dft constructor
        unsigned layerCount = 0;
        for(auto const& layerDef: layerDefs) {
            // Create new node layer
            Layer currLayer(layerCount, layerDef, prevLayer, generator, distribution);
            layers.push_back(currLayer);

            prevLayer = currLayer;
            layerCount++;
        }
    }

    ~NeuralNetwork() = default;

    void train(Data trainingData, unsigned epochs, unsigned miniBatchSize, double learningRate, bool verbose) {
        for(unsigned i = 0; i < epochs; ++i) {
            if(verbose)
                std::cout << "Epoch " << i << " starting\n\n";

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

    // todo:
//    std::vector<double> predict(std::vector<double> inputs, bool verbose) {
//
//    }

    void print() {
        std::cout << "Printing neural network\n";
        for(const auto& layer : layers)
            layer.print();
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
        std::vector<double> expected;
        for(const auto& [input, output] : miniBatch) {
            feedForward(input, verbose);
            expected.push_back(output);
        }

        backpropagate(expected, learningRate, verbose);
    }

    std::vector<double> feedForward(std::vector<double> inputs, bool verbose) {
        layers.at(0).putInput(inputs);
        for(auto& layer : layers)
            layer.feedForward(verbose);
        return layers.at(layers.size() - 1).getOutput();
    }

    void backpropagate(const std::vector<double>& expected, double learningRate, bool verbose) {
        layers.at(layers.size() - 1).updateErrorDerivatives(expected, costFunc);
        for(auto it = layers.rbegin(); it != layers.rend(); it++) {
            it->backpropagate(learningRate, verbose);
        }
    }
};

class ConcurrentNeuralNetwork: public NeuralNetwork {
public:
    ConcurrentNeuralNetwork();
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
        layerDefs.emplace_back(std::make_tuple(size, bias, std::shared_ptr<ActivationFunction>(ActivationFunctionBuilder::get(type)), std::map<int, int>()));
        return *this;
    }
    NNBuilder& addInputLayer(unsigned size) {
        if(!layerDefs.empty())
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
