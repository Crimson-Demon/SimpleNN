#include <iostream>

#include "neuralnetwork.h"

int main() {
    NNBuilder builder;
    auto *nn = builder
            .start()
            .addInputLayer(3)
            .addLayer(4, 0.5, ActivationType::SIGM)
            .addLayer(1, -0.5, ActivationType::RELU)
            .build();
    Data data = {{{0.91, 0.22, 0.63}, 0.5},
                 {{0.24, 0.65, 0.88}, 0.3},
                 {{0.73, 0.08, 0.96}, 0.8}};
    std::vector<std::vector<double>> input{{0.91, 0.22, 0.63}, {0.24, 0.65, 0.88}, {0.73, 0.08, 0.96}};
    std::cout << "INPUT: ";
    for (auto&[in, out]: data) {
        std::cout << "[ ";
        for (auto &i: in)
            std::cout << i << ",";
        std::cout << out << " ]";
    }
    std::cout << std::endl;
    nn->train(data, 22, 1, 0.2, true);
    auto predicted = nn->predict(input, false);
    std::cout << std::endl << "RESULT: ";
    for(auto& elem: predicted) {
        std::cout << "[ ";
        for (auto& elemm : elem)
            std::cout << elemm << ",";
        std::cout << " ]";
    }
    std::cout << std::endl;
    return 0;
}