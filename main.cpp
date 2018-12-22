#include <iostream>

#include "neuralnetwork.h"

int main() {
    NNBuilder builder;
    auto* nn = builder
            .start()
            .addLayer(3, 0, ActiFuncType::SIGM)
            .addLayer(4, 0.5, ActiFuncType::SIGM)
            .addLayer(3, -0.5, ActiFuncType::RELU)
            .build();
    std::vector<double> inputs{0.91, 0.22, 0.63};
    std::cout << "INPUT: ";
    for(auto input: inputs)
        std::cout << "[ " << input << " ]";
    std::cout << std::endl;
    std::vector<double> results = nn->feedForward(inputs, true);
    std::cout << std::endl << "RESULT: ";
    for(auto result: results)
        std::cout << "[ " << result << " ]";
    std::cout << std::endl;
    return 0;
}