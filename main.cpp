#include <iostream>

#include "neuralnetwork.h"

int main() {
    NNBuilder builder;
    auto* nn = builder
            .start()
            .addLayer(3, 0, ActiFuncType::SIGM)
            .addLayer(4, 0.5, ActiFuncType::SIGM)
            .addLayer(3, -0.5, ActiFuncType::RELU)
            .finish();
    std::vector<double> inputs{1, 2, 3};
    std::vector<double> results = nn->feedForward(inputs);
    for(auto result: results)
        std::cout << "[ " << result << " ]";
    std::cout << std::endl;
    return 0;
}