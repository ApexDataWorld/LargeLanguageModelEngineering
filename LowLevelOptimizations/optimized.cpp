
#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(int iterations, int param1, int param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; ++i) {
        int j = i * param1 - param2;
        result -= 1.0 / j;
        j = i * param1 + param2;
        result += 1.0 / j;
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(100000000, 4, 1) * 4;
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> execution_time = end_time - start_time;

    std::cout << "Result: " << std::fixed << std::setprecision(12) << result << std::endl;
    std::cout << "Execution Time: " << execution_time.count() << " seconds" << std::endl;
    
    return 0;
}
