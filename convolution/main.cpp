//#include<iostream>
//#include<limits>
//#include<vector>
//
//#include<iostream>
//#include<chrono>
//#include "convolution.cpp"
//#include <immintrin.h>
//#include <xmmintrin.h>
//#include<cstdlib>
//#define DEBUG
//int main() {
//	//F(2X2,3X3)
//	const int N = 10;
//	const int C = 100;
//	const int H = 100;
//	const int W = 100;
//	const int R = 3; // Filter size
//	const int K = 10; // Number of filters
//
//	std::vector<float> input(N * C * H * W);
//	for (int i = 0; i < N * C * H * W; ++i) {
//		input[i] = rand() % 10;
//		//input[i] = 1.0f;
//	}
//	// Define filters
//	std::vector<float> filters(C * R * R * K);
//	for (int i = 0; i < C * R * R * K; ++i) {
//		filters[i] = rand() % 10;
//		//filters[i] = 1;
//
//
//	}
//	float* Y = new float[N * K * (H - R+1) * (W - R+1)];
//	auto start = std::chrono::high_resolution_clock::now();
//	 Conv2dNaive(input.data(), filters.data(), N, C, H, W, K, R);
//	auto end = std::chrono::high_resolution_clock::now();
//	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
//	std::cout << "Time taken: " << duration.count() / 1e6 << "ms" << std::endl;
//	 start = std::chrono::high_resolution_clock::now();
//
//	winogradCall(2, 3, N, K, C, H, W, input.data(),0, 0, Y, filters.data());
// end = std::chrono::high_resolution_clock::now();
// duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
//std::cout << "Time taken: " << duration.count() / 1e6 << "ms" << std::endl;
//start = std::chrono::high_resolution_clock::now();
//Conv2dWinograd(input.data(), filters.data(), N, H, W, R);
//
//end = std::chrono::high_resolution_clock::now();
//duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
//std::cout << "Time taken: " << duration.count() / 1e6 << "ms" << std::endl;
//
////print(Y, N, K, H, W, R);
//
//
//
//
//
//
//
//
//
//
//
//
//
//	return 0;
//	
//}
//
//
////==
//

#include <cblas.h>  // Header for OpenBLAS routines
#include <vector>
#include <random>
#include <iostream>
#include <chrono>

int main() {
    size_t m = 2048, n = 2048, k = 2048;  // Matrix dimensions
    double alpha = 1.0, beta = 0.0;  // Coefficients for BLAS operations

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);

    // Initialize matrices A, B, and C
    std::vector<float> A(m * k);  // Column-major storage
    std::vector<float> B(k * n);
    std::vector<float> C(m * n, 0.0);

    // Fill matrices with random values
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            A[i * k + p] = dis(gen);
        }
    }

    for (size_t p = 0; p < k; ++p) {
        for (size_t j = 0; j < n; ++j) {
            B[p * n + j] = dis(gen);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Perform MMM using OpenBLAS
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
        m, n, k, alpha,
        &A[0], k,  // Matrix A
        &B[0], n,  // Matrix B
        beta,
        &C[0], m);  // Matrix C

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Output the execution time
    std::cout << "OpenBLAS MMM execution time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}