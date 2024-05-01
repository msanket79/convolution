#include<iostream>
#include<limits>
#include<vector>

#include<iostream>
#include<chrono>
#include "winogradConv.cpp"
#include <immintrin.h>
#include <xmmintrin.h>


template<typename Func, typename... Args>
double timeit(Func func, Args&&... args) {
	auto start = std::chrono::high_resolution_clock::now();
	func(std::forward<Args>(args)...);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	std::cout << "Time taken: " << duration.count() / 1e6 << "ms" << std::endl;
	return duration.count() / 1e6;
}

void test(float* A, float* B, int n) {
	int flag = 0;
	for (int i = 0; i < n; i++) {
		if (std::abs(A[i] - B[i]) > 1e-4) {
			flag = 1;
			break;
		}
	}
	if (flag == 1) {
		std::cout << "Test fail\n";
	}
	else std::cout << "Test passed\n";
}


void print(float* ans, int N, int K, int H, int W, int R) {
	for (int n = 0; n < N; ++n) {
		for (int k = 0; k < K; ++k) {
			std::cout << "Output for filter " << k << " (batch " << n << "):" << std::endl;
			for (int i = 0; i < (H - R + 1) * (W - R + 1); ++i) {
				std::cout << ans[(n * K + k) * ((H - R + 1) * (W - R + 1)) + i] << " ";
				if ((i + 1) % (H - R + 1) == 0) {
					std::cout << std::endl;
				}
			}
			std::cout << std::endl;
		}
	}
}

float* B = new float[16] {1, 0, 0, 0,
0, 1, -1, 1,
-1, 1, 1, 0,
0, 0, 0, -1};

float* G = new float[12] {1, 0, 0,
0.5, 0.5, 0.5,
0.5, -0.5, 0.5,
0, 0, 1};

float* A = new float[8] {1, 0,
1, 1,
1, -1,
0, -1};

float* B_T = new float[16] {1, 0, -1, 0
, 0, 1, 1, 0,
0, -1, 1, 0,
0, 1, 0, -1};
float* G_T = new float[12] {1, 0.5, 0.5, 0,
0, 0.5, -0.5, 0,
0, 0.5, 0.5, 1};
float* A_T = new float[8] {1, 1, 1, 0,
0, 1, -1, -1};




float* Conv2dNaive(float* I, float* f, int N, int C, int H, int W, int K, int R) {
	assert(H == W);
	assert(R == 3);

	int a = W - R + 1;
	float* output = new float[N * K * a * a];
	for (int k = 0; k < K; k++) {
		for (int n = 0; n < N; n++) {
			for (int c = 0; c < C; c++) {
				for (int h = 0; h < a; h++) {
					for (int w = 0; w < a; w++) {
						float ans = 0;
						for (int i = 0; i < R; i++) {

							for (int j = 0; j < R; j++) {
								ans += f[k * (C * R * R) + c * (R * R) + i * R + j] * I[n * (C * H * W) + c * (H * W) + (h + i) * W + w + j];
							}
						}
						output[n * (K * a * a) + k * (a * a) + h * a + w] = ans;
					}
				}
			}
		}
	}
	return output;
}
void mul_inplace(float* A, const float* B, float* C, const int m, int n) {
	//#pragma omp parallel
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			C[i * n + j] = A[i * n + j] * B[i * n + j];
		}
}
void Dot(const float* A, const float* B, float* C, const int m, const int n, const int k) {
	cblas_sgemm(
		CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k,
		1.0,
		A, k,
		B, n,
		0.0,
		C, n
	);
}
inline float* filterTransform(const float* f, const int R) {
	////filter transform  [GgG.T]
	float* U_temp = (float*)malloc(sizeof(float) * 4 * 3);
	float* U = (float*)malloc(sizeof(float) * 4 * 4);
	Dot(G, f, U_temp, 4, 3, 3);
	Dot(U_temp, G_T, U, 4, 4, 3);
	delete[]U_temp;
	return U;
}
////input transform [B.TDB]
inline float* inputTransform(const float* I, const int N, const int H, const int W) {
	int T = (H - 2) / 2;
	float* V = (float*)malloc(sizeof(float) * N * T * T * 4 * 4);
	float* V_temp = new float[N * 4 * 4];
	float* V2 = (float*)malloc(sizeof(float) * N * 4 * 4);
	for (int n = 0; n < N; n++) {
		unsigned long long n_idx_in = n * (H * W);
		unsigned long long n_idx_out = n * (T * T * 16);
		for (int th = 0; th < T; th++) {
			for (int tw = 0; tw < T; tw++) {
				unsigned long long vh = th * 2;
				unsigned long long vw = tw * 2;
				unsigned long long ind = n_idx_in + (vh)*W + vw;
				std::memcpy(V_temp + n * 16, I + ind, 16);
				std::memcpy(V_temp + n * 16 + 4, I + ind + W, 16);
				std::memcpy(V_temp + n * 16 + 8, I + ind + 2 * W, 16);
				std::memcpy(V_temp + n * 16 + 12, I + ind + 3 * W, 16);
				Dot(B_T, V_temp + n * 16, V2 + n * 16, 4, 4, 4);
				Dot(V2 + n * 16, B, V_temp + n * 16, 4, 4, 4);
				unsigned long long outind = n_idx_out + th * (T * 16) + tw * 16;
				std::memcpy(V + outind, V_temp + n * 16, 64);

			}
		}
	}
	delete[]V_temp;
	delete[]V2;
	return V;
}

inline float* inverseTransform(const float* V, const int N, const int T, const int a) {
	float* out = (float*)malloc(sizeof(float) * N * a * a);
	float* M = (float*)malloc(sizeof(float) * N * 16);
	float* M1 = (float*)malloc(sizeof(float) * N * 2 * 4);

	for (int n = 0; n < N; n++) {
		unsigned long long n_idx_in = n * (T * T * 16);
		unsigned long long n_idx_out = n * (a * a);
		for (int th = 0; th < T; th++) {
			for (int tw = 0; tw < T; tw++) {
				unsigned long long inind = n_idx_in + th * (T * 16) + tw * 16;
				std::memcpy(M + n * (16), V + inind, 64);
				Dot(A_T, M + n * (16), M1 + n * 8, 2, 4, 4);
				Dot(M1 + n * 8, A, M + n * (16), 2, 2, 4);
				unsigned long long outind = n_idx_out + th * 2 * a + tw * 2;
				std::memcpy(out + outind, M + n * (16), 8);
				std::memcpy(out + outind + a, M + n * (16) + 2, 8);
			}
		}
	}
	delete[]M;
	delete[]M1;
	return out;
}

float* Conv2dWinograd(const float* I, const float* f, const int N, const int H, const int W, const int R) {
	assert(R == 3);
	assert(H == W);
	if (H % 2 != 0) {
		std::cout << "blocks are not aligned\n";
		exit(0);
		return NULL;
	}
	float* U, * V, * out;
	int T = (H - 2) / 2;
	int a = H - R + 1;



	////filter transform  [GgG.T]
	U = filterTransform(f, R);



	////input transform [B.TDB]
	V = inputTransform(I, N, H, W);
	////M=UoV; here M is V; now apply input transform on it;
#pragma omp parallel for 
	for (int n = 0; n < N; n++) {
		int n_idx_out = n * (T * T * 16);
		for (int th = 0; th < T; th++) {
			for (int tw = 0; tw < T; tw++) {
				unsigned long long outind = n_idx_out + th * (T * 16) + tw * 16;
				mul_inplace(V + outind, U, V, 4, 4);
			}


		}
	}

	out = inverseTransform(V, N, T, a);

	delete[]V;
	delete[]U;
	return out;

}

//--------------------------------------------------------------
void add_inplace(float* A, const float* B, float* C, const int m, int n) {
	//#pragma omp parallel
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			C[i * n + j] = A[i * n + j] + B[i * n + j];
		}
}



inline float* filterTransform3d(const float* f, const int K, const int C, const int R) {
	////filter transform  [GgG.T]
	float* U_temp = (float*)malloc(sizeof(float) * 4 * 3);
	float* U = (float*)malloc(sizeof(float) * K * C * 4 * 4);

	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			int ind = k * C * R * R + c * R * R;
			Dot(G, f + ind, U_temp, 4, 3, 3);
			Dot(U_temp, G_T, U + k * C * 4 * 4 + c * 4 * 4, 4, 4, 3);
		}
	}
	delete[]U_temp;
	return U;
}
////input transform [B.TDB]
inline float* inputTransform3d(const float* I, const int N, const int C, const int H, const int W) {
	int T = (H - 2) / 2;
	float* V = (float*)malloc(sizeof(float) * N * C * T * T * 4 * 4);
	float* V_temp = new float[4 * 4];
	float* V2 = (float*)malloc(sizeof(float) * 4 * 4);
	for (int n = 0; n < N; n++) {
		for (int c = 0; c < C; c++) {
			unsigned long long n_idx_in = n * (H * W * C) + c * (H * W);
			unsigned long long n_idx_out = c * (N * T * T * 16) + n * (T * T * 16);
			for (int th = 0; th < T; th++) {
				for (int tw = 0; tw < T; tw++) {
					unsigned long long vh = th * 2;
					unsigned long long vw = tw * 2;
					unsigned long long ind = n_idx_in + (vh)*W + vw;
					std::memcpy(V_temp, I + ind, 16);
					std::memcpy(V_temp + 4, I + ind + W, 16);
					std::memcpy(V_temp + 8, I + ind + 2 * W, 16);
					std::memcpy(V_temp + 12, I + ind + 3 * W, 16);
					Dot(B_T, V_temp, V2, 4, 4, 4);
					Dot(V2, B, V_temp, 4, 4, 4);
					unsigned long long outind = n_idx_out + th * (T * 16) + tw * 16;
					std::memcpy(V + outind, V_temp, 64);
					//for (int i = 0; i < 16; i++) std::cout << V_temp[i]<<" ";
					//std::cout << std::endl;

				}
			}
		}
	}
	delete[]V_temp;
	delete[]V2;
	return V;
}

inline float* inverseTransform3d(const float* V, const int N, const int T, const int a, const int K) {
	//V= [N,K,T,T,4,4];
	float* out = (float*)malloc(sizeof(float) * N * K * a * a);
	float* M = (float*)malloc(sizeof(float) * 16);
	float* M1 = (float*)malloc(sizeof(float) * 2 * 4);

	for (int n = 0; n < N; n++) {
		for (int k = 0; k < K; k++) {
			unsigned long long n_idx_in = n * (T * T * 16 * K) + k * (T * T * 16);
			unsigned long long n_idx_out = n * (a * a * K) + k * (a * a);
			for (int th = 0; th < T; th++) {
				for (int tw = 0; tw < T; tw++) {
					unsigned long long inind = n_idx_in + th * (T * 16) + tw * 16;
					std::memcpy(M, V + inind, 64);
					Dot(A_T, M + n * (16), M1 + n * 8, 2, 4, 4);
					Dot(M1 + n * 8, A, M + n * (16), 2, 2, 4);
					unsigned long long outind = n_idx_out + th * 2 * a + tw * 2;
					std::memcpy(out + outind, M, 8);
					std::memcpy(out + outind + a, M + 2, 8);
				}
			}
		}
	}
	delete[]M;
	delete[]M1;
	return out;
}


float* Conv3dWinograd(const float* I, const float* f, const int N, const int C, const int H, const int W, const int K, const int R) {
	assert(R == 3);
	assert(H == W);
	if (H % 2 != 0) {
		std::cout << "blocks are not aligned\n";
		exit(0);
		return NULL;
	}
	float* U, * V, * out;
	int T = (H - 2) / 2;
	int a = H - R + 1;



	////filter transform  [GgG.T]
	U = filterTransform3d(f, K, C, R);
	//for (int k = 0; k < K; k++) {
	//	std::cout << "k = " << k + 1 << "\n";
	//	for (int c = 0; c < C; c++) {
	//		std::cout << "c = " << c + 1 << "\n";
	//		for (int i = 0; i < 4; i++) {
	//			for (int j = 0; j < 4; j++) {
	//				std::cout << U[k * (C * 4 * 4) + c * (4 * 4) + i * 4 + j] << " ";
	//			}
	//			std::cout << "\n";
	//		}
	//	}
	//}



	////input transform [B.TDB]
	V = inputTransform3d(I, N, C, H, W);
	//N *C* T * T * 4 * 4
	//for (int n = 0; n < N; n++){
	//	std::cout << "n = " << n + 1 << "\n";
	//	for (int c = 0; c < C; c++) {
	//		std::cout << "c = " << c + 1 << "\n";
	//		for (int th = 0; th < T; th++) {
	//			for (int tw = 0; tw < T; tw++) {
	//				for(int i=0;i<16;i++) std::cout << V[c * (N * T * T * 4 * 4) + n * (T * T * 4 * 4) + th * (T * 4 * 4) + tw * 16+i] << " ";
	//			}
	//			std::cout << "\n";
	//		}
	//	}
	//}
	////M=UoV; here M is V; now apply input transform on it;
	//V=[C N T T 4 4]
	//U=[K C 4 4]
	float* M = new float[N * K * T * T * 4 * 4] {0};
	for (int k = 0; k < K; k++) {
		for (int c = 0; c < C; c++) {
			for (int n = 0; n < N; n++) {
				int n_idx_out = c * (N * T * T * 16) + n * (T * T * 16);
				int add_idx_out = n * (K * T * T * 4 * 4) + k * (T * T * 4 * 4);
				for (int th = 0; th < T; th++) {
					for (int tw = 0; tw < T; tw++) {
						unsigned long long outind = n_idx_out + th * (T * 16) + tw * 16;
						unsigned long long addind = add_idx_out + th * (T * 16) + tw * 16;
						//multiply kia tile ko filter transform ke saath for vaapis V me store kardia abhi kya karunga waha se uthake copy kardunga M me
						mul_inplace(V + outind, U + k * (C * 4 * 4) + c * (4 * 4), V + outind, 4, 4);
						add_inplace(V + outind, M + addind, M + addind, 4, 4);
						std::cout << "\n";
					}


				}
			}
		}

	}
	//	for (int n = 0; n < N; n++){
	//	std::cout << "n = " << n + 1 << "\n";
	//	for (int k = 0; k < C; k++) {
	//		std::cout << "k = " << k + 1 << "\n";
	//		for (int th = 0; th < T; th++) {
	//			for (int tw = 0; tw < T; tw++) {
	//				for(int i=0;i<16;i++) std::cout << M[n * (C * T * T * 4 * 4) + k * (T * T * 4 * 4) + th * (T * 4 * 4) + tw * 16+i] << " ";
	//			}
	//			std::cout << "\n";
	//		}
	//	}
	//}

	out = inverseTransform3d(M, N, T, a, K);

	delete[]V;
	delete[]U;
	return out;

}


//Y=A.T[(GgG.T)o(B.TdB)]A

float* mul(float* A, float* B, int m, int n) {
	float* C = new float[m * n];
#pragma omp parallel for
	for (int i = 0; i < m * n; i++) {
		C[i] = A[i] * B[i];
	}
	return C;
}
//float* winograd_f_2_3(float* I, float* F) {
//	float* U_temp = Dot(G, F, 4, 3, 3);
//	float*	U = Dot(U_temp, G_T, 4, 4, 3);
//	delete[]U_temp;
//	float* V_temp = Dot(B_T,I, 4, 4, 4);
//	float*V = Dot(V_temp, B, 4, 4, 4);
//	delete[]V_temp;
//	float* M = mul(U, V, 4, 4);
//	delete[]U;
//	delete[]V;
//	float* out = Dot(A_T, M, 2, 4, 4);
//	delete[]M;
//	out = Dot(out, A, 2, 2, 4);
//	return out;
//
//
//
//}
// Notations
	// N-no of images
	// C - number of channels
	// H - input width 
	// W -input width
	// K- output channels
	// r- filter size
void printImg(float* I,int N,int C,int H,int W){
	for(int i=0;i<N;i++){
		std::cout<<"-----N== "<<i<< " ------\n";
		for(int j=0;j<C;j++){
			std::cout<<"-----C== "<<j<< " ------\n";
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					std::cout<<I[i*(C*H*W)+j*(H*W)+h*(W)+w]<<" ";
				}
				std::cout<<"\n";


			}
		}
	}
}
void printFilter(float* I,int K,int C,int H,int W){
	for(int i=0;i<K;i++){
		std::cout<<"-----K== "<<i<< " ------\n";
		for(int j=0;j<C;j++){
			std::cout<<"-----C== "<<j<< " ------\n";
			for(int h=0;h<H;h++){
				for(int w=0;w<W;w++){
					std::cout<<I[i*(C*H*W)+j*(H*W)+h*(W)+w]<<" ";
				}
				std::cout<<"\n";


			}
		}
	}
}


int main() {
	const int N = 1;
	const int C = 1;
	const int H = 4;
	const int W = 4;
	const int R = 3; // Filter size
	const int K = 1; // Number of filters

	std::vector<float> input(N * C * H * W);
	for (int i = 0; i < N * C * H * W; ++i) {
		input[i] = i*i;
		//input[i] = 1.0f;
	}
	float* a1 = new float[16] {1, 2, 3, 4,
		2, 3 ,4 ,5,
		3 ,4 ,5 ,6,
		4 ,5, 6, 7 };
	printImg(input.data(),N,C,W,W);
	// Define filters
	std::vector<float> filters(C * R * R * K);
	for (int i = 0; i < C * R * R * K; ++i) {
		filters[i] = 1;


	}
	float* A1 = inputTransform(input.data(), N, H, W);
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			std::cout<<A1[i*4+j]<<" ";
		}
		std::cout<<std::endl;
	}
	float* Y = new float[N * K * (H - R+1) * (W - R+1)];
	auto start = std::chrono::high_resolution_clock::now();
	float*filt=filterTransform3d(filters.data(), K, C, R);
	//for (int i = 0; i < 4; i++) {
	//	for (int j = 0; j < 4; j++) {
	//		std::cout << filt[i * 4 + j] << " ";
	//	}
	//	std::cout << std::endl;
	//}	

	winogradCall(2, 3, N, K, C, H, W, input.data(),0, 0, Y, filters.data());
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	print(Y, N, K, H, W, R);
	std::cout << "Time taken: " << duration.count() / 1e6 << "ms" << std::endl;
	// Print the output
	int a = N * (W - R + 1) * (H - R + 1) * K;
	//test(ans, out, a);
	//print(ans, N, K, H, W, R);












	return 0;
	
}


//==

