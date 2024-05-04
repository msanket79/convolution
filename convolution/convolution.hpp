
void filterTransform2X2_3X3(int m, int r, int c, int k, float* F, int ldF1, int ldF2, int ldF3, float* U);
void Winograd2X2_3X3(
	int m, int r, int n, int k, int c,
	int h, int w,
	int vpadding, int hpadding,
	float* D, int ldD1, int ldD2, int ldD3,
	float* Y, int ldY1, int ldY2, int ldY3,
	float* U, float* V, float* M

);
inline void winogradCall(int m, int r, int n, int k, int c, int h, int w, float* D, int vpadding, int hpadding, float* Y, float* F);

float* Conv2dNaive(float* I, float* f, int N, int C, int H, int W, int K, int R);
float* Conv2dWinograd(const float* I, const float* f, const int N, const int H, const int W, const int R);