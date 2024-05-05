// we are implementing winograd convolution using AVX2
//  the imaage in HCHW format
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <assert.h>
#include<iostream>
#include<ctime>
#include "gemm.hpp"
#include "avx_transpose.hpp"
#define DEBUG



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


#define min(a, b) ((a) > (b) ? (b) : (a))
#define max(a, b) ((a) > (b) ? (a) : (b))

// for accessing U,V,M
// U contains the transformed filters U=[GgG.T]
#define Urow(a1, a2, a3, a4) U[(a1) * (ldU1) + (a2) * (ldU2) + (a3) * (ldU3) + (a4)]
// V contains the transformed inputs V=[B.TDB]      
#define Vrow(a1, a2, a3, a4) V[(a1) * (ldV1) + (a2) * (ldV2) + (a3) * (ldV3) + (a4)]
// M contains the U@V
#define Mrow(a1, a2, a3, a4) M[(a1) * (ldM1) + (a2) * (ldM2) + (a3) * (ldM3) + (a4)]

// input matrix D
#define Drow(a1, a2, a3, a4) D[(a1) * (ldD1) + (a2) * (ldD2) + (a3) * (ldD3) + (a4)]
// filter matrix F
#define Frow(a1, a2, a3, a4) F[(a1) * (ldF1) + (a2) * (ldF2) + (a3) * (ldF3) + (a4)]
// output matrix Y
#define Yrow(a1, a2, a3, a4) Y[(a1) * (ldY1) + (a2) * (ldY2) + (a3) * (ldY3) + (a4)]

// overloading for adding
__m128 operator+(__m128 a, __m128 b) { return _mm_add_ps(a, b); }

__m128 operator-(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }

__m128 operator*(float a, __m128 b)
{

    __m128 c = _mm_set_ps1(a);
    return _mm_mul_ps(c, b);
}
__m256 operator+(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }

__m256 operator-(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }

__m256 operator*(float a, __m256 b)
{
    __m256 c = _mm256_set1_ps(a);
    return _mm256_mul_ps(c, b);
}




// Filter dimension is T *T *k*C
void filterTransform2X2_3X3(int m, int r, int c, int k, float* F, int ldF1, int ldF2, int ldF3, float* U)
{
    assert(r == 3);
    assert(m == 2);
    int t = m + r - 1; // w inograd input tile size
    int ik, ic, ldU1, ldU2, ldU3, i, j;
    __m128 _F0, _F1, _F2, _W0, _W1, _W2, _W3, _U0, _U1, _U2, _U3;
    float temp[16];

    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;
    // for debug purposes


#pragma omp parallel for collapse(2) private(ik, ic, _F0, _F1, _F2, _W0, _W1, _W2, _W3,_U0, _U1, _U2, _U3, i,temp) if ((k * c > 1))
    for (ik = 0; ik < k; ik++)
    {
        std::cout << ik << " ";
        for (ic = 0; ic < c; ic++)
        {
            // U[ic,ik]=G @ F @ G.t
            // lets load the filter F[k,c] rows into filter vectors
            // yaha memory me problem ho sakti thi because vector reads 4 floats at once hence alag alag read kara hai
            // here the last value in _F0,_F1 _F3 is garbage(0) and will not be used in futher dots
            _F0 = _mm_set_ps( 0,Frow(ik, ic, 0, 2), Frow(ik, ic, 0, 1),Frow(ik, ic, 0, 0));
            _F1 = _mm_set_ps(0,Frow(ik, ic, 1, 2), Frow(ik, ic, 1, 1), Frow(ik, ic, 1, 0));
            _F2 = _mm_set_ps(0,Frow(ik, ic, 2, 2), Frow(ik, ic, 2, 1), Frow(ik, ic, 2, 0));
 
            // Wi  = G_row(i)  *  [ _F0;_F1;_F2 ] (rows of F) with
            // G = [1.0,  0.0, 0.0,
            //      0.5,  0.5, 0.5,
            //      0.5, -0.5, 0.5,
            //      0.0,  0.0, 1.0];

            _W0 = _F0; // because G[0]=1,0,0 
            _W1 = 0.5f * (_F0 + _F1 + _F2);
            _W2 = 0.5f* (_F0 - _F1 + _F2);
            _W3 = _F2;


            //here w3 will contain garbage values which we will never use as the last 
            _MM_TRANSPOSE4_PS(_W0, _W1, _W2, _W3);

            _U0 = _W0;
            _U1 = 0.5f * (_W0 + _W1 + _W2);
            _U2 = 0.5f * ((_W0 - _W1) + _W2);
            _U3 = _W2;
            _mm_storeu_ps(temp, _U0);
            _mm_storeu_ps(temp+4, _U1);
            _mm_storeu_ps(temp+8, _U2);
            _mm_storeu_ps(temp+12, _U3);

            for (int i = 0; i < t; i++) {

                Urow(i, 0, ik, ic) = temp[i];
                Urow(i, 1, ik, ic) = temp[4 + i];
                Urow(i, 2, ik, ic) = temp[8 + i];
                Urow(i, 3, ik, ic) = temp[12 + i];
            }

        }
    }


}

void Winograd2X2_3X3(
    int m, int r, int n, int k, int c,
    int h, int w,
    int vpadding, int hpadding,
    float* D, int ldD1, int ldD2, int ldD3,
    float* Y, int ldY1, int ldY2, int ldY3,
    float* U, float* V, float* M

)
{
    assert(r == 3);
    assert(m == 2);
    int t = m + r - 1; // w inograd input tile size
    int s=m;
    const int vstride=1; 
    const int hstride=1;

    int tile_h,tile_w,ik,ic,in,ih,iw,hh,ww,hh_,ww_,fh,fw,oh,ow
    ,i,j,ho,wo,e,v;
    int  ldU1, ldU2, ldU3,
        ldV1, ldV2, ldV3,
        ldM1, ldM2, ldM3;

    __m128 d0, d1, d2, d3,
        U0, U1, U2, U3,
        _M0, _M1, _M2, _M3,
        W0, W1, W2, W3;

    ho=(h+2*vpadding-r)/vstride+1;
    wo=(w+2*hpadding-r)/hstride+1;

    tile_h = ceil(((double) h + 2 * vpadding - t) / s) + 1; //no of tiles in height 
    tile_w = ceil(((double) w + 2 * hpadding - t) / s) + 1; //no of tiles in width

    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;

    ldV3 = (n * tile_h * tile_w);
    ldV2 = c * ldV3;
    ldV1 = t * ldV2;

    ldM3 = (n * tile_h * tile_w);
    ldM2 = k * ldM3;
    ldM1 = t * ldM2;
    float temp_M[4 * 4];
    float Z[4];
    float temp_W[2 * 4];
    float temp0[4];
    float temp1[4];
    float temp2[4];
    float temp3[4];



    #pragma omp parallel for collapse(2) private(in,ic,ih,hh_,hh,fh,oh,iw,ww_,ww,fw,ow,d0,d1,d2,d3,W0,W1,W2,W3,U0,U1,U2,U3,i,j,temp1,temp2,temp3,temp0) if((n*c)>1)
        for(in=0;in<n;in++){
            for(ic=0;ic<c;ic++){
                for(ih=0;ih<tile_h;ih++){
                    hh_=min(h,ih*s-vpadding);
                    hh=max(hh_,0);
                    fh = min(max(-hh_, 0), t);
                    oh = max(min(h - hh, t), 0);
                    oh = oh < t ? oh + fh : oh;

                    for(iw=0;iw<tile_w;iw++){
                        ww_ = min(w, iw * s - hpadding);
                        ww = max(ww_, 0);
                        fw = min(max(-ww_, 0), t);
                        ow = max(min(w - ww, t), 0);
                        ow = ow < t ? ow + fw : ow;

                        for(j=0;j<4;j++){
                            temp0[j]=(fh<=0 && 0< oh && fw<=j && j<ow) ? Drow(in,ic,hh+0-fh,ww+j-fw):0;
                            temp1[j]=(fh<=1 && 1< oh && fw<=j && j<ow) ? Drow(in,ic,hh+1-fh,ww+j-fw):0;
                            temp2[j]=(fh<=2 && 2< oh && fw<=j && j<ow) ? Drow(in,ic,hh+2-fh,ww+j-fw):0;
                            temp3[j]=(fh<=3 && 3< oh && fw<=j && j<ow) ? Drow(in,ic,hh+3-fh,ww+j-fw):0;
                        }

                        d0=_mm_loadu_ps(temp0);
                        d1 = _mm_loadu_ps(temp1);
                        d2 = _mm_loadu_ps(temp2);
                        d3 = _mm_loadu_ps(temp3);

                        W0=d0-d2;
                        W1=d1+d2;
                        W2=d2-d1;
                        W3=d1-d3;

                        _MM_TRANSPOSE4_PS(W0,W1,W2,W3);

                        U0=W0-W2;
                        U1=W1+W2;
                        U2=W2-W1;
                        U3=W1-W3;
                        _mm_storeu_ps(temp0,U0);
                        _mm_storeu_ps(temp1,U1);
                        _mm_storeu_ps(temp2,U2);
                        _mm_storeu_ps(temp3,U3);

                        for( i=0;i<4;i++){
                            Vrow(i,0,ic,in*tile_h*tile_w+ih*tile_w+iw)=temp0[i];
                            Vrow(i,1,ic,in*tile_h*tile_w+ih*tile_w+iw)=temp1[i];
                            Vrow(i,2,ic,in*tile_h*tile_w+ih*tile_w+iw)=temp2[i];
                            Vrow(i,3,ic,in*tile_h*tile_w+ih*tile_w+iw)=temp3[i];

                        }

                    }


                }
            }
        }
        std::cout << " input transform complete\n";
#ifdef DEBUG
        double t1, t2, T1;
            t1 = clock();
#endif
        #pragma omp parallel for collapse(2) private(e,v)
        for (e = 0; e < t; e++) {
            for (v = 0; v < t; v++) {
                // M[e,v] = U[e,v] @ V[e,v];
                //gemm()
                gemm('R','R','R', 'N', 'N',
                    k, (n * tile_h * tile_w), c,
                    1.0, &Urow(e, v, 0, 0), c,
                    &Vrow(e, v, 0, 0), (n * tile_h * tile_w),
                    0.0, &Mrow(e, v, 0, 0), (n * tile_h * tile_w)
                );
                
            }
        }
#ifdef DEBUG
        t2 = clock();
        std::cout << "time for  winograd convo " << (static_cast<double>(t2 - t1) / CLOCKS_PER_SEC)*1000.0 << "ms \n";
#endif
        std::cout << "M calculated\n";

       
        #pragma omp parallel for collapse(2) private(in,ik,ih,iw,_M0,_M1,_M2,_M3,hh,ww,i,j,temp_M,temp_W,Z) if((n*k)>1)
        for (in = 0; in < n; in++)
            for (ik = 0; ik < k; ik++)
                for (ih = 0; ih < tile_h; ih++)
                    for (iw = 0; iw < tile_w; iw++) {

                        for (i = 0; i < 4; i++) {
                            temp_M[i] = Mrow(i, 0, ik, in * tile_h * tile_w + ih * tile_w + iw);
                            temp_M[4 + i] = Mrow(i, 1, ik, in * tile_h * tile_w + ih * tile_w + iw);
                            temp_M[8 + i] = Mrow(i, 2, ik, in * tile_h * tile_w + ih * tile_w + iw);
                            temp_M[12 + i] = Mrow(i, 3, ik, in * tile_h * tile_w + ih * tile_w + iw);
                        }
 

                        _M0 = _mm_load_ps(temp_M);
                        _M1 = _mm_load_ps(temp_M + 4);
                        _M2 = _mm_load_ps(temp_M + 8);
                        _M3 = _mm_load_ps(temp_M + 12);

                        W0 = _M0 + _M1 + _M2;
                        W1 = _M1 - _M2 - _M3;

                        _mm_store_ps(temp_W, W0);
                        _mm_store_ps(temp_W + 4, W1);

                        Z[0] = temp_W[0] + temp_W[1] + temp_W[2];
                        Z[1] = temp_W[1] - temp_W[2] - temp_W[3];
                        Z[2] = temp_W[4 + 0] + temp_W[4 + 1] + temp_W[4 + 2];
                        Z[3] = temp_W[4 + 1] - temp_W[4 + 2] - temp_W[4 + 3];
                        hh = ih * s;
                        ww = iw * s;
                        for (i = 0; i < min(m, ho - hh); i++)
                            for (j = 0; j < min(m, wo - ww); j++)
                                Yrow(in, ik, hh + i, ww + j) = Z[j * m + i];


                    }
        std::cout << "output cal\n";
}


inline void winogradCall(int m, int r, int n, int k, int c, int h, int w, float* D, int vpadding, int hpadding, float* Y, float* F) {
    assert(r == 3);
    assert(m == 2);
    int t = m + r - 1;
    int a1 = (h + 2 * vpadding - r) + 1;
    int a2 = (w + 2 * hpadding - r) + 1;
    int ldF3, ldF2, ldF1,
        ldD1, ldD2, ldD3,
        ldY1, ldY2, ldY3;
    int tile_H = ceil(((double)h + 2 * vpadding - t) / m) + 1;
    int tile_W = ceil(((double)w + 2 * hpadding - t) / m) + 1;
    
    float* U = (float*)malloc(sizeof(float) * t * t * k * c);
    float* V = (float*)malloc(sizeof(float) * t * t * c * (n * tile_H * tile_W));
    float* M = (float*)malloc(sizeof(float) * t * t * k * (n * tile_H * tile_W));


    ldD3 = w;
    ldD2 = h * ldD3;
    ldD1 = c * ldD2;

    ldF3 = r;
    ldF2 = r * ldF3;
    ldF1 = c * ldF2;

    ldY3 = a2;
    ldY2 = (a1)*ldY3;
    ldY1 = ldY2 * k;
    filterTransform2X2_3X3(m, r, c, k, F, ldF1, ldF2, ldF3, U);
    

    Winograd2X2_3X3(m, r, n, k, c, h, w, vpadding, hpadding, D, ldD1, ldD2, ldD3, Y, ldY1, ldY2, ldY3, U, V, M);
    delete[]U;
    delete []M;
    delete[]V;

}

//------------------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------naive-------------------------------------------------------------


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



inline float* Conv2dNaive(float* I, float* f, int N, int C, int H, int W, int K, int R) {
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
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << V[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }
    out = inverseTransform(V, N, T, a);

    delete[]V;
    delete[]U;
    return out;

}


void add_inplace(float* A, const float* B, float* C, const int m, int n) {
    //#pragma omp parallel
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
}

