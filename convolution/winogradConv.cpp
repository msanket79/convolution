// we are implementing winograd convolution using AVX2
//  the imaage in HCHW format
#include <cblas.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <assert.h>
#include<iostream>
#include "avx_transpose.hpp"

// defines used

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
#ifdef DEBUG
    doouble t1, t2, T1,
        t1 = clock();
#endif

#pragma omp parallel for collapse(2) private(ik, ic, _F0, _F1, _F2, _W0, _W1, _W2, _W3,_U0, _U1, _U2, _U3, i,temp) if ((k * c > 1))
    for (ik = 0; ik < k; ik++)
    {
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

#ifdef DEBUG
    t2 = clock();
    std::cout << "time for  filter transform " << static_cast<double>(t2 - t1) / CLOCKS_PER_SEC << " \n";
#endif
}

void inputTransform2X2_3X3
(
    int m, int r, int n, int c,
    int h, int w,
    int vpadding, int hpadding,
    float* D, int ldD1, int ldD2, int ldD3,
    float* V
)
{
    std::cout<<"lD1 ld2 ld3 "<<ldD1<<" "<<ldD2<<" "<<ldD3<<std::endl;
    assert(m == 2);
    assert(r == 3);

    const int t = m + r - 1;
    const int s = m;
    const int vstride = 1, hstride = 1;

    //we have performed macro tilling where the dimensions of each macro tile is 6 X 8 and each tile contain 4*4 input tiles
    // there are 2 vertical tiles in a macro tile (height of macro tile: 6)
    //there are 3 horizontal tiles in a macro tile (width of macro tile: 8) 

    int tile_h, tile_w, ic, in, ih, iw, hh, ww, hh_, ww_,
        ldV1, ldV2, ldV3,
        i, j, fh, fw, ow, oh, ho, wo,
        imtile_h, imtile_w, imt_h, imt_w, imt_hs, imt_vs, timt_h, timt_w;

    // registers for macro tiling

    __m256 UX[6], WX[8];

    float temp_ux[6 * 8];

    ho = (h + 2 * vpadding - r) / vstride + 1;  //height of output image 
    wo = (w + 2 * hpadding - r) / hstride + 1;  //width of output image

    tile_h = ceil(((double)h + 2 * vpadding - t) / s) + 1;  //no of tiles in height
    tile_w = ceil(((double)w + 2 * hpadding - t) / s) + 1; //no of tiles in width

    timt_h = 2;                     timt_w = 3;                     // Number of tiles per input macrotile: height and width
    imt_h = t + (timt_h - 1) * s;  imt_w = t + (timt_w - 1) * s;  // Input macrotile height(6) and width(8) 
    imt_vs = timt_h * s;            imt_hs = timt_w * s;            // Input Macrotile vstride(4) and hstride(6)
    imtile_h = ceil(((double)h + 2 * vpadding - imt_h) / imt_vs) + 1; //no of macro tile in height
    imtile_w = ceil(((double)w + 2 * hpadding - imt_w) / imt_hs) + 1; //no of macro tile in width


    ldV3 = (n * tile_h * tile_w);
    ldV2 = c * ldV3;
    ldV1 = t * ldV2;

//#pragma omp parallel for collapse(2) private(in,ic,ih,hh_,hh,fh,oh,iw,ww_,ww,fw,ow,UX,WX,i,j,temp_ux) if  ( n*c > 1)
    for (in = 0; in < n; in++) //no of images
        for (ic = 0; ic < c; ic++)  //no of channels
            for (ih = 0; ih < imtile_h; ih++) {  //no of macro tile
                hh_ = min(h, ih * imt_vs - vpadding); // we are making sure the current macro tile does not exceed the height
                hh = max(hh_, 0);   //starting height of current macro

                fh = min(max(-hh_, 0), imt_h);  //index start for current macro height
                oh = max(min(h - hh, imt_h), 0);
                oh = oh < imt_h ? oh + fh : oh; //index end for current macro width
                std::cout<<"\n----------ih hh_ hh fh oh "<<ih<<" "<<hh_<<" "<<hh<<" "<<fh<<" "<<oh<<"-----------------\n";
                for (iw = 0; iw < imtile_w; iw++) {  //no of macro tile in width
                    ww_ = min(w, iw * imt_hs - hpadding);  
                    ww = max(ww_, 0);  //starting width of current macro width
                    fw = min(max(-ww_, 0), imt_w); //index start for current macro width
                    ow = max(min(w - ww, imt_w), 0); 
                    ow = ow < imt_w ? ow + fw : ow; //index end for current macro height
                    std::cout<<"\niw ww_ ww fw ow "<<iw<<" "<<ww_<<" "<<ww<<" "<<fw<<" "<<ow<<"\n";
                    memset(temp_ux, 0, sizeof(float) * 6 * 8);
                    //loading macro tile into temp_ux
                    for (i = fh; i < oh; i++) {
                        for (j = fw; j < ow; j++) {
                            temp_ux[i * 8 + j] = Drow(in, ic, hh + i - fh, ww + j - fw);
                        }
                    }
                    for(int i=0;i<6;i++){
                        for(int j=0;j<8;j++){
                            std::cout<<temp_ux[i*8+j]<<" ";
                        }
                        std::cout<<std::endl;
                    }
                    for (i = 0; i < imt_h; i++) UX[i] = _mm256_loadu_ps(temp_ux + i * 8);


                    for (i = 0; i < timt_h; i++) {
                        WX[i * 4 + 0] = UX[i * 2 + 0] - UX[i * 2 + 2];
                        WX[i * 4 + 1] = UX[i * 2 + 1] + UX[i * 2 + 2];
                        WX[i * 4 + 2] = UX[i * 2 + 2] - UX[i * 2 + 1];
                        WX[i * 4 + 3] = UX[i * 2 + 1] - UX[i * 2 + 3];

                    }
                    _MM_TRANSPOSE8_PS(WX[0], WX[1], WX[2], WX[3], WX[4], WX[5], WX[6], WX[7]);

                    int max_mth = min(tile_h - (ih * timt_h), timt_h), mth;
                    int max_mtw = min(tile_w - (iw * timt_w), timt_w), mtw;

                    for (mtw = 0; mtw < max_mtw; mtw++) {
                        UX[0] = WX[mtw * 2 + 0] - WX[mtw * 2 + 2];
                        UX[1] = WX[mtw * 2 + 1] + WX[mtw * 2 + 2];
                        UX[2] = WX[mtw * 2 + 2] - WX[mtw * 2 + 1];
                        UX[3] = WX[mtw * 2 + 1] - WX[mtw * 2 + 3];

                        for (i = 0; i < 6; i++) _mm256_storeu_ps(temp_ux+i*8, UX[i]);

                        int ix = in * tile_h * tile_w + (iw * timt_w + mtw);
                        for (mth = 0; mth < max_mth; mth++)
                            for (i = 0; i < t; i++)
                                for (j = 0; j < t; j++)
                                    Vrow(i, j, ic, ix + (ih * timt_h + mth) * tile_w) = temp_ux[j * 8 + mth * 4 + i];
                    }

                }
            }
            
            for(int i=0;i<4;i++){
                for(int j=0;j<4;j++){
                    std::cout<<temp_ux[i*4+j]<<" ";
                }
                std::cout<<std::endl;    
            }
    
}

void inputTransform2X2_3X3_sse(
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
    wo=(h+2*hpadding-r)/hstride+1;

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



    //#pragma omp parallel for collapse(2) private(in,ic,ih,hh_,hh,fh,oh,iw,ww_,ww,fw,ow,d0,d1,d2,d3,W0,W1,W2,W3,U0,U1,U2,U3,i,j,temp1,temp2,temp3,temp0) if((n*c)>1)
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
                        //for(int oo=0;oo<4;oo++){std::cout<<temp0[oo]<<" ";}
                        //std::cout<<std::endl;
                        //for(int oo=0;oo<4;oo++){std::cout<<temp1[oo]<<" ";}
                        //std::cout<<std::endl;
                        //for(int oo=0;oo<4;oo++){std::cout<<temp2[oo]<<" ";}
                        //std::cout<<std::endl;
                        //for(int oo=0;oo<4;oo++){std::cout<<temp3[oo]<<" ";}
                        //std::cout<<std::endl;
                        //std::cout << "--------------\n";

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
 /*       for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                std::cout << V[i * 4 + j] << " ";
            }
            std::cout << std::endl;
        }*/
//-----------------------code is working till here-------------------------------------------------------
        //#pragma omp parallel for collapse(2) private(e,v)
        for (e = 0; e < t; e++) {
            for (v = 0; v < t; v++) {
                // M[e,v] = U[e,v] @ V[e,v];

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    k, (n * tile_h * tile_w), c,
                    1.0, &Urow(e, v, 0, 0), c,
                    &Vrow(e, v, 0, 0), (n * tile_h * tile_w),
                    0.0, &Mrow(e, v, 0, 0), (n * tile_h * tile_w)
                );
            }
        }
        //#pragma omp parallel for collapse(2) private(in,ik,ih,iw,_M0,_M1,_M2,_M3,_W0,_W1,hh,ww,i,j,temp_M,temp_W,Z) if((n*k)>1)
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

}
void hadamard_product2X2_3X3
(
    int m, int r, int k, int c, int n, int h, int w, int vpadding, int hpadding,
    float* U,
    float* V,
    float* M



)
{

    assert(r == 3);
    assert(m == 2);

    int t = r + n - 1;
    int s = r;
    int tile_h, tile_w, e, v;
    tile_h = ceil(((double)h + 2 * vpadding - t) / s) + 1;
    tile_w = ceil(((double)w + 2 * hpadding - t) / s) + 1;

    int     ldU1, ldU2, ldU3,
        ldV1, ldV2, ldV3,
        ldM1, ldM2, ldM3;
    ldU3 = c;
    ldU2 = k * ldU3;
    ldU1 = t * ldU2;

    ldV3 = (n * tile_h * tile_w);
    ldV2 = c * ldV3;
    ldV1 = t * ldV2;

    ldM3 = (n * tile_h * tile_w);
    ldM2 = k * ldM3;    
    ldM1 = t * ldM2;



//#pragma omp parallel for collapse(2) private(e,v)
    for (e = 0; e < t; e++) {
        for (v = 0; v < t; v++) {
            // M[e,v] = U[e,v] @ V[e,v];

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                k, (n * tile_h * tile_w), c,
                1.0, &Urow(e, v, 0, 0), c,
                &Vrow(e, v, 0, 0), (n * tile_h * tile_w),
                0.0, &Mrow(e, v, 0, 0), (n * tile_h * tile_w)
            );
        }
    }





}
void inverseTransform2X2_3X3
(
    int m, int r, int k, int c, int n, int h, int w, int vpadding, int hpadding,
    float* M,float*Y,int ldY1,int ldY2,int ldY3
)
{
    assert(r == 3);
    assert(m == 2);

    int t = r + n - 1;
    int s = r;
    int tile_h, tile_w,
        ldM3, ldM2, ldM1;
    tile_h = ceil(((double)h + 2 * vpadding - t) / s) + 1;
    tile_w = ceil(((double)w + 2 * hpadding - t) / s) + 1;


    ldM3 = (n * tile_h * tile_w);
    ldM2 = k * ldM3;
    ldM1 = t * ldM2;

    int in, ik, ih, iw, hh, ww, i, j,ho,wo;

    __m128 _M0, _M1, _M2, _M3, _W0, _W1, _Z;
    float temp_M[4 * 4];
    float Z[4];
    float temp_W[2 * 4];
    const int vstride = 1, hstride = 1;
    ho = (h + 2 * vpadding - r) / vstride + 1;  //height of output image 
    wo = (w + 2 * hpadding - r) / hstride + 1;  //width of output image

//#pragma omp parallel for collapse(2) private(in,ik,ih,iw,_M0,_M1,_M2,_M3,_W0,_W1,hh,ww,i,j,temp_M,temp_W,Z) if((n*k)>1)
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

                    _W0 = _M0 + _M1 + _M2;
                    _W1 = _M1 - _M2 - _M3;

                    _mm_store_ps(temp_W, _W0);
                    _mm_store_ps(temp_W + 4, _W1);

                    Z[0] = temp_W[0] + temp_W[1] + temp_W[2];
                    Z[1] = temp_W[1] - temp_W[2] - temp_W[3];
                    Z[2] = temp_W[4 + 0] + temp_W[4 + 1] + temp_W[4 + 2];
                    Z[3] = temp_W[4 + 1] - temp_W[4 + 2] - temp_W[4 + 3];

                    for (i = 0; i < min(m, ho - hh); i++)
                        for (j = 0; j < min(m, wo - ww); j++)
                            Yrow(in, ik, hh + i, ww + j) = Z[j * m + i];

                }
}

void winograd2X2_3X3(int m, int r, int n, int k, int c, int h, int w, int vpadding, int hpadding, float* D, int ldD1, int ldD2, int ldD3, float* Y, int ldY1, int ldY2, int ldY3, float* U, float* V, float* M, float* F, int ldF1, int ldF2, int ldF3) {
    std::cout << "\n\n|----------------------------------------------------------------------------|\n";
    std::cout << "|------------------------Starting filter transform---------------------------|\n";
    std::cout <<   "|----------------------------------------------------------------------------|\n";
    filterTransform2X2_3X3(m, r, c, k, F, ldF1, ldF2, ldF3, U);
    std::cout << "\n\n|----------------------------------------------------------------------------|\n";
    std::cout << "|------------------------Finished filter transform---------------------------|\n";
    std::cout <<   "|----------------------------------------------------------------------------|\n";
    std::cout << "\n\n|----------------------------------------------------------------------------|\n";
    std::cout << "|------------------------Starting Input transform---------------------------|\n";
    std::cout <<   "|----------------------------------------------------------------------------|\n";
    inputTransform2X2_3X3_sse(m, r, n, k, c, h, w, vpadding, hpadding, D, ldD1, ldD2, ldD3, Y, ldY1, ldY2, ldY3, U, V, M);
    std::cout << "\n\n|----------------------------------------------------------------------------|\n";
    std::cout << "|------------------------Finished input transform---------------------------|\n";
    std::cout <<   "|----------------------------------------------------------------------------|\n";



}
void winogradCall(int m, int r, int n, int k, int c, int h, int w, float* D, int vpadding, int hpadding, float* Y, float* F) {
    std::cout << "sanket\n ";
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

    ldY3 = k;
    ldY2 = (a2)*ldY3;
    ldY1 = ldY2 * a1;
    winograd2X2_3X3(m, r, n, k, c, h, w, vpadding, hpadding, D, ldD1, ldD2, ldD3, Y, ldY1, ldY2, ldY3, U, V, M, F, ldF1, ldF2, ldF3);

}



