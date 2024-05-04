
#include <stdio.h>
#include <stdlib.h>


#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]

void gemm(char orderA, char orderB, char orderC,
    char transA, char transB,
    int m, int n, int k,
    float alpha, const float* A, int ldA,
    const float* B, int ldB,
    float beta, float* C, int ldC) {
    int    ic, jc, pc, i, j, p;
    float  zero = (float)0.0, one = (float)1.0, tmp;

    // Quick return if possible
    if ((m == 0) || (n == 0) || (k == 0) || ((alpha == zero) && (beta == one)))
        return;

    if (alpha == zero) {
        if (beta == zero)
            if (orderC == 'C') {
#pragma omp parallel for private(ic)
                for (jc = 0; jc < n; jc++)
                    for (ic = 0; ic < m; ic++)
                        Ccol(ic, jc) = 0.0;
            }
            else {
#pragma omp parallel for private(ic)
                for (jc = 0; jc < n; jc++)
                    for (ic = 0; ic < m; ic++)
                        Crow(ic, jc) = 0.0;
            }
        else
            if (orderC == 'C') {
#pragma omp parallel for private(ic)
                for (jc = 0; jc < n; jc++)
                    for (ic = 0; ic < m; ic++)
                        Ccol(ic, jc) = beta * Ccol(ic, jc);
            }
            else {
#pragma omp parallel for private(ic)
                for (jc = 0; jc < n; jc++)
                    for (ic = 0; ic < m; ic++)
                        Crow(ic, jc) = beta * Crow(ic, jc);
            }
        return;
    }

    if ((transA == 'N') && (transB == 'N')) {
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
                tmp = zero;
                if ((orderA == 'C') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(i, p) * Bcol(p, j);
                }
                else if ((orderA == 'C') && (orderB == 'R')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(i, p) * Brow(p, j);
                }
                else if ((orderA == 'R') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Arow(i, p) * Bcol(p, j);
                }
                else {
                    for (p = 0; p < k; p++)
                        tmp += Arow(i, p) * Brow(p, j);
                }

                if (beta == zero) {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp;
                    else
                        Crow(i, j) = alpha * tmp;
                }
                else {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp + beta * Ccol(i, j);
                    else
                        Crow(i, j) = alpha * tmp + beta * Crow(i, j);
                }
            }
    }
    else if ((transA == 'N') && (transB == 'T')) {
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
                tmp = zero;
                if ((orderA == 'C') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(i, p) * Bcol(j, p);
                }
                else if ((orderA == 'C') && (orderB == 'R')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(i, p) * Brow(j, p);
                }
                else if ((orderA == 'R') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Arow(i, p) * Bcol(j, p);
                }
                else {
                    for (p = 0; p < k; p++)
                        tmp += Arow(i, p) * Brow(j, p);
                }

                if (beta == zero) {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp;
                    else
                        Crow(i, j) = alpha * tmp;
                }
                else {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp + beta * Ccol(i, j);
                    else
                        Crow(i, j) = alpha * tmp + beta * Crow(i, j);
                }
            }
    }
    else if ((transA == 'T') && (transB == 'N')) {
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
                tmp = zero;
                if ((orderA == 'C') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(p, i) * Bcol(p, j);
                }
                else if ((orderA == 'C') && (orderB == 'R')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(p, i) * Brow(p, j);
                }
                else if ((orderA == 'R') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Arow(p, i) * Bcol(p, j);
                }
                else {
                    for (p = 0; p < k; p++)
                        tmp += Arow(p, i) * Brow(p, j);
                }

                if (beta == zero) {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp;
                    else
                        Crow(i, j) = alpha * tmp;
                }
                else {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp + beta * Ccol(i, j);
                    else
                        Crow(i, j) = alpha * tmp + beta * Crow(i, j);
                }
            }
    }
    else if ((transA == 'T') && (transB == 'T')) {
        for (j = 0; j < n; j++)
            for (i = 0; i < m; i++) {
                tmp = zero;
                if ((orderA == 'C') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(p, i) * Bcol(j, p);
                }
                else if ((orderA == 'C') && (orderB == 'R')) {
                    for (p = 0; p < k; p++)
                        tmp += Acol(p, i) * Brow(j, p);
                }
                else if ((orderA == 'R') && (orderB == 'C')) {
                    for (p = 0; p < k; p++)
                        tmp += Arow(p, i) * Bcol(j, p);
                }
                else {
                    for (p = 0; p < k; p++)
                        tmp += Arow(p, i) * Brow(j, p);
                }

                if (beta == zero) {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp;
                    else
                        Crow(i, j) = alpha * tmp;
                }
                else {
                    if (orderC == 'C')
                        Ccol(i, j) = alpha * tmp + beta * Ccol(i, j);
                    else
                        Crow(i, j) = alpha * tmp + beta * Crow(i, j);
                }
            }
    }
    else {
        printf("Error: Invalid options for transA, transB: %c %c\n", transA, transB);
        exit(-1);
    }
}