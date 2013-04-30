// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

#include "malloc.h"

#define BSIZE 8
#define min(x,y) ((x < y) ? x : y)

void matmul(int N, const double* A, const double* B, double* C) {
//int i, j, k;
//
//for (i = 0; i < N; i++)
//  for (j = 0; j < N; j++)
//    for (k = 0; k < N; k++)
//      C[i*N + j] += A[i*N+k]*B[k*N+j];
  
  
  int i, j, k, bi, bj, bk;
  double * BT;

  BT = (double*) malloc(sizeof(double)*N*N);

  for(bi = 0; bi < N; bi += BSIZE)
    for(bj = 0; bj < N; bj += BSIZE)
      for(i = bi; i < min(bi+BSIZE,N); i++)
        for(j = bj; j < min(bj+BSIZE,N); j++)
          BT[j*N+i] = B[i*N+j];

  for (bi = 0; bi < N; bi += BSIZE)
    for (bj = 0; bj < N; bj += BSIZE)
      for (bk = 0; bk < N; bk += BSIZE)
        for (i = bi; i < min(bi+BSIZE,N); i++)
          for (j = bj; j < min(bj+BSIZE,N); j++)
            for (k = bk; k < min(bk+BSIZE,N); k++)
              //C[i*N + j] += A[i*N+k]*B[k*N+j];
              C[i*N + j] += A[i*N+k]*BT[j*N+k];
}
