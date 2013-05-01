// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

#include "malloc.h"
//#include "emmintrin.h"
#include "x86intrin.h"

#define B1SIZE 128
#define B2SIZE 256
#define min(x,y) ((x < y) ? x : y)

void matmul(int N, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C) {
//int i, j, k;
//
//for (i = 0; i < N; i++)
//  for (j = 0; j < N; j++)
//    for (k = 0; k < N; k++)
//      C[i*N + j] += A[i*N+k]*B[k*N+j];
//
  
  int i, j, k, bi, bj, bk, bbi, bbj, bbk;
  double * __restrict__ BT;
  __m128d _A, _B, _sum;
  double sum[2];
  
  BT = (double*) memalign(16,sizeof(double)*N*N);
  
  for(bi = 0; bi < N; bi += B1SIZE)
    for(bj = 0; bj < N; bj += B1SIZE)
      for(i = bi; i < min(bi+B1SIZE,N); i++)
        for(j = bj; j < min(bj+B1SIZE,N); j++)
          BT[j*N+i] = B[i*N+j];
  
	// TODO: add special cases for small matrices. They dont need blocking,
	// and the inner loop should be fully unrolled
  if(N < 16)
  {
    for (bi = 0; bi < N; bi += B1SIZE)
      for (bj = 0; bj < N; bj += B1SIZE)
        for (bk = 0; bk < N; bk += B1SIZE)
          for (i = bi; i < min(bi+B1SIZE,N); i++)
            for (j = bj; j < min(bj+B1SIZE,N); j++)
            {
              _sum = _mm_set_pd1(0.0);
              for (k = bk; k < min(bk+B1SIZE,N); k+=2)
              {
                _A = _mm_load_pd(&A[i*N+k]);
                _B = _mm_load_pd(&BT[j*N+k]);
                _sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
              }
              _mm_store_pd(sum,_sum);
              C[i*N+j] += sum[0]+sum[1];
            }
  }
  else
  {
  	// the for for for for for for for for for loop is not commonly
  	// seen in amateur play
  	for(bbi = 0; bbi < N; bbi += B2SIZE)
  		for(bbj = 0; bbj < N; bbj += B2SIZE)
  			for(bbk = 0; bbk < N; bbk += B2SIZE)
  				for (bi = bbi; bi < min(bbi+B2SIZE,N); bi += B1SIZE)
  					for (bj = bbj; bj < min(bbj+B2SIZE,N); bj += B1SIZE)
  						for (bk = bbk; bk < min(bbk+B2SIZE,N); bk += B1SIZE)
  							for (i = bi; i < min(bi+B1SIZE,N); i++)
  								for (j = bj; j < min(bj+B1SIZE,N); j++)
  								{
  									_sum = _mm_set_pd1(0.0);
  									for (k = bk; k < min(bk+B1SIZE,N); k+=16)
  									{
  										_A = _mm_load_pd(&A[i*N+k]);
  										_B = _mm_load_pd(&BT[j*N+k]);
  										_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
  										_A = _mm_load_pd(&A[i*N+k+2]);
  										_B = _mm_load_pd(&BT[j*N+k+2]);
  										_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
  										_A = _mm_load_pd(&A[i*N+k+4]);
  										_B = _mm_load_pd(&BT[j*N+k+4]);
  										_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
  										_A = _mm_load_pd(&A[i*N+k+6]);
  										_B = _mm_load_pd(&BT[j*N+k+6]);
  										_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
  										_A = _mm_load_pd(&A[i*N+k]+8);
  										_B = _mm_load_pd(&BT[j*N+k]+8);
  										_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
  										_A = _mm_load_pd(&A[i*N+k+10]);
  										_B = _mm_load_pd(&BT[j*N+k+10]);
  										_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
  										_A = _mm_load_pd(&A[i*N+k+12]);
  										_B = _mm_load_pd(&BT[j*N+k+12]);
  										_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
  										_A = _mm_load_pd(&A[i*N+k+14]);
  										_B = _mm_load_pd(&BT[j*N+k+14]);
  										_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
  									}
										_mm_store_pd(sum,_sum);
										C[i*N+j] += sum[0]+sum[1];
  								}
  
  }
  
  free(BT);
  
//for (bi = 0; bi < N; bi += B1SIZE)
//  for (bj = 0; bj < N; bj += B1SIZE)
//    for (bk = 0; bk < N; bk += B1SIZE)
//      for (i = bi; i < min(bi+B1SIZE,N); i++)
//        for (j = bj; j < min(bj+B1SIZE,N); j++)
//          for (k = bk; k < min(bk+B1SIZE,N); k++)
//            C[i*N + j] += A[i*N+k]*BT[j*N+k]; 
}
