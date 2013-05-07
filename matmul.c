// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

#include "malloc.h"
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
  
  int i, j, k, bi, bj, bk,  I, J, K, BI, BJ, BK, NN, NB1SIZE;
  __m128d _A, _B, _C;


	NN = N*N;
  
	// TODO: add special cases for small matrices. They dont need blocking,
	// and the inner loop should be fully unrolled
  if(N < 16)
  {
		for (I = 0; I < NN; I += N)
			for (k = 0, K = 0; K < NN; k++, K += N)
				for (j = 0; j < N; j++)
		      C[I + j] += A[I+k]*B[K+j];
  }
  else
  {
  	// the for for for for for for for for for loop is not commonly
  	// seen in amateur play

		/*Ok seriously what the sweet fuck is actually going on here?*/
		NB1SIZE = N*B1SIZE;
		
		//first level blocking
		for (BI = 0; BI < NN; BI += NB1SIZE)
			for (bk = 0, BK = 0; BK < NN; bk += B1SIZE, BK += NB1SIZE)
				for (bj = 0; bj < N; bj += B1SIZE)
					// matrix multiply current block
					for (I = BI; I < min(BI+NB1SIZE,NN); I += N)
						// prefetch C[(I+1) + j] ?
						for (k = bk, K = BK; K < min(BK+NB1SIZE,NN); k++, K += N)
						{
							// prefetch B[(K+1) + j] ?
							_A = _mm_set_pd(A[I+k], A[I+k]);
							for (j = bj; j < min(bj+B1SIZE,N); j+=16)
							{
								_B = _mm_load_pd(&B[K+j]);
								_C = _mm_load_pd(&C[I+j]);
								_mm_store_pd(&C[I+j],_mm_add_pd(_mm_mul_pd(_A,_B),_C));
								_B = _mm_load_pd(&B[K+j+2]);
								_C = _mm_load_pd(&C[I+j+2]);
								_mm_store_pd(&C[I+j+2],_mm_add_pd(_mm_mul_pd(_A,_B),_C));
								_B = _mm_load_pd(&B[K+j+4]);
								_C = _mm_load_pd(&C[I+j+4]);
								_mm_store_pd(&C[I+j+4],_mm_add_pd(_mm_mul_pd(_A,_B),_C));
								_B = _mm_load_pd(&B[K+j+6]);
								_C = _mm_load_pd(&C[I+j+6]);
								_mm_store_pd(&C[I+j+6],_mm_add_pd(_mm_mul_pd(_A,_B),_C));
								_B = _mm_load_pd(&B[K+j+8]);
								_C = _mm_load_pd(&C[I+j+8]);
								_mm_store_pd(&C[I+j+8],_mm_add_pd(_mm_mul_pd(_A,_B),_C));
								_B = _mm_load_pd(&B[K+j+10]);
								_C = _mm_load_pd(&C[I+j+10]);
								_mm_store_pd(&C[I+j+10],_mm_add_pd(_mm_mul_pd(_A,_B),_C));
								_B = _mm_load_pd(&B[K+j+12]);
								_C = _mm_load_pd(&C[I+j+12]);
								_mm_store_pd(&C[I+j+12],_mm_add_pd(_mm_mul_pd(_A,_B),_C));
								_B = _mm_load_pd(&B[K+j+14]);
								_C = _mm_load_pd(&C[I+j+14]);
								_mm_store_pd(&C[I+j+14],_mm_add_pd(_mm_mul_pd(_A,_B),_C));
							}
						}
  }
  
}


















//	_A = _mm_load_pd(&A[i*N+k]);
//	_B = _mm_load_pd(&BT[k*N+j]);
//	_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
//	_A = _mm_load_pd(&A[i*N+k+2]);
//	_B = _mm_load_pd(&BT[k*N+j+2]);
//	_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
//	_A = _mm_load_pd(&A[i*N+k+4]);
//	_B = _mm_load_pd(&BT[k*N+j+4]);
//	_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
//	_A = _mm_load_pd(&A[i*N+k+6]);
//	_B = _mm_load_pd(&BT[k*N+j+6]);
//	_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
//	_A = _mm_load_pd(&A[i*N+k]+8);
//	_B = _mm_load_pd(&BT[k*N+j]+8);
//	_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
//	_A = _mm_load_pd(&A[i*N+k+10]);
//	_B = _mm_load_pd(&BT[k*N+j+10]);
//	_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
//	_A = _mm_load_pd(&A[i*N+k+12]);
//	_B = _mm_load_pd(&BT[k*N+j+12]);
//	_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
//	_A = _mm_load_pd(&A[i*N+k+14]);
//	_B = _mm_load_pd(&BT[k*N+j+14]);
//	_sum = _mm_add_pd(_mm_mul_pd(_A,_B),_sum);
//}
//_mm_store_pd(sum,_sum);
//C[i*N+j] += sum[0]+sum[1];

		//second level blocking
	//for(bbi = 0; bbi < N; bbi += B2SIZE)
	//	for(bbj = 0; bbj < N; bbj += B2SIZE)
	//		for(bbk = 0; bbk < N; bbk += B2SIZE)
	//			//first level blocking
	//			for (bj = bbj; bj < min(bbj+B2SIZE,N); bj += B1SIZE)
	//				for (bk = bbk; bk < min(bbk+B2SIZE,N); bk += B1SIZE)
	//					for (bi = bbi; bi < min(bbi+B2SIZE,N); bi += B1SIZE)
	//						//actual matrix multiplication
	//						for (j = bj; j < min(bj+B1SIZE,N); j++)
	//							for (k = bk; k < min(bk+B1SIZE,N); k++)
	//								for (i = bi; i < min(bi+B1SIZE,N); i++)
	//									C[i*N + j] += A[i*N+k]*B[k*N+j];
