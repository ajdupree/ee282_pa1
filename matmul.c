// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

#include "x86intrin.h"

//#define B1SIZE 256
#define min(x,y) ((x < y) ? x : y)

void matmul_aux(int N, const double* __restrict__ A, const double* __restrict__ B, 
	double* __restrict__ C, int B1SIZE); 
void matmul_aux_prefetched(int N, const double* __restrict__ A, const double* __restrict__ B, 
	double* __restrict__ C, int B1SIZE); 
void matmul_aux_nonblocked(int N, const double* __restrict__ A, const double* __restrict__ B, 
	double* __restrict__ C);

void matmul(int N, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C) {
  
  int i, j, k, bi, bj, bk,  I, J, K, BI, BJ, BK, NN, NB1SIZE;
	__m128d _a00, _a11, _a22, _a33, _a44, _a55, _a66, _a77, _b01, _b23, _b45, _b67, _c01, _c23, _sum0, _sum1;
	__m128d _A0, _B0, _C0, _A1, _B1, _C1, _A2, _A3, _A4,_A5,_A6,_A7,_C2,_C3;


  switch(N)
  {
		case 2:
			// VECTORSSSSS!!!!!!11!1111!!11
			_a00 = _mm_set1_pd(A[0]);
			_a11 = _mm_set1_pd(A[1]);
			_b01 = _mm_load_pd(&B[0]);
			_b23 = _mm_load_pd(&B[2]);
			_a22 = _mm_set1_pd(A[2]);
			_a33 = _mm_set1_pd(A[3]);
			_c01 = _mm_load_pd(&C[0]);
			_c23 = _mm_load_pd(&C[2]);
			
			_mm_store_pd(C, _mm_add_pd(_mm_add_pd(_mm_mul_pd(_a00,_b01),_mm_mul_pd(_a11,_b23)), _c01));
			_mm_store_pd(C+2, _mm_add_pd(_mm_add_pd(_mm_mul_pd(_a22,_b01),_mm_mul_pd(_a33,_b23)), _c23));

			break;


		case 4:
    	
			// code is 100% self explanatory. no comments needed.

			_a00 = _mm_set1_pd(A[0]);
			_a11 = _mm_set1_pd(A[1]);
			_a22 = _mm_set1_pd(A[2]);
			_a33 = _mm_set1_pd(A[3]);
			_a44 = _mm_set1_pd(A[4]);
			_a55 = _mm_set1_pd(A[5]);
			_a66 = _mm_set1_pd(A[6]);
			_a77 = _mm_set1_pd(A[7]);
			_b01 = _mm_load_pd(&B[0]);
			_b23 = _mm_load_pd(&B[4]);
			_b45 = _mm_load_pd(&B[8]);
			_b67 = _mm_load_pd(&B[12]);
			_c01 = _mm_load_pd(&C[0]);
			_c23 = _mm_load_pd(&C[4]);
			
			_sum0 = _mm_add_pd(_mm_mul_pd(_a00,_b01),_mm_mul_pd(_a11,_b23));
			_sum1 = _mm_add_pd(_mm_mul_pd(_a22,_b45),_mm_mul_pd(_a33,_b67));
			_sum0 = _mm_add_pd(_mm_add_pd(_sum0,_sum1),_c01);
			_mm_store_pd(C, _sum0);
			
			_sum0 = _mm_add_pd(_mm_mul_pd(_a44,_b01),_mm_mul_pd(_a55,_b23));
			_sum1 = _mm_add_pd(_mm_mul_pd(_a66,_b45),_mm_mul_pd(_a77,_b67));
			_sum0 = _mm_add_pd(_mm_add_pd(_sum0,_sum1),_c23);
			_mm_store_pd(C+4, _sum0);
			
			_b01 = _mm_load_pd(&B[2]);
			_b23 = _mm_load_pd(&B[6]);
			_b45 = _mm_load_pd(&B[10]);
			_b67 = _mm_load_pd(&B[14]);
			_c01 = _mm_load_pd(&C[2]);
			_c23 = _mm_load_pd(&C[6]);
			
			_sum0 = _mm_add_pd(_mm_mul_pd(_a00,_b01),_mm_mul_pd(_a11,_b23));
			_sum1 = _mm_add_pd(_mm_mul_pd(_a22,_b45),_mm_mul_pd(_a33,_b67));
			_sum0 = _mm_add_pd(_mm_add_pd(_sum0,_sum1),_c01);
			_mm_store_pd(C+2, _sum0);
			
			_sum0 = _mm_add_pd(_mm_mul_pd(_a44,_b01),_mm_mul_pd(_a55,_b23));
			_sum1 = _mm_add_pd(_mm_mul_pd(_a66,_b45),_mm_mul_pd(_a77,_b67));
			_sum0 = _mm_add_pd(_mm_add_pd(_sum0,_sum1),_c23);
			_mm_store_pd(C+6, _sum0);

			_a00 = _mm_set1_pd(A[8]);
			_a11 = _mm_set1_pd(A[9]);
			_a22 = _mm_set1_pd(A[10]);
			_a33 = _mm_set1_pd(A[11]);
			_a44 = _mm_set1_pd(A[12]);
			_a55 = _mm_set1_pd(A[13]);
			_a66 = _mm_set1_pd(A[14]);
			_a77 = _mm_set1_pd(A[15]);
			_b01 = _mm_load_pd(&B[0]);
			_b23 = _mm_load_pd(&B[4]);
			_b45 = _mm_load_pd(&B[8]);
			_b67 = _mm_load_pd(&B[12]);
			_c01 = _mm_load_pd(&C[8]);
			_c23 = _mm_load_pd(&C[12]);
			
			_sum0 = _mm_add_pd(_mm_mul_pd(_a00,_b01),_mm_mul_pd(_a11,_b23));
			_sum1 = _mm_add_pd(_mm_mul_pd(_a22,_b45),_mm_mul_pd(_a33,_b67));
			_sum0 = _mm_add_pd(_mm_add_pd(_sum0,_sum1),_c01);
			_mm_store_pd(C+8, _sum0);
			
			_sum0 = _mm_add_pd(_mm_mul_pd(_a44,_b01),_mm_mul_pd(_a55,_b23));
			_sum1 = _mm_add_pd(_mm_mul_pd(_a66,_b45),_mm_mul_pd(_a77,_b67));
			_sum0 = _mm_add_pd(_mm_add_pd(_sum0,_sum1),_c23);
			_mm_store_pd(C+12, _sum0);
			
			_b01 = _mm_load_pd(&B[2]);
			_b23 = _mm_load_pd(&B[6]);
			_b45 = _mm_load_pd(&B[10]);
			_b67 = _mm_load_pd(&B[14]);
			_c01 = _mm_load_pd(&C[10]);
			_c23 = _mm_load_pd(&C[14]);
			
			_sum0 = _mm_add_pd(_mm_mul_pd(_a00,_b01),_mm_mul_pd(_a11,_b23));
			_sum1 = _mm_add_pd(_mm_mul_pd(_a22,_b45),_mm_mul_pd(_a33,_b67));
			_sum0 = _mm_add_pd(_mm_add_pd(_sum0,_sum1),_c01);
			_mm_store_pd(C+10, _sum0);
			
			_sum0 = _mm_add_pd(_mm_mul_pd(_a44,_b01),_mm_mul_pd(_a55,_b23));
			_sum1 = _mm_add_pd(_mm_mul_pd(_a66,_b45),_mm_mul_pd(_a77,_b67));
			_sum0 = _mm_add_pd(_mm_add_pd(_sum0,_sum1),_c23);
			_mm_store_pd(C+14, _sum0);

			break;


		case 8:
			NN = N*N;
			

			for (I = 0; I < NN; I += N)
				for (k = 0, K = 0; K < NN; k++, K += N)
					for (j = 0; j < N; j++)
						C[I + j] += A[I+k]*B[K+j];
			break;

		case 16:
		case 32:
		case 64:
		case 128:
		case 256:
		case 512:
			matmul_aux_nonblocked(N,A,B,C);
			break;
		case 1024:
			matmul_aux_prefetched(N,A,B,C,256);
			break;
		case 2048:
			matmul_aux_prefetched(N,A,B,C,256);
			break;

		default:
			break;
  }
  
}

void matmul_aux_prefetched(int N, const double* __restrict__ A, const double* __restrict__ B, 
	double* __restrict__ C, int B1SIZE)
{
  int i, j, k, bi, bj, bk,  I, J, K, BI, BJ, BK, NN, NB1SIZE;
	__m128d _A0, _B0, _C0, _A1, _B1, _C1, _A2, _A3, _A4,_A5,_A6,_A7,_C2,_C3;
	
	NN = N*N;
	NB1SIZE = N*B1SIZE;

	// do while all up in this bitch. should be a tiny bit faster than a for loop.
	BI = 0;
	do {
		bk = 0; BK = 0;
		do {
			bj = 0;
			do {
				I = BI;
				do {
					 __builtin_prefetch(C + I + N + j); //hurts performance under around 128
					 //__builtin_prefetch(A + I + (N<<2) + bk + j);
					k = bk; K = BK;
					do {
						__builtin_prefetch(B + K + N + j); //hurts performance under around 128
						//__builtin_prefetch(A + I + (N<<2) + j); //hurts performance under around 128
						_A0 = _mm_set1_pd(A[I+k]);
						_A1 = _mm_set1_pd(A[I+k+1]);
						_A2 = _mm_set1_pd(A[I+N+k]);
						_A3 = _mm_set1_pd(A[I+N+k+1]);
						_A4 = _mm_set1_pd(A[I+(N<<1)+k]);
						_A5 = _mm_set1_pd(A[I+(N<<1)+k+1]);
						_A6 = _mm_set1_pd(A[I+3*N+k]);
						_A7 = _mm_set1_pd(A[I+3*N+k+1]);

						j = bj;
						do {
							// VECTORS: How do they work? (intrinsicly?)
							_B0 = _mm_load_pd(&B[K+j]);
							_B1 = _mm_load_pd(&B[K+N+j]);
							_C0 = _mm_load_pd(&C[I+j]);
							_C1 = _mm_load_pd(&C[I+N+j]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j]);
							_C3 = _mm_load_pd(&C[I+3*N+j]);
							_mm_store_pd(C+I+j, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+2]);
							_B1 = _mm_load_pd(&B[K+N+j+2]);
							_C0 = _mm_load_pd(&C[I+j+2]);
							_C1 = _mm_load_pd(&C[I+N+j+2]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+2]);
							_C3 = _mm_load_pd(&C[I+3*N+j+2]);
							_mm_store_pd(C+I+j+2, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+2, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+2,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+2,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+4]);
							_B1 = _mm_load_pd(&B[K+N+j+4]);
							_C0 = _mm_load_pd(&C[I+j+4]);
							_C1 = _mm_load_pd(&C[I+N+j+4]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+4]);
							_C3 = _mm_load_pd(&C[I+3*N+j+4]);
							_mm_store_pd(C+I+j+4, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+4, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+4,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+4,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+6]);
							_B1 = _mm_load_pd(&B[K+N+j+6]);
							_C0 = _mm_load_pd(&C[I+j+6]);
							_C1 = _mm_load_pd(&C[I+N+j+6]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+6]);
							_C3 = _mm_load_pd(&C[I+3*N+j+6]);
							_mm_store_pd(C+I+j+6, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+6, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+6,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+6,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+8]);
							_B1 = _mm_load_pd(&B[K+N+j+8]);
							_C0 = _mm_load_pd(&C[I+j+8]);
							_C1 = _mm_load_pd(&C[I+N+j+8]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+8]);
							_C3 = _mm_load_pd(&C[I+3*N+j+8]);
							_mm_store_pd(C+I+j+8, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+8, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+8,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+8,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+10]);
							_B1 = _mm_load_pd(&B[K+N+j+10]);
							_C0 = _mm_load_pd(&C[I+j+10]);
							_C1 = _mm_load_pd(&C[I+N+j+10]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+10]);
							_C3 = _mm_load_pd(&C[I+3*N+j+10]);
							_mm_store_pd(C+I+j+10, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+10, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+10,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+10,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+12]);
							_B1 = _mm_load_pd(&B[K+N+j+12]);
							_C0 = _mm_load_pd(&C[I+j+12]);
							_C1 = _mm_load_pd(&C[I+N+j+12]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+12]);
							_C3 = _mm_load_pd(&C[I+3*N+j+12]);
							_mm_store_pd(C+I+j+12, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+12, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+12,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+12,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+14]);
							_B1 = _mm_load_pd(&B[K+N+j+14]);
							_C0 = _mm_load_pd(&C[I+j+14]);
							_C1 = _mm_load_pd(&C[I+N+j+14]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+14]);
							_C3 = _mm_load_pd(&C[I+3*N+j+14]);
							_mm_store_pd(C+I+j+14, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+14, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+14,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+14,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							
							j += 16;
						} while(j < min(bj+B1SIZE,N));
						k+=2; K += N<<1;
					} while(K < min(BK+NB1SIZE,NN));
					I += N<<2;
				} while(I < min(BI+NB1SIZE,NN));
				bj += B1SIZE;
			} while(bj < N);
			bk += B1SIZE; BK += NB1SIZE;
		} while(BK < NN);
		BI += NB1SIZE;
	} while(BI < NN);

}

void matmul_aux(int N, const double* __restrict__ A, const double* __restrict__ B, 
	double* __restrict__ C, int B1SIZE)
{
  int i, j, k, bi, bj, bk,  I, J, K, BI, BJ, BK, NN, NB1SIZE;
	__m128d _A0, _B0, _C0, _A1, _B1, _C1, _A2, _A3, _A4,_A5,_A6,_A7,_C2,_C3;
	
	NN = N*N;
	NB1SIZE = N*B1SIZE;

	// do while all up in this bitch. should be a tiny bit faster than a for loop.
	BI = 0;
	do {
		bk = 0; BK = 0;
		do {
			bj = 0;
			do {
				I = BI;
				do {
					k = bk; K = BK;
					do {
						_A0 = _mm_set1_pd(A[I+k]);
						_A1 = _mm_set1_pd(A[I+k+1]);
						_A2 = _mm_set1_pd(A[I+N+k]);
						_A3 = _mm_set1_pd(A[I+N+k+1]);
						_A4 = _mm_set1_pd(A[I+(N<<1)+k]);
						_A5 = _mm_set1_pd(A[I+(N<<1)+k+1]);
						_A6 = _mm_set1_pd(A[I+3*N+k]);
						_A7 = _mm_set1_pd(A[I+3*N+k+1]);

						j = bj;
						do {
							// VECTORS: How do they work? (intrinsicly?)
							_B0 = _mm_load_pd(&B[K+j]);
							_B1 = _mm_load_pd(&B[K+N+j]);
							_C0 = _mm_load_pd(&C[I+j]);
							_C1 = _mm_load_pd(&C[I+N+j]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j]);
							_C3 = _mm_load_pd(&C[I+3*N+j]);
							_mm_store_pd(C+I+j, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+2]);
							_B1 = _mm_load_pd(&B[K+N+j+2]);
							_C0 = _mm_load_pd(&C[I+j+2]);
							_C1 = _mm_load_pd(&C[I+N+j+2]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+2]);
							_C3 = _mm_load_pd(&C[I+3*N+j+2]);
							_mm_store_pd(C+I+j+2, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+2, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+2,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+2,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+4]);
							_B1 = _mm_load_pd(&B[K+N+j+4]);
							_C0 = _mm_load_pd(&C[I+j+4]);
							_C1 = _mm_load_pd(&C[I+N+j+4]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+4]);
							_C3 = _mm_load_pd(&C[I+3*N+j+4]);
							_mm_store_pd(C+I+j+4, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+4, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+4,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+4,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+6]);
							_B1 = _mm_load_pd(&B[K+N+j+6]);
							_C0 = _mm_load_pd(&C[I+j+6]);
							_C1 = _mm_load_pd(&C[I+N+j+6]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+6]);
							_C3 = _mm_load_pd(&C[I+3*N+j+6]);
							_mm_store_pd(C+I+j+6, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+6, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+6,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+6,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+8]);
							_B1 = _mm_load_pd(&B[K+N+j+8]);
							_C0 = _mm_load_pd(&C[I+j+8]);
							_C1 = _mm_load_pd(&C[I+N+j+8]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+8]);
							_C3 = _mm_load_pd(&C[I+3*N+j+8]);
							_mm_store_pd(C+I+j+8, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+8, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+8,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+8,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+10]);
							_B1 = _mm_load_pd(&B[K+N+j+10]);
							_C0 = _mm_load_pd(&C[I+j+10]);
							_C1 = _mm_load_pd(&C[I+N+j+10]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+10]);
							_C3 = _mm_load_pd(&C[I+3*N+j+10]);
							_mm_store_pd(C+I+j+10, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+10, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+10,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+10,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+12]);
							_B1 = _mm_load_pd(&B[K+N+j+12]);
							_C0 = _mm_load_pd(&C[I+j+12]);
							_C1 = _mm_load_pd(&C[I+N+j+12]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+12]);
							_C3 = _mm_load_pd(&C[I+3*N+j+12]);
							_mm_store_pd(C+I+j+12, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+12, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+12,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+12,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							_B0 = _mm_load_pd(&B[K+j+14]);
							_B1 = _mm_load_pd(&B[K+N+j+14]);
							_C0 = _mm_load_pd(&C[I+j+14]);
							_C1 = _mm_load_pd(&C[I+N+j+14]);
							_C2 = _mm_load_pd(&C[I+(N<<1)+j+14]);
							_C3 = _mm_load_pd(&C[I+3*N+j+14]);
							_mm_store_pd(C+I+j+14, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
							_mm_store_pd(C+I+N+j+14, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
							_mm_store_pd(C+I+(N<<1)+j+14,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
							_mm_store_pd(C+I+3*N+j+14,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
							
							j += 16;
						} while(j < min(bj+B1SIZE,N));
						k+=2; K += N<<1;
					} while(K < min(BK+NB1SIZE,NN));
					I += N<<2;
				} while(I < min(BI+NB1SIZE,NN));
				bj += B1SIZE;
			} while(bj < N);
			bk += B1SIZE; BK += NB1SIZE;
		} while(BK < NN);
		BI += NB1SIZE;
	} while(BI < NN);

}

void matmul_aux_nonblocked(int N, const double* __restrict__ A, const double* __restrict__ B, 
	double* __restrict__ C)
{
  int i, j, k, bi, bj, bk,  I, J, K, BI, BJ, BK, NN;
	__m128d _A0, _B0, _C0, _A1, _B1, _C1, _A2, _A3, _A4,_A5,_A6,_A7,_C2,_C3;
	
	NN = N*N;

	// do while all up in this bitch. should be a tiny bit faster than a for loop.
	I = 0;
	do {
		k = 0; K = 0;
		do {
			_A0 = _mm_set1_pd(A[I+k]);
			_A1 = _mm_set1_pd(A[I+k+1]);
			_A2 = _mm_set1_pd(A[I+N+k]);
			_A3 = _mm_set1_pd(A[I+N+k+1]);
			_A4 = _mm_set1_pd(A[I+(N<<1)+k]);
			_A5 = _mm_set1_pd(A[I+(N<<1)+k+1]);
			_A6 = _mm_set1_pd(A[I+3*N+k]);
			_A7 = _mm_set1_pd(A[I+3*N+k+1]);

			j = 0;
			do {
				// VECTORS: How do they work? (intrinsicly?)
				_B0 = _mm_load_pd(&B[K+j]);
				_B1 = _mm_load_pd(&B[K+N+j]);
				_C0 = _mm_load_pd(&C[I+j]);
				_C1 = _mm_load_pd(&C[I+N+j]);
				_C2 = _mm_load_pd(&C[I+(N<<1)+j]);
				_C3 = _mm_load_pd(&C[I+3*N+j]);
				_mm_store_pd(C+I+j, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
				_mm_store_pd(C+I+N+j, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
				_mm_store_pd(C+I+(N<<1)+j,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
				_mm_store_pd(C+I+3*N+j,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
				_B0 = _mm_load_pd(&B[K+j+2]);
				_B1 = _mm_load_pd(&B[K+N+j+2]);
				_C0 = _mm_load_pd(&C[I+j+2]);
				_C1 = _mm_load_pd(&C[I+N+j+2]);
				_C2 = _mm_load_pd(&C[I+(N<<1)+j+2]);
				_C3 = _mm_load_pd(&C[I+3*N+j+2]);
				_mm_store_pd(C+I+j+2, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
				_mm_store_pd(C+I+N+j+2, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
				_mm_store_pd(C+I+(N<<1)+j+2,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
				_mm_store_pd(C+I+3*N+j+2,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
				_B0 = _mm_load_pd(&B[K+j+4]);
				_B1 = _mm_load_pd(&B[K+N+j+4]);
				_C0 = _mm_load_pd(&C[I+j+4]);
				_C1 = _mm_load_pd(&C[I+N+j+4]);
				_C2 = _mm_load_pd(&C[I+(N<<1)+j+4]);
				_C3 = _mm_load_pd(&C[I+3*N+j+4]);
				_mm_store_pd(C+I+j+4, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
				_mm_store_pd(C+I+N+j+4, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
				_mm_store_pd(C+I+(N<<1)+j+4,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
				_mm_store_pd(C+I+3*N+j+4,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
				_B0 = _mm_load_pd(&B[K+j+6]);
				_B1 = _mm_load_pd(&B[K+N+j+6]);
				_C0 = _mm_load_pd(&C[I+j+6]);
				_C1 = _mm_load_pd(&C[I+N+j+6]);
				_C2 = _mm_load_pd(&C[I+(N<<1)+j+6]);
				_C3 = _mm_load_pd(&C[I+3*N+j+6]);
				_mm_store_pd(C+I+j+6, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
				_mm_store_pd(C+I+N+j+6, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
				_mm_store_pd(C+I+(N<<1)+j+6,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
				_mm_store_pd(C+I+3*N+j+6,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
				_B0 = _mm_load_pd(&B[K+j+8]);
				_B1 = _mm_load_pd(&B[K+N+j+8]);
				_C0 = _mm_load_pd(&C[I+j+8]);
				_C1 = _mm_load_pd(&C[I+N+j+8]);
				_C2 = _mm_load_pd(&C[I+(N<<1)+j+8]);
				_C3 = _mm_load_pd(&C[I+3*N+j+8]);
				_mm_store_pd(C+I+j+8, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
				_mm_store_pd(C+I+N+j+8, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
				_mm_store_pd(C+I+(N<<1)+j+8,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
				_mm_store_pd(C+I+3*N+j+8,			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
				_B0 = _mm_load_pd(&B[K+j+10]);
				_B1 = _mm_load_pd(&B[K+N+j+10]);
				_C0 = _mm_load_pd(&C[I+j+10]);
				_C1 = _mm_load_pd(&C[I+N+j+10]);
				_C2 = _mm_load_pd(&C[I+(N<<1)+j+10]);
				_C3 = _mm_load_pd(&C[I+3*N+j+10]);
				_mm_store_pd(C+I+j+10, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
				_mm_store_pd(C+I+N+j+10, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
				_mm_store_pd(C+I+(N<<1)+j+10,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
				_mm_store_pd(C+I+3*N+j+10,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
				_B0 = _mm_load_pd(&B[K+j+12]);
				_B1 = _mm_load_pd(&B[K+N+j+12]);
				_C0 = _mm_load_pd(&C[I+j+12]);
				_C1 = _mm_load_pd(&C[I+N+j+12]);
				_C2 = _mm_load_pd(&C[I+(N<<1)+j+12]);
				_C3 = _mm_load_pd(&C[I+3*N+j+12]);
				_mm_store_pd(C+I+j+12, 				_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
				_mm_store_pd(C+I+N+j+12, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
				_mm_store_pd(C+I+(N<<1)+j+12,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
				_mm_store_pd(C+I+3*N+j+12,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
				_B0 = _mm_load_pd(&B[K+j+14]);
				_B1 = _mm_load_pd(&B[K+N+j+14]);
				_C0 = _mm_load_pd(&C[I+j+14]);
				_C1 = _mm_load_pd(&C[I+N+j+14]);
				_C2 = _mm_load_pd(&C[I+(N<<1)+j+14]);
				_C3 = _mm_load_pd(&C[I+3*N+j+14]);
				_mm_store_pd(C+I+j+14, 		 	 	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A0,_B0),_mm_mul_pd(_A1,_B1)),_C0));
				_mm_store_pd(C+I+N+j+14, 			_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A2,_B0),_mm_mul_pd(_A3,_B1)),_C1));
				_mm_store_pd(C+I+(N<<1)+j+14,	_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A4,_B0),_mm_mul_pd(_A5,_B1)),_C2));
				_mm_store_pd(C+I+3*N+j+14,		_mm_add_pd(_mm_add_pd(_mm_mul_pd(_A6,_B0),_mm_mul_pd(_A7,_B1)),_C3));
				
				j += 16;
			} while(j < N);
			k+=2; K += N<<1;
		} while(K < NN);
		I += N<<2;
	} while(I <NN);

}
