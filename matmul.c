// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

#include "x86intrin.h"

#define B1SIZE 256
#define min(x,y) ((x < y) ? x : y)


void matmul(int N, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C) {
  
  int i, j, k, bi, bj, bk,  I, J, K, BI, BJ, BK, NN, NB1SIZE;
  register __m128d _A0, _B0, _C0, _A1, _B1, _C1, _A2, _A3, _A4,_A5,_A6,_A7,_C2,_C3;

	NN = N*N;

  switch(N)
  {
		case 2:
		case 4:
		case 8:

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
		case 1024:
		case 2048:

			// yeah yeah, i know the compiler optimizer should handle available expressions...
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
								_A0 = _mm_set_pd(A[I+k], 					A[I+k]);
								_A1 = _mm_set_pd(A[I+k+1], 				A[I+k+1]);
								_A2 = _mm_set_pd(A[I+N+k], 				A[I+N+k]);
								_A3 = _mm_set_pd(A[I+N+k+1], 			A[I+N+k+1]);
								_A4 = _mm_set_pd(A[I+(N<<1)+k], 	A[I+(N<<1)+k]);
								_A5 = _mm_set_pd(A[I+(N<<1)+k+1], A[I+(N<<1)+k+1]);
								_A6 = _mm_set_pd(A[I+3*N+k], 			A[I+3*N+k]);
								_A7 = _mm_set_pd(A[I+3*N+k+1], 		A[I+3*N+k+1]);

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
			break;

		default:
			break;
  }
  
}
