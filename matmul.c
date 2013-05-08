// This is the matrix multiply kernel you are to replace.
// Note that you are to implement C = C + AB, not C = AB!

#include "x86intrin.h"

#define B1SIZE 256
#define min(x,y) ((x < y) ? x : y)


void matmul(int N, const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C) {
  
  int i, j, k, bi, bj, bk,  I, J, K, BI, BJ, BK, NN, NB1SIZE;
	__m128d _a00, _a11, _a22, _a33, _b01, _b23, _c01, _c23;
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
		case 1024:
		case 2048:
			NN = N*N;
  		

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
			break;

		default:
			break;
  }
  
}
