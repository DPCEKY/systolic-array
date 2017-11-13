#include <iostream>
#include <iomanip>
using namespace std;

const int N = 20;
const int M = 5;

void printArr(int A[N][N]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      cout << setw(3) << A[i][j] << " ";
    cout << endl;
  }
  cout << endl;
}
/*
void top2( int A[N][N], int B[N][N], int C[N][N] ){

#pragma HLS array_partition variable=A dim=0
#pragma HLS array_partition variable=B dim=0
#pragma HLS array_partition variable=C dim=0

    for( int i = 0; i < N; i++ ){
        for( int j = 0; j < N; j++ ){
            for( int k = 0; k < N; k++ ){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
*/

void top(int A[N][N], int B[N][N], int C[N][N]) {

  int inA[M][M];
  int inB[M][M];

#pragma HLS array_partition variable=inA dim=0
#pragma HLS array_partition variable=inB dim=0
#pragma HLS array_partition variable=A dim=1
#pragma HLS array_partition variable=B dim=2
#pragma HLS array_partition variable=C dim=0

  // initialization
  for (int i = 0; i < M; i++) {
#pragma HLS pipeline
    for (int j = 0; j < M; j++) {
      inA[i][j] = 0;
      inB[i][j] = 0;
    }
  }

    for( int ii = 0; ii < N/M; ii++ ){
        for( int jj = 0; jj < N/M; jj++ ){

            for (int r = 0; r < N + 2 * M - 2; r++) {
                #pragma HLS pipeline
            // update data (i.e., reads data from previous PE)
            for (int i = 0; i < M; i++)
              for (int j = M - 1; j >= 1; j--)
                inA[i][j] = inA[i][j-1];

            for (int i = M - 1; i >= 1; i--)
                for (int j = 0; j < M; j++)
                    inB[i][j] = inB[i-1][j];

            // read new data from inputs
            // not ok here!
            for (int i = 0; i < M; i++) {
                if (r >= i && r < i+N)
                    inA[i][0] = A[i + ii * M][r-i];
                else
                    inA[i][0] = 0;
            }

            for (int j = 0; j < M; j++) {
                if (r >= j && r < j+N)
                    inB[0][j] = B[r-j][j + jj * M];
                else
                    inB[0][j] = 0;
            }


            // PE
            for (int i = 0; i < M; i++)
                for (int j = 0; j < M; j++)
                    C[i + ii * M][j + jj * M] += inA[i][j] * inB[i][j];
            }

        }
    }

}



int main(void) {

  int A[N][N];
  int B[N][N];
  int C[N][N];
  int O[N][N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = i + j;
      B[i][j] = i - j;
      C[i][j] = 0;
      O[i][j] = 0;
    }
  }

  top(A, B, C);

  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      for (int r = 0; r < N; r++)
        O[i][j] += A[i][r] * B[r][j];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (O[i][j] != C[i][j]) {
        cout << "Wrong value at (" << j << ", " << i << "): " << O[i][j] << " != " << C[i][j] << endl;
        return 1;
      }
    }
  }
  cout << "Success!!" << endl;

  return 0;
}
