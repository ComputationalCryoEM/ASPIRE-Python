#pragma once
#include <math.h>
#include <stdint.h>

/* From i,j indices to the common index in the N-choose-2 sized array */
/* Careful, this is strictly the upper triangle! */
#define PAIR_IDX(N,I,J) ((2*N-I-1)*I/2 + J-I-1)


/* convert euler angles (a,b,c) in ZYZ to rotation matrix r */
__host__ __device__
inline void ang2orth(double* r, double a, double b, double c){
  double sa = sin(a);
  double sb = sin(b);
  double sc = sin(c);
  double ca = cos(a);
  double cb = cos(b);
  double cc = cos(c);

  /* ZYZ Proper Euler angles */
  /* https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix */
  r[0] = ca*cb*cc - sa*sc;
  r[1] = -cc*sa -ca*cb*sc;
  r[2] = ca*sb;
  r[3] = ca*sc + cb*cc*sa;
  r[4] = ca*cc - cb*sa*sc;
  r[5] = sa*sb;
  r[6] = -cc*sb;
  r[7] = sb*sc;
  r[8] = cb;
}


/* Matrix multiplication: out = R1 * R2 */
__host__ __device__
inline void mult_3x3(double *out, double *R1, double *R2) {
  /* 3X3 matrices multiplication: out = R1*R2
   * Note, this differs from the MATLAB mult_3x3.
   */

  int i,j,k;

  for(i=0; i<3; i++){
    for(j=0; j<3; j++){
      out[i*3 + j] = 0;
      for (k=0; k<3; k++){
        out[i*3 + j] += R1[i*3+k] * R2[k*3+j];
      }
    }
  }
}


/* Multiply 3x3 matrix by J on both sides: A = J R J */
__host__ __device__
inline void JRJ(double *R, double *A) {
  /* multiple 3X3 matrix by J from both sizes: A = JRJ */
  A[0]=R[0];
  A[1]=R[1];
  A[2]=-R[2];
  A[3]=R[3];
  A[4]=R[4];
  A[5]=-R[5];
  A[6]=-R[6];
  A[7]=-R[7];
  A[8]=R[8];
}


/* Squared Frobenius norm of R1 - R2 */
__host__ __device__
inline double diff_norm_3x3(const double *R1, const double *R2) {
  /* difference 2 matrices and return squared norm: ||R1-R2||^2 */
  int i;
  double norm = 0;
  for (i=0; i<9; i++) {norm += (R1[i]-R2[i])*(R1[i]-R2[i]);}
  return norm;
}
