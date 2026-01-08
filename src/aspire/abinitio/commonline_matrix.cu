#include <stdint.h>
#include <math.h>
#include <cupy/complex.cuh>

extern "C" __global__
void build_clmatrix_kernel(
    const int n,
    const int m,
    const int r,
    const complex<double>* __restrict__ pf,
    int16_t* const __restrict__ clmatrix,
    const int n_shifts,
    const complex<double>* const __restrict__ shift_phases)
{
  /* n n_img */
  /* m angular componentns, n_theta//2 */
  /* r radial componentns */
  /* (n, m, r) = pf.shape in python (before transpose for CUDA kernel) */

  /* thread index (2d), represents "i" and "j" indices */
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

  /* no-op when out of bounds */
  if(i >= n) return;
  if(j >= n) return;
  /* no-op lower triangle */
  if(j <= i) return;

  int k;
  int s;
  int cl1, cl2;
  int best_cl1, best_cl2;
  double xcorr, best_cl_xcorr;
  double p1, p2;
  complex<double> pfik, pfjk;

  best_cl1 = -1;
  best_cl2 = -1;
  best_cl_xcorr = -INFINITY;

  for(cl1=0; cl1<m; cl1++){
    for(cl2=0; cl2<m; cl2++){
      for(s=0; s<n_shifts; s++){
        p1 = 0;
        p2 = 0;
        /* inner most dim of dot (matmul) */
        for(k=0; k<r; k++){
          pfik = pf[k*m*n + cl1*n + i];
          pfjk = conj(pf[k*m*n + cl2*n + j]) * shift_phases[s*r + k];
          p1 += real(pfik) * real(pfjk);
          p2 += imag(pfik) * imag(pfjk);
        } /* k */

        xcorr = p1 - p2;
        if(xcorr > best_cl_xcorr){
          best_cl_xcorr = xcorr;
          best_cl1 = cl1;
          best_cl2 = cl2;
        }

        xcorr = p1 + p2;
        if(xcorr > best_cl_xcorr){
          best_cl_xcorr = xcorr;
          best_cl1 = cl1;
          best_cl2 = cl2 + m; /* m is pf.shape[1], which should be n_theta//2 */
        }

      } /* s */
    } /* cl2 */
  }/* cl1 */

  /* update global best for i, j */
  clmatrix[i*n + j] = best_cl1;
  clmatrix[j*n+i] = best_cl2;  /* [j,i] */

} /* build_clmatrix_kernel */

extern "C" __global__
void fbuild_clmatrix_kernel(
    const int n,
    const int m,
    const int r,
    const complex<float>* __restrict__ pf,
    int16_t* const __restrict__ clmatrix,
    const int n_shifts,
    const complex<float>* const __restrict__ shift_phases)
{
  /* n n_img */
  /* m angular componentns, n_theta//2 */
  /* r radial componentns */
  /* (n, m, r) = pf.shape in python (before transpose for CUDA kernel) */

  /* thread index (2d), represents "i" and "j" indices */
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

  /* no-op when out of bounds */
  if(i >= n) return;
  if(j >= n) return;
  /* no-op lower triangle */
  if(j <= i) return;

  int k;
  int s;
  int cl1, cl2;
  int best_cl1, best_cl2;
  float xcorr, best_cl_xcorr;
  float p1, p2;
  complex<float> pfik, pfjk;

  best_cl1 = -1;
  best_cl2 = -1;
  best_cl_xcorr = -INFINITY;

  for(cl1=0; cl1<m; cl1++){
    for(cl2=0; cl2<m; cl2++){
      for(s=0; s<n_shifts; s++){
        p1 = 0;
        p2 = 0;
        /* inner most dim of dot (matmul) */
        for(k=0; k<r; k++){
          pfik = pf[k*m*n + cl1*n + i];
          pfjk = conj(pf[k*m*n + cl2*n + j]) * shift_phases[s*r + k];
          p1 += real(pfik) * real(pfjk);
          p2 += imag(pfik) * imag(pfjk);
        } /* k */

        xcorr = p1 - p2;
        if(xcorr > best_cl_xcorr){
          best_cl_xcorr = xcorr;
          best_cl1 = cl1;
          best_cl2 = cl2;
        }

        xcorr = p1 + p2;
        if(xcorr > best_cl_xcorr){
          best_cl_xcorr = xcorr;
          best_cl1 = cl1;
          best_cl2 = cl2 + m; /* m is pf.shape[1], which should be n_theta//2 */
        }

      } /* s */
    } /* cl2 */
  }/* cl1 */

  /* update global best for i, j */
  clmatrix[i*n + j] = best_cl1;
  clmatrix[j*n+i] = best_cl2;  /* [j,i] */

} /* fbuild_clmatrix_kernel */
