
extern "C" __global__
void build_clmatrix_kernel(int n, int m, int r, double* pf, double* clmatrix, double* cl_dist, double* shifts_1d, int n_shifts, int* shifts,  double* shift_phases)
{
  /* n n_img */
  /* m,r st (n, m, r) = pf.shape, ie len(pf[i])  */

  /* thread index (1d), represents "i" index */
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

  /* no-op when out of bounds */
  if(i >= n) return;
  if(j >= n) return;
  /* no-op lower triangle */
  if(j <= i) return;

  int ind;
  int k;
  int s;
  int cl1, cl2;
  int best_cl1, best_cl2, best_s;
  double dist, best_cl_dist;
  double p1, p2;

  double p1_realk, p1_imagk;
  double p2conj_realk, p2conj_imagk;
  double p2_realk, p2_imagk;
  double shift_phases_real, shift_phases_imag;


  best_s = -99999;
  best_cl1 = -1;
  best_cl2 = -1;
  best_cl_dist = -1/0;

  for(cl1=0; cl1<m; cl1++){
    for(cl2=0; cl2<m; cl2++){
      for(s=0; s<n_shifts; s++){
        p1 = 0;
        p2 = 0;
        /* inner most dim of dot (matmul) */
        for(k=0; k<r; k++){
          /* factor */
          p1_realk = pf[2*(i*m*r + cl1*r + k)];
          p1_imagk = pf[2*(i*m*r + cl1*r + k) + 1];
          p2conj_realk =  pf[2*(j*m*r + cl2*r + k)];
          p2conj_imagk = -pf[2*(j*m*r + cl2*r + k) + 1];
          shift_phases_real = shift_phases[2*(s*r + k)];
          shift_phases_imag = shift_phases[2*(s*r + k) +1];
          p2_realk = (p2conj_realk * shift_phases_real) - (p2conj_imagk*shift_phases_imag);
          p2_imagk = (p2conj_imagk * shift_phases_real) + (p2conj_realk*shift_phases_imag);
          p1 += p1_realk * p2_realk;
          p2 += p1_imagk * p2_imagk;

        }

        dist = p1 - p2;
        if(dist > best_cl_dist){
          best_cl_dist = dist;
          best_cl1 = cl1;
          best_cl2 = cl2;
          best_s = s;
        }

        dist = p1 + p2;
        if(dist > best_cl_dist){
          best_cl_dist = dist;
          best_cl1 = cl1;
          best_cl2 = cl2 + m; // m is pf.shape[1], which should be n_theta//2...
          best_s = s;
        }

      } /* s */
    } /* cl2 */
  }/* cl1 */


  /* update global best for i, j*/
  ind = i*n + j;
  clmatrix[ind] = best_cl1;
  clmatrix[j*n+i] = best_cl2;  /* [j,i] */
  cl_dist[ind] = 2*best_cl_dist;  // 2 of mystery
  shifts_1d[ind] = shifts[best_s];

}
