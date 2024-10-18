#include "stdint.h"
#include "math.h"

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

__host__ __device__
inline double diff_norm_3x3(const double *R1, const double *R2) {
  /* difference 2 matrices and return squared norm: ||R1-R2||^2 */
  int i;
  double norm = 0;
  for (i=0; i<9; i++) {norm += (R1[i]-R2[i])*(R1[i]-R2[i]);}
  return norm;
}


extern "C" __global__
void signs_times_v(int n, double* Rijs, const double* vec, double* new_vec, bool J_weighting)
{
  /* thread index (1d), represents "i" index */
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  /* no-op when out of bounds */
  if(i >= n) return;

  double c[4];
  unsigned int j;
  unsigned int k;
  for(k=0;k<4;k++){c[k]=0;}
  unsigned long ij, jk, ik;
  int best_i;
  double best_val;
  double s_ij_jk, s_ik_jk, s_ij_ik;
  double alt_ij_jk, alt_ij_ik, alt_ik_jk;

  double *Rij, *Rjk, *Rik;
  double JRijJ[9], JRjkJ[9], JRikJ[9];
  double tmp[9];

  int signs_confs[4][3];
  for(int a=0; a<4; a++) { for(k=0; k<3; k++) { signs_confs[a][k]=1; } }
  signs_confs[1][0]=-1; signs_confs[1][2]=-1;
  signs_confs[2][0]=-1; signs_confs[2][1]=-1;
  signs_confs[3][1]=-1; signs_confs[3][2]=-1;

  /* initialize alternatives */
  /* when we find the best J-configuration, we also compare it to the alternative 2nd best one.
   * this comparison is done for every pair in the triplete independently. to make sure that the
   * alternative is indeed different in relation to the pair, we document the differences between
   * the configurations in advance:
   * ALTS(:,best_conf,pair) = the two configurations in which J-sync differs from
   * best_conf in relation to pair */

  int ALTS[2][4][3];
  ALTS[0][0][0]=1; ALTS[0][1][0]=0; ALTS[0][2][0]=0; ALTS[0][3][0]=1;
  ALTS[1][0][0]=2; ALTS[1][1][0]=3; ALTS[1][2][0]=3; ALTS[1][3][0]=2;
  ALTS[0][0][1]=2; ALTS[0][1][1]=2; ALTS[0][2][1]=0; ALTS[0][3][1]=0;
  ALTS[1][0][1]=3; ALTS[1][1][1]=3; ALTS[1][2][1]=1; ALTS[1][3][1]=1;
  ALTS[0][0][2]=1; ALTS[0][1][2]=0; ALTS[0][2][2]=1; ALTS[0][3][2]=0;
  ALTS[1][0][2]=3; ALTS[1][1][2]=2; ALTS[1][2][2]=3; ALTS[1][3][2]=2;


  for(j=i+1; j< (n - 1); j++){
    ij = PAIR_IDX(n, i, j);
    for(k=j+1; k< n; k++){
      ik = PAIR_IDX(n, i, k);
      jk = PAIR_IDX(n, j, k);

      /* compute configurations matches scores */
      Rij = Rijs + 9*ij;
      Rjk = Rijs + 9*jk;
      Rik = Rijs + 9*ik;

      JRJ(Rij, JRijJ);
      JRJ(Rjk, JRjkJ);
      JRJ(Rik, JRikJ);

      mult_3x3(tmp, Rij, Rjk);
      c[0] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, JRijJ, Rjk);
      c[1] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, Rij, JRjkJ);
      c[2] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, Rij, Rjk);
      c[3] = diff_norm_3x3(tmp, JRikJ);

      /* find best match */
      best_i=0; best_val=c[0];
      if (c[1]<best_val) {best_i=1; best_val=c[1];}
      if (c[2]<best_val) {best_i=2; best_val=c[2];}
      if (c[3]<best_val) {best_i=3; best_val=c[3];}

      /* set triangles entries to be signs */
      s_ij_jk = signs_confs[best_i][0];
      s_ik_jk = signs_confs[best_i][1];
      s_ij_ik = signs_confs[best_i][2];

      /* J weighting */
      if(J_weighting){
        /* for each triangle side, find the best alternative */
        alt_ij_jk = c[ALTS[0][best_i][0]];
        if (c[ALTS[1][best_i][0]] < alt_ij_jk){
          alt_ij_jk = c[ALTS[1][best_i][0]];
        }

        alt_ik_jk = c[ALTS[0][best_i][1]];
        if (c[ALTS[1][best_i][1]] < alt_ik_jk){
          alt_ik_jk = c[ALTS[1][best_i][1]];
        }
        alt_ij_ik = c[ALTS[0][best_i][2]];
        if (c[ALTS[1][best_i][2]] < alt_ij_ik){
          alt_ij_ik = c[ALTS[1][best_i][2]];
        }

        /* Update scores */
        s_ij_jk *= 1 - sqrt(best_val / alt_ij_jk);
        s_ik_jk *= 1 - sqrt(best_val / alt_ik_jk);
        s_ij_ik *= 1 - sqrt(best_val / alt_ij_ik);
      }


      /* update multiplication */
      atomicAdd(&(new_vec[ij]), s_ij_jk*vec[jk] + s_ij_ik*vec[ik]);
      atomicAdd(&(new_vec[jk]), s_ij_jk*vec[ij] + s_ik_jk*vec[ik]);
      atomicAdd(&(new_vec[ik]), s_ij_ik*vec[ij] + s_ik_jk*vec[jk]);

    } /* k */
  } /* j */

  return;
};

extern "C" __global__
void pairs_probabilities(int n, double* Rijs, double P2, double A, double a, double B, double b, double x0, double* ln_f_ind, double* ln_f_arb)
{
  /* thread index (1d), represents "i" index */
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  /* no-op when out of bounds */
  if(i >= n) return;

  double c[4];
  unsigned int j;
  unsigned int k;
  for(k=0;k<4;k++){c[k]=0;}
  unsigned long ij, jk, ik;
  int best_i;
  double best_val;
  double s_ij_jk, s_ik_jk, s_ij_ik;
  double alt_ij_jk, alt_ij_ik, alt_ik_jk;
  double f_ij_jk, f_ik_jk, f_ij_ik;


  double *Rij, *Rjk, *Rik;
  double JRijJ[9], JRjkJ[9], JRikJ[9];
  double tmp[9];

  int signs_confs[4][3];
  for(int a=0; a<4; a++) { for(k=0; k<3; k++) { signs_confs[a][k]=1; } }
  signs_confs[1][0]=-1; signs_confs[1][2]=-1;
  signs_confs[2][0]=-1; signs_confs[2][1]=-1;
  signs_confs[3][1]=-1; signs_confs[3][2]=-1;

  /* initialize alternatives */
  /* when we find the best J-configuration, we also compare it to the alternative 2nd best one.
   * this comparison is done for every pair in the triplete independently. to make sure that the
   * alternative is indeed different in relation to the pair, we document the differences between
   * the configurations in advance:
   * ALTS(:,best_conf,pair) = the two configurations in which J-sync differs from
   * best_conf in relation to pair */

  int ALTS[2][4][3];
  ALTS[0][0][0]=1; ALTS[0][1][0]=0; ALTS[0][2][0]=0; ALTS[0][3][0]=1;
  ALTS[1][0][0]=2; ALTS[1][1][0]=3; ALTS[1][2][0]=3; ALTS[1][3][0]=2;
  ALTS[0][0][1]=2; ALTS[0][1][1]=2; ALTS[0][2][1]=0; ALTS[0][3][1]=0;
  ALTS[1][0][1]=3; ALTS[1][1][1]=3; ALTS[1][2][1]=1; ALTS[1][3][1]=1;
  ALTS[0][0][2]=1; ALTS[0][1][2]=0; ALTS[0][2][2]=1; ALTS[0][3][2]=0;
  ALTS[1][0][2]=3; ALTS[1][1][2]=2; ALTS[1][2][2]=3; ALTS[1][3][2]=2;


  for(j=i+1; j< (n - 1); j++){
    ij = PAIR_IDX(n, i, j);
    for(k=j+1; k< n; k++){
      ik = PAIR_IDX(n, i, k);
      jk = PAIR_IDX(n, j, k);

      /* compute configurations matches scores */
      Rij = Rijs + 9*ij;
      Rjk = Rijs + 9*jk;
      Rik = Rijs + 9*ik;

      JRJ(Rij, JRijJ);
      JRJ(Rjk, JRjkJ);
      JRJ(Rik, JRikJ);

      mult_3x3(tmp, Rij, Rjk);
      c[0] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, JRijJ, Rjk);
      c[1] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, Rij, JRjkJ);
      c[2] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, Rij, Rjk);
      c[3] = diff_norm_3x3(tmp, JRikJ);

      /* find best match */
      best_i=0; best_val=c[0];
      if (c[1]<best_val) {best_i=1; best_val=c[1];}
      if (c[2]<best_val) {best_i=2; best_val=c[2];}
      if (c[3]<best_val) {best_i=3; best_val=c[3];}

      /* for each triangle side, find the best alternative */
      alt_ij_jk = c[ALTS[0][best_i][0]];
      if (c[ALTS[1][best_i][0]] < alt_ij_jk){
        alt_ij_jk = c[ALTS[1][best_i][0]];
      }

      alt_ik_jk = c[ALTS[0][best_i][1]];
      if (c[ALTS[1][best_i][1]] < alt_ik_jk){
        alt_ik_jk = c[ALTS[1][best_i][1]];
      }
      alt_ij_ik = c[ALTS[0][best_i][2]];
      if (c[ALTS[1][best_i][2]] < alt_ij_ik){
        alt_ij_ik = c[ALTS[1][best_i][2]];
      }

      /* Assign scores */
      s_ij_jk = 1 - sqrt(best_val / alt_ij_jk);
      s_ik_jk = 1 - sqrt(best_val / alt_ik_jk);
      s_ij_ik = 1 - sqrt(best_val / alt_ij_ik);


      /* the probability of a pair ij to have the observed triangles scores,
         given it has an indicative common line */
      f_ij_jk = log( P2*(B*pow(1-s_ij_jk,b)*exp(-b/(1-x0)*(1-s_ij_jk))) + (1-P2)*A*pow((1-s_ij_jk),a) );
      f_ik_jk = log( P2*(B*pow(1-s_ik_jk,b)*exp(-b/(1-x0)*(1-s_ik_jk))) + (1-P2)*A*pow((1-s_ik_jk),a) );
      f_ij_ik = log( P2*(B*pow(1-s_ij_ik,b)*exp(-b/(1-x0)*(1-s_ij_ik))) + (1-P2)*A*pow((1-s_ij_ik),a) );
      atomicAdd(&(ln_f_ind[ij]), f_ij_jk + f_ij_ik);
      atomicAdd(&(ln_f_ind[jk]), f_ij_jk + f_ik_jk);
      atomicAdd(&(ln_f_ind[ik]), f_ik_jk + f_ij_ik);

      /* the probability of a pair ij to have the observed triangles scores,
         given it has an arbitrary common line */
      f_ij_jk = log( A*pow((1-s_ij_jk),a) );
      f_ik_jk = log( A*pow((1-s_ik_jk),a) );
      f_ij_ik = log( A*pow((1-s_ij_ik),a) );
      atomicAdd(&(ln_f_arb[ij]), f_ij_jk + f_ij_ik);
      atomicAdd(&(ln_f_arb[jk]), f_ij_jk + f_ik_jk);
      atomicAdd(&(ln_f_arb[ik]), f_ik_jk + f_ij_ik);


    } /* k */
  } /* j */

  return;
};


extern "C" __global__
void triangle_scores_inner(int n, double* Rijs, int n_intervals, unsigned int* scores_hist)
{
  /* thread index (1d), represents "i" index */
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  /* no-op when out of bounds */
  if(i >= n) return;

  double c[4];
  unsigned int j;
  unsigned int k;
  for(k=0;k<4;k++){c[k]=0;}
  unsigned long ij, jk, ik;
  int best_i;
  double best_val;
  double s_ij_jk, s_ik_jk, s_ij_ik;
  double alt_ij_jk, alt_ij_ik, alt_ik_jk;
  unsigned int l1,l2,l3;
  double threshold;
  double h = 1. / n_intervals;

  double *Rij, *Rjk, *Rik;
  double JRijJ[9], JRjkJ[9], JRikJ[9];
  double tmp[9];

  /* initialize alternatives */
  /* when we find the best J-configuration, we also compare it to the alternative 2nd best one.
   * this comparison is done for every pair in the triplete independently. to make sure that the
   * alternative is indeed different in relation to the pair, we document the differences between
   * the configurations in advance:
   * ALTS(:,best_conf,pair) = the two configurations in which J-sync differs from
   * best_conf in relation to pair */

  int ALTS[2][4][3];
  ALTS[0][0][0]=1; ALTS[0][1][0]=0; ALTS[0][2][0]=0; ALTS[0][3][0]=1;
  ALTS[1][0][0]=2; ALTS[1][1][0]=3; ALTS[1][2][0]=3; ALTS[1][3][0]=2;
  ALTS[0][0][1]=2; ALTS[0][1][1]=2; ALTS[0][2][1]=0; ALTS[0][3][1]=0;
  ALTS[1][0][1]=3; ALTS[1][1][1]=3; ALTS[1][2][1]=1; ALTS[1][3][1]=1;
  ALTS[0][0][2]=1; ALTS[0][1][2]=0; ALTS[0][2][2]=1; ALTS[0][3][2]=0;
  ALTS[1][0][2]=3; ALTS[1][1][2]=2; ALTS[1][2][2]=3; ALTS[1][3][2]=2;


  for(j=i+1; j< (n - 1); j++){
    ij = PAIR_IDX(n, i, j);
    for(k=j+1; k< n; k++){
      ik = PAIR_IDX(n, i, k);
      jk = PAIR_IDX(n, j, k);

      /* compute configurations matches scores */
      Rij = Rijs + 9*ij;
      Rjk = Rijs + 9*jk;
      Rik = Rijs + 9*ik;

      JRJ(Rij, JRijJ);
      JRJ(Rjk, JRjkJ);
      JRJ(Rik, JRikJ);

      mult_3x3(tmp, Rij, Rjk);
      c[0] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, JRijJ, Rjk);
      c[1] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, Rij, JRjkJ);
      c[2] = diff_norm_3x3(tmp, Rik);

      mult_3x3(tmp, Rij, Rjk);
      c[3] = diff_norm_3x3(tmp, JRikJ);

      /* find best match */
      best_i=0; best_val=c[0];
      if (c[1]<best_val) {best_i=1; best_val=c[1];}
      if (c[2]<best_val) {best_i=2; best_val=c[2];}
      if (c[3]<best_val) {best_i=3; best_val=c[3];}

      /* for each triangle side, find the best alternative */
      alt_ij_jk = c[ALTS[0][best_i][0]];
      if (c[ALTS[1][best_i][0]] < alt_ij_jk){
        alt_ij_jk = c[ALTS[1][best_i][0]];
      }

      alt_ik_jk = c[ALTS[0][best_i][1]];
      if (c[ALTS[1][best_i][1]] < alt_ik_jk){
        alt_ik_jk = c[ALTS[1][best_i][1]];
      }
      alt_ij_ik = c[ALTS[0][best_i][2]];
      if (c[ALTS[1][best_i][2]] < alt_ij_ik){
        alt_ij_ik = c[ALTS[1][best_i][2]];
      }

      /* Assign scores */
      s_ij_jk = 1 - sqrt(best_val / alt_ij_jk);
      s_ik_jk = 1 - sqrt(best_val / alt_ik_jk);
      s_ij_ik = 1 - sqrt(best_val / alt_ij_ik);

      /* update scores histogram */
      threshold = 0;
      for (l1=0; l1<n_intervals-1; l1++) {
        threshold += h;
        if (s_ij_jk < threshold) {break;}
      }

      threshold = 0;
      for(l2=0; l2<n_intervals-1; l2++) {
        threshold += h;
        if(s_ik_jk < threshold) {break;}
      }

      threshold = 0;
      for(l3=0; l3<n_intervals-1; l3++) {
        threshold += h;
        if (s_ij_ik < threshold) {break;}
      }

      atomicAdd(&(scores_hist[l1]), 1);
      atomicAdd(&(scores_hist[l2]), 1);
      atomicAdd(&(scores_hist[l3]), 1);

    } /* k */
  } /* j */

  return;
};

extern "C" __global__
void estimate_all_angles1(int j,
                          int n,
                          int n_theta,
                          double hist_bin_width,
                          int full_width,
                          double sigma,
                          int sync,
                          int16_t* __restrict__ clmatrix,
                          double* __restrict__ hist,
                          uint16_t* __restrict__ k_map,
                          double* __restrict__ angles_map,
                          double* __restrict__ angles)
{
  /* n n_img */
  /* j is image j index */

  /* thread index represents "i" index */
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  /* thread index represents "k" index */
  const unsigned int k = blockDim.y * blockIdx.y + threadIdx.y;

  int cl_diff1, cl_diff2, cl_diff3;
  double theta1, theta2, theta3;
  double c1, c2, c3;
  double cond;
  double cos_phi2;
  double w_theta_need;

  /* no-op when out of bounds */
  if(i >= n) return;
  if(k >= n) return;
  if(i >= j) return;
  /*
    These are also tested later via the clmatrix values,
    testing now avoids extra reads.
  */
  if(k==i) return;
  if(k==j) return;

  int map_idx; /* tmp index var */

  int cl_idx12, cl_idx21;
  int cl_idx13, cl_idx31;
  int cl_idx23, cl_idx32;
  const int ntics = 180. / hist_bin_width;
  const double TOL_idx = 1e-12;
  bool ind1, ind2;
  double grid_angle, angle_diff, angle;
  int b;
  const double two_sigma_sq = 2*sigma*sigma;


  const int pair_idx = PAIR_IDX(n,i,j);

  cl_idx12 = clmatrix[i*n + j];
  cl_idx21 = clmatrix[j*n + i];
  /*
    MATLAB code indicated this condition might occur outside i==j;
    Ask Yoel what other reasons this would occur.
  */
  if(cl_idx12 == -1) return;

  /* Assume that k_list starts as all n images */

  cl_idx13 = clmatrix[i*n + k];
  cl_idx31 = clmatrix[k*n + i];
  cl_idx23 = clmatrix[j*n + k];
  cl_idx32 = clmatrix[k*n + j];

  /* test `k` values */
  if(cl_idx13 == -1) return;  /* i, k */
  if(cl_idx23 == -1) return;  /* j, k */

  /* get cosine angles */
  cl_diff1 = cl_idx13 - cl_idx12;
  cl_diff2 = cl_idx23 - cl_idx21;
  cl_diff3 = cl_idx32 - cl_idx31;

  theta1 = cl_diff1 * 2 * M_PI / n_theta;
  theta2 = cl_diff2 * 2 * M_PI / n_theta;
  theta3 = cl_diff3 * 2 * M_PI / n_theta;

  c1 = cos(theta1);
  c2 = cos(theta2);
  c3 = cos(theta3);

  /* test if we have a good index */
  cond = 1 + 2 * c1 * c2 * c3 - (c1*c1 + c2*c2 + c3*c3);
  if(cond <= 1e-5) return;  /* current value of k is not good, skip */

  /* Calculated cos values of angle between i and j images */
  if( sync == 1){

    cos_phi2 = (c3 - c1*c2) / (sqrt(1 - c1*c1) * sqrt(1 - c2*c2));

    /*
      Some synchronization must be applied when common line is out by 180 degrees.
      Here fix the angles between c_ij(c_ji) and c_ik(c_jk) to be smaller than pi/2,
      otherwise there will be an ambiguity between alpha and pi-alpha.
    */

    /* Check sync conditions */
    ind1 = (theta1 > (M_PI + TOL_idx)) || (
        (theta1 < -TOL_idx) && (theta1 > -M_PI)
                                           );
    ind2 = (theta2 > (M_PI + TOL_idx)) || (
        (theta2 < -TOL_idx) && (theta2 > -M_PI)
                                           );
    if( (ind1 && !ind2) || (!ind1 && ind2)){
      /* Apply sync */
      cos_phi2 = -cos_phi2;
    }

  }  /* end sync */
  else{
    cos_phi2 = (c3 - c1*c2 ) / (sin(theta1) * sin(theta2));
  } /* end not sync */

  /* clip cosine phi between [-1,1] */
  if(cos_phi2 > 1){
    cos_phi2 = 1;
  }
  if(cos_phi2 < -1){
    cos_phi2 = -1;
  }

  /* compute histogram contribution, angle mapping, and index mappings. */
  angle = acos(cos_phi2) * 180. / M_PI;
  /* index of angle's bin */
  map_idx = i*n + k;
  /*
    For each k, keep track of bin and angles.
    Note, this is slightly different than the host
    which uses slightly different angle/hist grids (likely an oversight).
  */
  k_map[map_idx] = angle / hist_bin_width;
  angles_map[map_idx] = angle;  /* degrees */
  for(b=0; b<ntics; b++){
    /* Potential optimization, just compute in radians to avoid extra arithmetic. */
    grid_angle = b * (180./ntics);  /* grid angle */
    angle_diff = (grid_angle - angle);
    /* accumulate histogram contribution, atomic due to concurrent `k` */
    atomicAdd(&(hist[i*ntics + b ]),
              exp(-(angle_diff*angle_diff) / ( two_sigma_sq)));
  } /* b bins */

  /*
    At this point, the kernel should have accumulated all "good"
    k contributions into the histogram for I, j.
    The next kernel solves the histogram.

    The two known euler angles are initialized while we have the cl_idx values.
  */

  /* Initialize euler angles */
  /* Can be done once, but checking seemed to make the kernel slower... */
  {
    map_idx = pair_idx*3;
    angles[map_idx    ] = cl_idx12 * 2 * M_PI / n_theta + M_PI / 2;
    angles[map_idx + 1] = 0;
    angles[map_idx + 2] = -M_PI / 2 - cl_idx21 * 2 * M_PI / n_theta;
  }
} /* estimate_all_angles1 kernel */




extern "C" __global__
void estimate_all_angles2(int j,
                          int n,
                          double hist_bin_width,
                          int full_width,
                          double* __restrict__ hist,
                          uint16_t* __restrict__ k_map,
                          double* __restrict__ angles_map,
                          double* __restrict__ angles)
{
  /* n n_img */
  /* j is image j index */

  /* thread index (1d), represents image i index */
  const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  int k;
  int cnt;
  double w_theta_needed;

  /* no-op when out of bounds */
  if(i >= n) return;
  if(i >= j) return;

  int map_idx; /* tmp index var */

  const int ntics = 180. / hist_bin_width;
  int b;
  int peak_idx;
  double peak;

  const int pair_idx = PAIR_IDX(n,i,j);

  /* Find peak and peak index in histogram */
  peak = -99999;
  peak_idx = -1;
  for(b=0; b<ntics; b++){
    map_idx = i*ntics + b;
    if(hist[map_idx] > peak){
      peak = hist[map_idx];
      peak_idx = b;
    }
  }

  /* find mean of rotations */

  if(full_width==-1){
    /* adaptive width*/
    w_theta_needed = 0;
    cnt = 0;
    while(cnt == 0){
      /* broaden search width */
      w_theta_needed += hist_bin_width;
      /* find satisfying indices */
      for(k=0; k<n; k++){
        /* determine if image k in peak bin(s) */
        // Perhaps transpose the maps so thread i fast
        map_idx = i*n + k;
        if(abs(k_map[map_idx] - peak_idx) < w_theta_needed){
          cnt += 1;  /* count this image */
          angles[pair_idx*3 + 1] += angles_map[map_idx];  /* accumulate angle */
        } /* < w_theta_needed */
      } /* k */
    } /* cnt */
  } /* full_width -1, adaptive */
  else {
    /* fixed width */
    cnt = 0;
    /* determine if image k in peak bin(s) */
    for(k=0; k<n; k++){
      map_idx = i*n + k;
      if(abs(k_map[map_idx] - peak_idx) < full_width){
        cnt += 1;  /* count this image */
        angles[pair_idx*3 + 1] += angles_map[map_idx];  /* accumulate angle */
      } /* full_width */
    } /* k */
  } /* fixed width */

  /* Divide accumulated angles (resulting in the mean alpha angle) */
  // (todo, can we have cnt = 0?)
  /* convert degree to radian */
  angles[pair_idx*3 + 1] *= M_PI / (180*cnt);

} /* estimate_all_angles2 kernel */

extern "C" __global__
void angles_to_rots(int n,
                    double* __restrict__ angles,
                    double* __restrict__ rotations)
{
  /* Convert stack of ZYZ Euler angles to stack of rotation matrices */

  /* thread index (1d), represents "i" index */
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  /* no-op when out of bounds */
  if(i >= n) return;

  ang2orth(&(rotations[i*9]), angles[i*3], angles[i*3+1], angles[i*3+2]);

}
