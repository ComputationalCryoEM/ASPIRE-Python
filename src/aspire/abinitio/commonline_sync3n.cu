# define M_PI           3.14159265358979323846  /* pi */

/* from i,j indices to the common index in the N-choose-2 sized array */
#define PAIR_IDX(N,I,J) ((2*N-I-1)*I/2 + J-I-1)

/* convert euler angles (a,b,c) in ZYZ to rotation matrix r */
inline void ang2orth(double* r, double a, double b, double c){
  double sa = sin(a);
  double sb = sin(b);
  double sc = sin(c);
  double ca = cos(a);
  double cb = cos(b);
  double cc = cos(c);

  r[0] = ca*cb*cc - sa*sc;
  r[1] = -ca*cb*cc - sa*sc;
  r[2] = ca*sb;
  r[3] = sa*cb*cc + ca*sc;
  r[4] = -sa*cb*cc + ca*sc;
  r[5] = sa*sb;
  r[6] = -sb*cc;
  r[7] = sb*sc;
  r[8] = cb;  
}
  

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
void estimate_all_Rijs(int n, int n_theta, double* __restrict__ clmatrix, double* __restrict__ hist, int* __restrict__ kmap, double* rotations)
{
  // try toget kmap as uint16_t
  /* n n_img */

  /* thread index (2d), represents "i" index, "j" index */
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k;
  int kk;
  int cl_diff1, cl_diff2, cl_diff3;
  double theta1, theta2, theta3;
  double c1, c2, c3;
  double cond;
  double cos_phi2;
  double angles[3];
  double r[9];
  int cnt;

  /* no-op when out of bounds */
  if(i >= n) return;
  if(j >= n) return;
  /* no-op lower triangle */
  if(j <= i) return;


  // vote_ij creates good_k list per (i,j)

  // We shouldn't need this... confirm and rm later.
  if(clmatrix[i*n + j] == -1) return;

  int cl_idx12 = clmatrix[i*n + j];
  int cl_idx21 = clmatrix[j*n + i];
  int cl_idx13, cl_idx31;
  int cl_idx23, cl_idx32;
  const int ntics = 60;
  const double sigma = 3.0;
  double ga;
  double angle;
  int b;
  int peak_idx;
  double peak;
  
  /* Assume that k_list starts as [0,n] */
  for(k=0; k<n; k++){

    cl_idx13 = clmatrix[i*n + k];
    cl_idx31 = clmatrix[k*n + i];
    cl_idx23 = clmatrix[j*n + k];
    cl_idx32 = clmatrix[k*n + j];

    // test `k` values
    if(k==i) return;
    if(cl_idx13 == -1) return;  // i, k
    if(cl_idx31 == -1) return;  // k, i
    if(cl_idx23 == -1) return;  // j, k
    // if(cl_idx32 == -1) return;  // k, j

    // self._get_cos_phis(cl_diff1, cl_diff2, cl_diff3, n_theta)
    cl_diff1 = cl_idx13 - cl_idx12;
    cl_diff2 = cl_idx21 - cl_idx23;
    cl_diff3 = cl_idx32 - cl_idx31;

    theta1 = cl_diff1 * 2 * M_PI / n_theta;
    theta2 = cl_diff2 * 2 * M_PI / n_theta;
    theta3 = cl_diff3 * 2 * M_PI / n_theta;

    c1 = cos(theta1);
    c2 = cos(theta2);
    c3 = cos(theta3);

    cond = 1 + 2 * c1 * c2 * c3 - (c1*c1 + c2*c2 + c3*c3);

    // test if we have a good_idx
    if(cond < 1e-5) return;

    cos_phi2 = (c3 - c1*c2 ) / (sin(theta1) * sin(theta2));
    
    //end _get_cos_phis

    // clip [-1,1]
    if(cos_phi2 >1){
      cos_phi2 = 1;
    }
    if(cos_phi2 <-1){
      cos_phi2 = -1;
    }
                   
    /* compute histogram contribution and index map */
    angle = acos(cos_phi2) * 180. / M_PI;
    // angle's bin
    kmap[PAIR_IDX(n,i,j)*n + k] = angle / ntics;
    for(b=0; b<ntics; b++){
      ga = b*(180/ntics);  // grid angle // todo, just compute in radians to avoid extra arithmetic
      // histogram contribution
      hist[PAIR_IDX(n,i,j)*ntics + b ] += exp(
          (2*angle*ga - (angle*angle + ga*ga))/(2*sigma*sigma));
    } /* bins */
  } /* k*/
  
  /* find peak of histogram */
  peak = -1;
  peak_idx = -1;
  for(b=0; b<ntics; b++){
    if(hist[PAIR_IDX(n,i,j)*ntics + b] > peak){
      peak = hist[PAIR_IDX(n,i,j)*ntics + b];
      peak_idx = b;
    }
  }
  
  /* find mean of rotations */
  // initialize
  cnt = 0;

  // _rotratio_eulerangle_vec loops over good_k list per (i,j)
  // find satisfying indices  
  for(k=0; k<n; k++){
    // image k in peak bin
    if(abs(kmap[PAIR_IDX(n,i,j)*n + k] - peak_idx) < 2){
      cnt += 1;
      // convert to euler angles  // check
      angles[0] = cl_idx12 * 2 * M_PI / n + M_PI / 2;
      angles[1] = angle;
      angles[2] = -M_PI / 2 - cl_idx21 * 2 * M_PI / n;
      
      // convert euler to rotation
      ang2orth(r, angles[0], angles[1], angles[2]);
      // add rotation matrix contribution to mean
      for(kk=0; kk<9;kk++){
        rotations[PAIR_IDX(n,i,j)*9 + kk] += r[kk];
      } /* kk */
    } /* if */   
  } /* k */
    

  // divide  (todo, better handle 0/0?)
  for(kk=0; kk<9;kk++){
    rotations[PAIR_IDX(n,i,j)*9 + kk] /= cnt;
  } /* kk */
    
} /* kernel */
