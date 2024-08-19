
/* from i,j indices to the common index in the N-choose-2 sized array */
#define PAIR_IDX(N,I,J) ((2*N-I-1)*I/2 + J-I-1)


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
