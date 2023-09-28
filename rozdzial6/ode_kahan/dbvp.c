#include <stdio.h>
#include <math.h>
#include "omp.h"
#include <stdlib.h>
#include "immintrin.h"


void s_bvp_set(double *u, int N)
{
  int k;
  double h=1.0/((double)N);
  double h2=h*h;

  double pi=4.0*(double)atan(1.0);
  double pip=2.0*(double)atan(1.0);
  double pi2p4=pi*pi*0.25;

  for(k=0;k<N;k++){

    double x2=(h*(double)k)*(h*(double)k);

    u[k] = h2*20000.0*exp(-100*x2)*(1-200*x2);


  }

  u[0]*=0.5;



}





void s_bvp_solve(double *u, int N)
{
  int k;
//  double s=0;
  double temp;
  double s=u[0];
  double e=0;
  double y;
  for(k=1;k<N;k++){
    u[k] += u[k-1];
/*
    temp=s;
    y=u[k]+e;
    u[k]=s=temp+y;
    e=(temp-s)+y;
*/
  }

  s=u[N-1];
  e=0;
  for(k=N-2;k>=0;k--){
    u[k] += u[k+1];
/*
    temp=s;
    y=u[k]+e;
    u[k]=s=temp+y;
    e=(temp-s)+y;
*/
  }



}


void c_bvp_solve(double *u, int N)
{
  int k;
//  double s=0;
  double temp;
  double s=u[0];
  double e=0;
  double y;
  for(k=1;k<N;k++){
//    u[k] += u[k-1];
    temp=s;
    y=u[k]+e;
    u[k]=s=temp+y;
    e=(temp-s)+y;
  }

  s=u[N-1];
  e=0;
  for(k=N-2;k>=0;k--){
//    u[k] += u[k+1];
    temp=s;
    y=u[k]+e;
    u[k]=s=temp+y;
    e=(temp-s)+y;
  }



}



void v_bvp_set(double *u, int s, int r){

  int j,k,m;
  double h=1.0/((double)(s*r));
  double h2=h*h;

  double pi=4.0*(double)atan(1.0);
  double pip=2.0*(double)atan(1.0);
  double pi2p4=pi*pi*0.25;


  for(j=0;j<r;j++) {

    m=j*s;

    for(k=0;k<s;k++){

    double x2=(h*(double)(m+k))*(h*(double)(m+k));

    u[m+k] = h2*20000.0*exp(-100*x2)*(1-200*x2);

    }

  }
  u[0]*=0.5;

}


void v_bvp_solve(double *u, int s, int r){

  int j,k,m;


  // 1A

  for(j=0;j<r;j++) {

    m=j*s;

    for(k=1;k<s;k++){
      u[m+k]+=u[m+k-1];
    }

  }

 // 1B

  for (k=1;k<r;k++) {
     u[(k+1)*s-1]+=u[k*s-1];
  }


 // 1C

  for(j=1;j<r;j++) {
    m=j*s;
    double a=u[m-1];
    for (k=0;k<s-1;k++) {
       u[m+k]+=a;
    }
  }

    // 2A

  for(j=0;j<r;j++) {
    m=j*s;
    for(k=s-2;k>=0;k--){
      u[m+k]+=u[m+k+1];
    }

  }

 // 2B

  for (k=r-2;k>=0;k--) {
    u[k*s]+=u[k*s+s];
  }


  for(j=0;j<r-1;j++) {
    m=j*s;
    double a=u[m+s];
    for (k=1;k<s;k++) {
       u[m+k]+=a;
    }
  }



}


void xxxv_bvp_solve(double *u, int s, int r){

  int j,k,m;

  __m256i idx;
  __m512d curr,prev;

  int ind[] = {0,1,2,3,4,5,6,7};

  for(j=0;j<8;j++){
    ind[j]*=s;
  }

  idx= _mm256_load_si256((__m256i*)ind);

  // 1A


  for(j=0;j<r;j=j+8) {
    m=j*s;
    prev=_mm512_i32gather_pd(idx,&u[m],8);
    for(k=1;k<s;k++){
      curr=_mm512_i32gather_pd(idx,&u[m+k],8);
      curr=_mm512_add_pd(prev,curr);
      _mm512_store_pd(&u[m+k],curr);
      prev=curr;
    }

  }
// 1B

  for (k=1;k<r;k++) {
     u[(k+1)*s-1]+=u[k*s-1];
  }


 // 1C2A

  for(j=0;j<r;j++) {

    m=j*s;
    double a=u[m-1];
    __m512d tmp=_mm512_set1_pd(a);
    for (k=0;k<s-1;k=k+8) {
       curr=_mm512_load_pd(&u[m+k]);
       curr=_mm512_add_pd(curr,tmp);
       _mm512_store_pd(&u[m+k],curr);
    }

    // gora

    for(k=s-2;k>=0;k--){
      u[m+k]+=u[m+k+1];
    }

  }

 // 2B

  for (k=r-2;k>=0;k--) {
    u[k*s]+=u[k*s+s];
  }


  for(j=1;j<r;j++) {
    m=j*s;
    double a=u[m+s];
    for (k=1;k<s;k++) {
       u[m+k]+=a;
    }
  }



}



void row_bvp_set(double *u, int s, int r){

  int k,m;
  double h=1.0/((double)(s*r));
  double h2=h*h;

  double pi=4.0*(double)atan(1.0);
  double pip=2.0*(double)atan(1.0);
  double pi2p4=pi*pi*0.25;


  for(m=0;m<r;m++){
    for(k=0;k<s;k++){
      double x2=(h*(double)(m*s+k))*(h*(double)(m*s+k));
      u[m+k*r] = h2*20000.0*exp(-100*x2)*(1-200*x2);
    }
  }
  u[0]*=0.5;
}



void v2_bvp_solve(double *u, int s, int r){

  int j,k,m;


  // 1A

#pragma omp parallel private(j,k,m)
{

  for(k=1;k<s;k++){
    int m1=(k-1)*r;
    int m2=k*r;
    #pragma omp for
    for(j=0;j<r;j++){
      u[m2+j]+=u[m1+j];
    }

  }



 // 1B

#pragma omp single
{
  m=(s-1)*r;
  for (j=1;j<r;j++) {
     u[m+j]+=u[m+j-1];
  }
}


  // 1C
 #pragma omp for
 for(k=0;k<s-1;k++){
  m=k*r;
  for (j=1;j<r;j++) {
     u[m+j]+=u[(s-1)*r+j-1];
  }
 }

 // 2A

  for(k=s-2;k>=0;k--){
    int m1=k*r;
    int m2=(k+1)*r;
    #pragma omp for
    for(j=0;j<r;j++){
      u[m1+j]+=u[m2+j];
    }

  }


 // 2B


#pragma omp single
{
  for (k=r-2;k>=0;k--) {
    u[k]+=u[k+1];
  }
}

 // 2C

 #pragma omp for
 for(k=1;k<s;k++){
  m=k*r;
  for (j=0;j<r-1;j++) {
     u[m+j]+=u[j+1];
  }
 }

}

}

void v3_bvp_solve(double *u, int s, int r){

  int j,k,m;


  // 1A

  for(k=1;k<s;k++){
    int m1=(k-1)*r;
    int m2=k*r;
    for(j=0;j<r;j=j+8){
//      u[m2+j]+=u[m1+j];
      __m512d vp=_mm512_load_pd(&u[m1+j]);
      __m512d vc=_mm512_load_pd(&u[m2+j]);
      vc=_mm512_add_pd(vc,vp);
      _mm512_store_pd(&u[m2+j],vc);

    }
  }



 // 1B

  m=(s-1)*r;
  for (j=1;j<r;j++) {
     u[m+j]+=u[m+j-1];
  }


  // 1C

 for(k=0;k<s-1;k++){
  m=k*r;
  for (j=1;j<r;j++) {
     u[m+j]+=u[(s-1)*r+j-1];
  }
 }

 // 2A

  for(k=s-2;k>=0;k--){
    int m1=k*r;
    int m2=(k+1)*r;
    for(j=0;j<r;j++){
      u[m1+j]+=u[m2+j];
    }

  }


 // 2B


  for (k=r-2;k>=0;k--) {
    u[k]+=u[k+1];
  }

 // 2C

 for(k=1;k<s;k++){
  m=k*r;
  for (j=0;j<r-1;j++) {
     u[m+j]+=u[j+1];
  }
 }


}
// main routine that executes on the host
int main(int argc,char **argv)
{
  double *u_h, *u_d, *f_h, *v_temp;  // Pointer to host & device arrays

  double *vd_tmp;

  double tp0,tp1,ts0,ts1;

  double times,timep,xtimes,xtimep;


  int N,r,s;   // Number of elements in arrays

  //printf("s,r,bsize =");
  N=atoi(argv[1]);
  r=atoi(argv[2]);
  s=N/r;
  //scanf("%d",&s);
  //scanf("%d",&r);

  //N=r*s;

  size_t size = N * sizeof(double);

  u_h = (double *)malloc(size);        // Allocate array on host
  f_h = (double *)malloc(size);


//  cudaMemcpy(u_d, u_h, size, cudaMemcpyHostToDevice);

  // Do calculation on device:

  timep=1.0e10;



//  v_bvp_set(u_h, s,r);
   s_bvp_set(u_h, N);
   ts0 = omp_get_wtime();
   s_bvp_solve(u_h, N);
//   v_bvp_solve(u_h, s,r);
   ts1 = omp_get_wtime();
   times=ts1-ts0;



  s_bvp_set(f_h, N);
  row_bvp_set(f_h, s,r);
  tp0 = omp_get_wtime();
//  s_bvp_solve(f_h, N);
  v3_bvp_solve(f_h, s,r);
  tp1 = omp_get_wtime();
  timep=tp1-tp0;


  double h=1.0/((double)N);
  double pi=4.0*atan(1.0);
  double pip=2.0*atan(1.0);

  double sums=0.0;
  double sump=0.0;
  double sumw=0.0;

  // Print results
  for (int i=0; i<N; i++) {

     double x2=(h*(double)i)*(h*(double)i);
     double y=100*exp(-100*x2)-100*exp(-100.0);

     int ik=i/s;
     int iw=i%s;

     int j=iw*r+ik;

     double difs=fabs(u_h[i]-y);
     double difp=fabs(f_h[j]-y);
     double maxy=fabs(y);

     if(difs>sums) sums=difs;
     if(difp>sump) sump=difp;
     if(maxy>sumw) sumw=y;


//     sums+=(f_h[i]-y)*(f_h[i]-y);
//     sump+=(u_h[j]-y)*(u_h[j]-y);
//     sump+=(u_h[i]-y)*(u_h[i]-y);

//     sumw+=y*y;

//     printf("%d %f %f %f\n", i, u_h[j], f_h[i], cos(pip*h*(double)i));

  }

//  printf("ord=%le comp=%le \n", sqrt(sump)/sqrt(sumw), sqrt(sums)/sqrt(sumw));

   printf("%e %e ", (sums)/(sumw), (sump)/(sumw));


  printf("%e ",times);
  printf("%e ",timep);
  printf("%e",times/timep);


}





