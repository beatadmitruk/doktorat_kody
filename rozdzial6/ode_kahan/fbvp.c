#include <stdio.h>
#include <math.h>
#include "omp.h"

void s_bvp_set(float *u, int N)
{
  int k;
  float h=1.0/((float)N);
  float h2=h*h;

  float pi=4.0*(float)atan(1.0);
  float pip=2.0*(float)atan(1.0);
  float pi2p4=pi*pi*0.25;

  for(k=0;k<N;k++){

    float x2=(h*(float)k)*(h*(float)k);

    u[k] = h2*20000.0*exp(-100*x2)*(1-200*x2);


  }

  u[0]*=0.5;



}


void s_bvp_solve(float *u, int N)
{
  int k;
//  float s=0;
  float temp;
  float s=u[0];
  float e=0;
  float y;
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


void c_bvp_solve(float *u, int N)
{
  int k;
//  float s=0;
  float temp;
  float s=u[0];
  float e=0;
  float y;
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



void v_bvp_set(float *u, int s, int r){

  int j,k,m;
  float h=1.0/((float)(s*r));
  float h2=h*h;

  float pi=4.0*(float)atan(1.0);
  float pip=2.0*(float)atan(1.0);
  float pi2p4=pi*pi*0.25;


  for(j=0;j<r;j++) {

    m=j*s;

    for(k=0;k<s;k++){

    float x2=(h*(float)(m+k))*(h*(float)(m+k));

    u[m+k] = h2*20000.0*exp(-100*x2)*(1-200*x2);

    }

  }

  u[0]*=0.5;

}



void v_bvp_solve(float *u, int s, int r){

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


 // 1C2A

  for(j=0;j<r;j++) {

    m=j*s;
    float a=u[m-1];
    for (k=0;k<s-1;k++) {
       u[m+k]+=a;
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

  for(j=0;j<r;j++) {
    m=j*s;
    float a=u[m+s];
    for (k=1;k<s;k++) {
       u[m+k]+=a;
    }
  }



}





// main routine that executes on the host
int main(int argc,char **argv)
{
  float *u_h, *u_d, *f_h, *v_temp;  // Pointer to host & device arrays

  float *vd_tmp;

  double tp0,tp1,ts0,ts1;

  float times,timep,xtimes,xtimep;


  int N,r,s;   // Number of elements in arrays
  N=atoi(argv[1]);
  r=atoi(argv[2]);
  s=N/r;
  //printf("s,r,bsize =");
  //scanf("%d",&s);
  //scanf("%d",&r);

  //N=r*s;

  size_t size = N * sizeof(float);

  u_h = (float *)malloc(size);        // Allocate array on host
  f_h = (float *)malloc(size);


//  cudaMemcpy(u_d, u_h, size, cudaMemcpyHostToDevice);

  // Do calculation on device:

  timep=1.0e10;



//  v_bvp_set(u_h, s,r);
   s_bvp_set(u_h, N);
   ts0 = omp_get_wtime();
   s_bvp_solve(u_h, N);
   ts1 = omp_get_wtime();
   times=ts1-ts0;



  s_bvp_set(f_h, N);
  tp0 = omp_get_wtime();
  c_bvp_solve(f_h, N);
  tp1 = omp_get_wtime();
  timep=tp1-tp0;


  float h=1.0/((float)N);
  float pi=4.0*atan(1.0);
  float pip=2.0*atan(1.0);

  float sums=0.0;
  float sump=0.0;
  float sumw=0.0;

  // Print results
  for (int i=0; i<N; i++) {

     float x2=(h*(float)i)*(h*(float)i);
     float y=100*exp(-100*x2)-100*exp(-100.0);

     int ik=i/s;
     int iw=i%s;

     int j=iw*r+ik;

     float difs=fabs(u_h[i]-y);
     float difp=fabs(f_h[i]-y);
     float maxy=fabs(y);

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

   //printf("ord=%le comp=%le \n", (sums)/(sumw), (sump)/(sumw));


  //printf("Czas ord  . = %le\n",times);
  //printf("Czas Kahan. = %le\n",timep);
  //printf("Speedup = %lf\n",times/timep);
   printf("%e %e ", (sums)/(sumw), (sump)/(sumw));


  printf("%e ",times);
  printf("%e ",timep);
  printf("%e",times/timep);


}


