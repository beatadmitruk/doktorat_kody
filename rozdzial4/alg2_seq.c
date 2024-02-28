#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <openacc.h>

void seq(int n,double t1,double t2,double t3,double *b){

    double *v=malloc(sizeof (double)*(n-1));
    double b0=b[0];

    for(int i=0;i<n-1;i++)
        v[i]=b[i+1];

    v[n-2]/=t1;
    v[n-3]=(v[n-3]-t2*v[n-2])/t1;
    for(int i=n-4;i>=0;i--)
        v[i]=(v[i]-t2*v[i+1]-t3*v[i+2])/t1;


    b[n-2]=t2/t1;
    b[n-3]=(t3-t2*b[n-2])/t1;
    for(int i=n-4;i>=0;i--)
        b[i]=(-t2*b[i+1]-t3*b[i+2])/t1;


    b[n-1]=(t2*v[0]+t3*v[1]-b0)/(t2*b[0]+t3*b[1]);
    for(int i=0;i<n-1;i++)
        b[i]=v[i]-b[n-1]*b[i];


    free(v);

}


float fx(int i,float h, float p, float q){
    return p*cos(i*h)+(q-1)*sin(i*h);

}

int main(int argc, char **argv){
    int n=atoi(argv[1]);


    double *b=malloc(sizeof (double)*n);

    double t;



    double l1=-10, l2=11,l3=-1;

  b[0]=l2;
  for (int i=1;i<n;i++){
    b[i]=0;
  }


    t=omp_get_wtime();
    seq(n,l1,l2,l3,b);
    t=omp_get_wtime()-t;


    printf("%.6lf",t);



    free(b);


    return 0;
}
