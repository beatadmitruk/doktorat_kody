#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include "lapacke.h"


double test(int n, double t1,double t2,double t3,double *x, double *b){
    double sum=0, tmp, tmp2, norm=0;

    for(int i=1;i<n-1;i++){
        tmp=(t1*x[i-1]+t2*x[i]+t3*x[i+1]-b[i]);
        sum+=tmp*tmp;
        norm+=b[i]*b[i];
    }

    tmp =(t2*x[0]+t3*x[1]-b[0]);
    tmp2=t1*x[n-2]+t2*x[n-1]-b[n-1];
    sum+=tmp*tmp+tmp2*tmp2;
    norm+=b[0]*b[0]+b[n-1]*b[n-1];


    return sqrt(sum)/sqrt(norm);
}


int main(int argc, char **argv){


  int n=atoi(argv[1]);
  double *d1, *d2, *d3, *b,*x;

  d1 =malloc(sizeof(double)*(n-1));
  d2 = malloc(sizeof(double)*n);
  d3 = malloc(sizeof(double)*(n-1));
  b = malloc(sizeof(double)*n);
  x = malloc(sizeof(double)*n);


  int i;

    double l1=-10, l2=11,l3=-1;

    if(l2*l2-4*l1*l3<0){
        printf("%lf\n",l2*l2-4*l1*l3);
        return -1;
    }

  b[0]=x[0]=l2;
  for (int i=1;i<n;i++){
    x[i]=b[i]=0;
  }


  for (i=0;i<n-1;i++){
       d1[i]=l1;
  }
  for (i=0;i<n;i++){
      d2[i]=l2;
  }
  for (i=0;i<n-1;i++){
      d3[i]=l3;
  }


  //printf("==============================\n");

    int info;
  int bdim=1;
  double t0=omp_get_wtime();
  // calculate solution using the DGTSV subroutine
  LAPACK_dgtsv(&n,&bdim,d1,d2,d3,x,&n,&info);
  printf("%lf\n",omp_get_wtime()-t0);
  //printf("\n==============================\n");
  //printf("%e",test(n,l1,l2,l3,x,b));

  if (info!=0){
    //printf("Error: dgeev returned error code %d\n", info);
    return -1;
  }

  // output solution to stdout
//    puts("\n--- Solution ---\n");
//   for (i=n-10;i<n;i++){
//     printf("x[%d] = %e\n", i, x[i]);
//   }
//   puts(" ");

  // deallocate
  free(d1);
  free(d2);
  free(d3);
  free(b);
  free(x);
  return 0;
}
