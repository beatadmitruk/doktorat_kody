#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include "lapacke.h"




int main(int argc, char **argv){


  int n=atoi(argv[1]);
  double *d1, *d2, *d3, *x;

  d1 =malloc(sizeof(double)*(n-1));
  d2 = malloc(sizeof(double)*n);
  d3 = malloc(sizeof(double)*(n-1));

  x = malloc(sizeof(double)*n);


  int i;

    double l1=-10, l2=11,l3=-1;

    if(l2*l2-4*l1*l3<0){
        printf("%lf\n",l2*l2-4*l1*l3);
        return -1;
    }

  x[0]=l2;
  for (int i=1;i<n;i++){
    x[i]=0;
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




    int info;
  int bdim=1;
  double t0=omp_get_wtime();
  // calculate solution using the DGTSV subroutine
  LAPACK_dgtsv(&n,&bdim,d1,d2,d3,x,&n,&info);
  printf("%lf\n",omp_get_wtime()-t0);


  if (info!=0){

    return -1;
  }


  free(d1);
  free(d2);
  free(d3);

  free(x);
  return 0;
}
