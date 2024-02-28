#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>

#define BSIZE 128
#define VL 16
#define NC BSIZE/VL


void subParallel(int n,int r,double t1,double t2,double t3,double *b){
  int s=(n-1)/r;
  double tmp1, tmp2, b0=b[0], t2t1=t2/t1, t3t1=t3/t1;
  double *y=acc_malloc(sizeof (double)*(s+1));
  double *v=acc_malloc(sizeof (double)*(n-1));


#pragma acc parallel num_gangs(1) deviceptr(y)
  {
    y[s]=0;
    y[s-1]=1/t1;
    y[s-2]=-t2t1/t1;
    for(int i=s-3;i>=0;i--)
      y[i]=(-t2*y[i+1]-t3*y[i+2])/t1;
  }

#pragma acc parallel present(b) deviceptr(v)
  {
#pragma acc loop independent
    for(int i=0;i<n-1;i++)
      v[i]=b[i+1];
  }



#pragma acc parallel deviceptr(v)
{


    double vc[VL+2][BSIZE];
    #pragma acc cache(vc)


     #pragma acc loop gang
     for(int j=0;j<r;j+=BSIZE){

    #pragma avv loop vector
    for(int k=0;k<BSIZE;k++){
      vc[VL][k]=vc[VL+1][k]=0.0;
    }



       for(int k=s-VL;k>=0;k-=VL){

        for(int l=0;l<BSIZE;l+=NC){
           #pragma acc loop vector
           for(int i=0;i<BSIZE;i++){
              vc[i%VL][l+i/VL]=v[(j+l+i/VL)*s+k+i%VL];
           }
        }



        #pragma acc loop vector
        for(int j=0;j<BSIZE;j++){

     for(int i=VL-1;i>=0;i--){


          vc[i][j]=(vc[i][j]-t2*vc[i+1][j]-t3*vc[i+2][j])/t1;
        }

     }


        for(int l=0;l<BSIZE;l+=NC){
           #pragma acc loop vector
           for(int i=0;i<BSIZE;i++){
              v[(j+l+i/VL)*s+k+i%VL]=vc[i%VL][l+i/VL];
           }
        }


    #pragma acc loop vector
    for(int kk=0;kk<BSIZE;kk++){
      vc[VL][kk]=vc[0][kk];
      vc[VL+1][kk]=vc[1][kk];

    }




     }


   }


}

#pragma acc parallel deviceptr(v,y)
  {
    for(int j=r-2;j>=0;j--){
      tmp1=t2*v[(j+1)*s]+t3*v[(j+1)*s+1];
      tmp2=t3*v[(j+1)*s];
#pragma acc loop  independent
      for(int i=0;i<2;i++)
	v[j*s+i]-=(tmp1*y[i]+tmp2*y[i+1]);
    }
  }

#pragma acc parallel deviceptr(v,y)
  {
#pragma acc loop independent
    for(int j=r-2;j>=0;j--){
      tmp1=t2*v[(j+1)*s]+t3*v[(j+1)*s+1];
      tmp2=t3*v[(j+1)*s];
#pragma acc loop independent
      for(int i=2;i<s;i++)
	v[j*s+i]-=(tmp1*y[i]+tmp2*y[i+1]);
    }
  }





#pragma acc parallel num_gangs(1) present(b)
  {
    b[n-2]=t2t1;
    b[n-3]=(t3-t2*b[n-2])/t1;
    for(int i=n-4;i>=n-s-1;i--)
      b[i]=(-t2*b[i+1]-t3*b[i+2])/t1;
  }



#pragma acc parallel present(b) deviceptr(y)
  {

    for(int j=r-2;j>=0;j--){
      double tmp1=t2*b[(j+1)*s]+t3*b[(j+1)*s+1];
      double tmp2=t3*b[(j+1)*s];
#pragma acc loop  independent
      for(int i=0;i<2;i++)
	b[j*s+i]=-(tmp1*y[i]+tmp2*y[i+1]);
    }
  }

#pragma acc parallel present(b) deviceptr(y)
  {
#pragma acc loop independent
    for(int j=r-2;j>=0;j--){
      double tmp1=t2*b[(j+1)*s]+t3*b[(j+1)*s+1];
      double tmp2=t3*b[(j+1)*s];
#pragma acc loop  independent
      for(int i=2;i<s;i++)
	b[j*s+i]=-(tmp1*y[i]+tmp2*y[i+1]);
    }
  }
  acc_free(y);
#pragma acc parallel num_gangs(1) present(b) deviceptr(v)
  {
    b[n-1]=(t2*v[0]+t3*v[1]-b0)/(t2*b[0]+t3*b[1]);
    tmp1=b[n-1];
  }

#pragma acc parallel present(b) deviceptr(v)
  {
#pragma acc loop  independent
    for(int i=0;i<n-1;i++)
      b[i]=v[i]-tmp1*b[i];
  }


  acc_free(v);

}


float fx(int i,float h, float p, float q){
    return p*cos(i*h)+(q-1)*sin(i*h);
}




int main(int argc, char **argv){
  int n=atoi(argv[1]);
  int r=atoi(argv[2]);


  double *b=malloc(sizeof (double)*n);

  double t;



double l1=-10, l2=11, l3=-1;

  b[0]=l2;
  for (int i=1;i<n;i++){
    b[i]=0;
  }


  acc_init(acc_device_nvidia);
#pragma acc data copy(b[0:n])
  {

    t=omp_get_wtime();
    subParallel(n,r,l1,l2,l3,b);
    t=omp_get_wtime()-t;

  }
  printf("%.6lf",t);



  free(b);



  return 0;
}
