#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<stdlib.h>
#include<openacc.h>

#define BSIZE 128
#define VL 16
#define NW BSIZE/VL






void tridiagonal(int n,int r, int s, double t1, double t2, double t3, double *b){
    const double r1=(t2>0)?(t2-sqrt(t2*t2-4*t1*t3))/(2*t3):(t2+sqrt(t2*t2-4*t1*t3))/(2*t3);
    const double r2=t2-t3*r1;
    double *u;
    u=(double*)acc_malloc(sizeof(double)*n);
    double *e0;
    e0=(double*)acc_malloc(sizeof(double)*s);
    double *es;
    es=(double*)acc_malloc(sizeof(double)*s);
    double tmp=-t3/r2;
    double tmp2,e0s,es0;

#pragma acc parallel present(b)
    {
        for(int i=1;i<s;i++){
#pragma acc loop  independent
            for(int j=0;j<r;j++)
                b[i*r+j]-=r1*b[(i-1)*r+j];
        }
    }


#pragma acc parallel num_gangs(1) deviceptr(u,e0)
    {
        u[0]=e0[0]=1;
        for(int i=1;i<s;i++){
            u[i*r]=e0[i]=-r1*e0[i-1];
        }
        e0s=e0[s-1];
    }

#pragma acc parallel num_gangs(1) deviceptr(es)
    {
        es[s-1]=1/r2;
        for(int i=1;i<s;i++){
            es[s-1-i]=tmp*es[s-i];

        }
        es0=es[0];
    }



#pragma acc parallel num_gangs(1) present(b)
    {
        for(int j=1;j<r;j++){
            b[n-r+j]-=r1*b[n-r+j-1]*e0s;
        }
    }
#pragma acc parallel num_gangs(1) deviceptr(u)
    {
        for(int j=1;j<r;j++){
            u[n-r+j]=-r1*u[n-r+j-1]*e0s;
        }
    }


#pragma acc parallel deviceptr(u,e0)
    {
#pragma acc loop independent
        for(int i=0;i<s-1;i++){
#pragma acc loop independent
            for(int j=1;j<r;j++){
                u[i*r+j]=-r1*u[(s-1)*r+j-1]*e0[i];
            }
        }
    }


#pragma acc parallel deviceptr(e0) present(b)
    {
#pragma acc loop independent
        for(int i=0;i<s-1;i++){
#pragma acc loop independent
            for(int j=1;j<r;j++){
                b[i*r+j]-=r1*b[(s-1)*r+j-1]*e0[i];
            }
        }
    }



#pragma acc parallel deviceptr(u)
    {
#pragma acc loop independent
        for(int j=0;j<r;j++){
            u[n-r+j]/=r2;
        }
    }

#pragma acc parallel present(b)
    {
#pragma acc loop independent
        for(int j=0;j<r;j++){
            b[n-r+j]/=r2;
        }
    }




#pragma acc parallel deviceptr(u) present(b)
    {
        for(int i=s-2;i>=0;i--){
#pragma acc loop  independent
            for(int j=r-1;j>=0;j--){
                b[i*r+j]=(b[i*r+j]-t3*b[(i+1)*r+j])/r2;
                u[i*r+j]=(u[i*r+j]-t3*u[(i+1)*r+j])/r2;
            }
        }
    }


#pragma acc parallel num_gangs(1) present(b)
    {
        for(int j=r-2;j>=0;j--){
            b[j]-=t3*b[j+1]*es0;
        }
    }
#pragma acc parallel num_gangs(1) deviceptr(u)
    {
        for(int j=r-2;j>=0;j--){
            u[j]-=t3*u[j+1]*es0;
        }
    }

#pragma acc parallel num_gangs(1) deviceptr(u) present(b)
    {

        b[0]=b[0]/(1+t3*r1*u[0]);
        tmp2=t3*r1*b[0];
    }

#pragma acc parallel deviceptr(es) present(b)
    {
#pragma acc loop independent
        for(int i=1;i<s;i++ ){
#pragma acc loop independent
            for(int j=r-2;j>=0;j--){
                b[i*r+j]-=b[j+1]*t3*es[i];
            }
        }
    }


#pragma acc parallel deviceptr(u,es)
    {
#pragma acc loop independent
        for(int i=1;i<s;i++ ){
#pragma acc loop independent
            for(int j=r-2;j>=0;j--){
                u[i*r+j]-=u[j+1]*t3*es[i];
            }
        }
    }

    acc_free(u);
    acc_free(e0);
    acc_free(es);

#pragma acc parallel present(b) deviceptr(u)
    {
#pragma acc loop  independent
        for(int i=1;i<n;i++){
            b[i]-=tmp2*u[i];
        }
    }


}



float fx(int i,float h, float p, float q){
    return p*cos(h*i)+(q-1)*sin(h*i);
}



int main(int argc, char **argv){
    int n=atoi(argv[1]);
    int r=atoi(argv[2]);
    int s=n/r;
    double *x;
    x=(double*)malloc(sizeof(double)*n);

    double t;
    double *b;
    b=(double*)malloc(sizeof(double)*n);
double l1=-10, l2=11, l3=-1;

  b[0]=x[0]=l2;
  for (int i=1;i<n;i++){
    x[i]=b[i]=0;
  }

    if(fabs(l1)+fabs(l3)>fabs(l2)){
        printf("data error\n");
        return 1;
    }
    acc_init(acc_device_nvidia);
#pragma acc data copyin(x[0:n]) copyout(b[0:n])
{


  t=omp_get_wtime();

  #pragma acc parallel  present(b,x) vector_length(BSIZE)
  {
    float xc[VL][BSIZE];
    #pragma acc cache(xc)

    for(int k=0;k<s;k+=VL){
      #pragma acc loop gang
      for(int j=0;j<r;j+=BSIZE){
          #pragma acc loop seq
          for(int l=0;l<BSIZE;l+=NW)
          #pragma acc loop vector
          for(int i=0;i<BSIZE;i++)
             xc[i%VL][l+i/VL]= x[(j+l+i/VL)*s+k+i%VL];
      }

      #pragma acc loop  gang vector
      for(int j=0;j<r;j++){
         #pragma acc loop seq
         for(int i=0;i<VL;i++)
            b[(k+i)*r+j]=xc[i][j%BSIZE];
      }
  }
}





#pragma acc data present(b)
{
    tridiagonal(n,r,s,l1,l2,l3,b);
}

  #pragma acc parallel  present(b,x) vector_length(BSIZE)
  {
  float xc[VL][BSIZE];
  #pragma acc cache(xc)

   for(int k=0;k<r;k+=VL){
   #pragma acc loop gang
   for(int j=0;j<s;j+=BSIZE){
       #pragma acc loop seq
       for(int l=0;l<BSIZE;l+=NW)
       #pragma acc loop vector
       for(int i=0;i<BSIZE;i++)
          xc[i%VL][l+i/VL]= x[(j+l+i/VL)*r+k+i%VL];
   }


     #pragma acc loop  gang vector
     for(int j=0;j<s;j++){
       #pragma acc loop seq
       for(int i=0;i<VL;i++)
          b[(k+i)*s+j]=xc[i][j%BSIZE];
   }




  }

  }


    t=omp_get_wtime()-t;

    }

    printf("%.6lf",t);

    free(x);
    free(b);

    return 0;
}
