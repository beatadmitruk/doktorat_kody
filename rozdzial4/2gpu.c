#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<stdlib.h>
#include<openacc.h>

#define BSIZE 128
#define VL 16
#define NW BSIZE/VL


void tridiagonal(int n,int r, int s, double t1, double t2, double t3,
                            double *b, double *x, double normB, double *wsp){



  r=r/2;

  n=n/2;


  int ndevs=acc_get_num_devices(acc_device_nvidia);


#pragma omp parallel num_threads(ndevs)  if(ndevs>1)
  {
    double norm=0;



    int dn=omp_get_thread_num();
#pragma acc set device_num(dn)

    int m = n/ndevs;  // rozmiar danych z tablicy x przetwarzany przez gpu
    int k1 = dn*m;    // indeks poczatkowego elementu  przetwarzanego przez gpu (wlacznie)
    int k2 = k1+m;    // indeks koncowego elementu  przetwarzanego przez gpu (wylacznie)

    double *u;



#pragma acc enter data create(wsp[0:3])

    u=(double*)acc_malloc(sizeof(double)*n);


    double *e0;
    e0=(double*)acc_malloc(sizeof(double)*s);
    double *es;
    es=(double*)acc_malloc(sizeof(double)*s);



    const double r1=(t2>0)?(t2-sqrt(t2*t2-4*t1*t3))/(2*t3):(t2+sqrt(t2*t2-4*t1*t3))/(2*t3);
    const double r2=t2-t3*r1;
    double tmp=-t3/r2;
    double tmp2,e0s,es0;


    // GPU: tworzymy robocze tablice



    // GPU: konwersja prawego bloku z CWF na RWF


#pragma acc parallel  present(x,b) vector_length(BSIZE)
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
              xc[i%VL][l+i/VL]= b[(j+l+i/VL)*s+k+i%VL];
        }

#pragma acc loop  gang vector
        for(int j=0;j<r;j++){
#pragma acc loop seq
          for(int i=0;i<VL;i++)
            x[(k+i)*r+j]=xc[i][j%BSIZE];
        }
      }
    }




#pragma acc parallel present(x)
    {
      for(int i=1;i<s;i++){
#pragma acc loop  independent
        for(int j=0;j<r;j++)
          x[i*r+j]-=r1*x[(i-1)*r+j];
      }
    }




#pragma acc parallel num_gangs(1) deviceptr(e0) deviceptr(u)
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




    if (dn==0){

#pragma acc parallel num_gangs(1) present(x,wsp)
      {
        for(int j=1;j<r;j++){
          x[n-r+j]-=r1*x[n-r+j-1]*e0s;
        }

        wsp[0]=x[n-1];

      }





#pragma acc parallel num_gangs(1) present(wsp) deviceptr(u)
      {
        for(int j=1;j<r;j++){
          u[n-r+j]=-r1*u[n-r+j-1]*e0s;
        }
        wsp[1]=u[n-1];
      }



    }

#pragma acc update self(wsp[0:2])


    // synchronizacja - przeslanie na hosta

#pragma omp barrier



    if(dn==1){

#pragma acc update device(wsp[0:2])

#pragma acc parallel num_gangs(1) present(x,wsp[0:2])
      {
        x[n-r]-=r1*wsp[0]*e0s;
        for(int j=1;j<r;j++){
          x[n-r+j]-=r1*x[n-r+j-1]*e0s;
        }


      }





#pragma acc parallel num_gangs(1) present(wsp[0:2]) deviceptr(u)
      {
        u[n-r]-=r1*wsp[1]*e0s;
        for(int j=1;j<r;j++){
          u[n-r+j]=-r1*u[n-r+j-1]*e0s;
        }
      }

      //calosc
#pragma acc parallel deviceptr(e0) present(x) deviceptr(u)
      {
#pragma acc loop independent
        for(int i=0;i<s-1;i++){
#pragma acc loop independent
          for(int j=0;j<r;j++){
            u[i*r+j]=-r1*( j==0 ? wsp[1] : u[(s-1)*r+j-1])*e0[i];
            x[i*r+j]-=r1*( j==0 ? wsp[0] : x[(s-1)*r+j-1])*e0[i];
          }
        }
      }


    }


    if(dn==0){
#pragma acc parallel deviceptr(e0,u) present(x)
      {
#pragma acc loop independent
        for(int i=0;i<s-1;i++){
#pragma acc loop independent
          for(int j=1;j<r;j++){
            u[i*r+j]=-r1*u[(s-1)*r+j-1]*e0[i];
            x[i*r+j]-=r1*x[(s-1)*r+j-1]*e0[i];
          }
        }
      }

    }


#pragma acc parallel present(x) deviceptr(u)
    {
#pragma acc loop independent
      for(int j=0;j<r;j++){
        x[n-r+j]/=r2;
        u[n-r+j]/=r2;
      }
    }






#pragma acc parallel  present(x)
    {
      for(int i=s-2;i>=0;i--){
#pragma acc loop  independent
        for(int j=0;j<r;j++){
          x[i*r+j]=(x[i*r+j]-t3*x[(i+1)*r+j])/r2;
        }
      }
    }

#pragma acc parallel  deviceptr(u)
    {
      for(int i=s-2;i>=0;i--){
#pragma acc loop  independent
        for(int j=0;j<r;j++){
          u[i*r+j]=(u[i*r+j]-t3*u[(i+1)*r+j])/r2;
        }
      }
    }




    if(dn==1){

#pragma acc parallel num_gangs(1) present(x,wsp) deviceptr(u)
      {
        for(int j=r-2;j>=0;j--){
          x[j]-=t3*x[j+1]*es0;
          u[j]-=t3*u[j+1]*es0;
        }
        wsp[0]=x[0];
        wsp[1]=u[0];


      }

#pragma acc update self(wsp[0:2])

    }



#pragma omp barrier



    if(dn==0){


#pragma acc update device(wsp[0:2])
#pragma acc parallel num_gangs(1) present(x,wsp) deviceptr(u)
      {
        x[r-1]-=t3*wsp[0]*es0;
        u[r-1]-=t3*wsp[1]*es0;
        for(int j=r-2;j>=0;j--){
          x[j]-=t3*x[j+1]*es0;
          u[j]-=t3*u[j+1]*es0;
        }

        x[0]=x[0]/(1+t3*r1*u[0]);
        wsp[2]=t3*r1*x[0];


      }
#pragma acc update self(wsp[2:1])


    }




#pragma omp barrier

#pragma acc update device(wsp[2:1])


    if(dn==1){

#pragma acc parallel deviceptr(es,u) present(x)
      {
#pragma acc loop independent
        for(int i=1;i<s;i++ ){
#pragma acc loop independent
          for(int j=r-2;j>=0;j--){
            x[i*r+j]-=x[j+1]*t3*es[i];
            u[i*r+j]-=u[j+1]*t3*es[i];
          }
        }
      }


    }else{

#pragma acc parallel deviceptr(es,u) present(x)
      {
#pragma acc loop independent
        for(int i=1;i<s;i++ ){
#pragma acc loop independent
          for(int j=r-1;j>=0;j--){
            x[i*r+j]-= (j== r-1 ? wsp [0]: x[j+1])*t3*es[i];
            u[i*r+j]-= (j== r-1 ? wsp [0]: u[j+1])*t3*es[i];
          }
        }
      }


    }



    int ii = (dn==0 ? 1: 0);

#pragma acc parallel present(x) deviceptr(u)
    {
#pragma acc loop  independent
      for(int i= ii;i<n;i++){
        x[i]=x[i]-wsp[2]*u[i];
      }
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



#pragma acc exit data delete(wsp)

    acc_free(e0);
    acc_free(es);
    acc_free(u);

  }

}



float fx(int i,float h, float p, float q){
  return p*cos(h*i)+(q-1)*sin(h*i);
}



int main(int argc, char **argv){
  int n=atoi(argv[1]);
  int r=atoi(argv[2]);




  int s=n/r;//
  double *x;
  x=(double*)malloc(sizeof(double)*n);

  double t,t1;
  double *b;
  b=(double*)malloc(sizeof(double)*n);

  double dh=M_PI/(n+1);
  double dh2=dh*dh;
  float h2=(float)dh2;
  float h=(float)dh;
  float A=0,B=0;
  double normB=0;
  float p=100, q=pow(10,12);
  float l1=1+p*dh/2,l2=-2-dh2*q,l3=1-p*dh/2;
  if(fabs(l1)+fabs(l3)>=fabs(l2)){
    printf("data error\n");
    return 1;
  }





  int ndevs=acc_get_num_devices(acc_device_nvidia);

#pragma omp parallel num_threads(ndevs) reduction(+:normB) if(ndevs>1)
  {
    double norm=0;

    int dn=omp_get_thread_num();
#pragma acc set device_num(dn)
    acc_init(acc_device_nvidia);

    int m = n/ndevs;  // rozmiar danych z tablicy x przetwarzany przez gpu
    int k1 = dn*m;    // indeks poczatkowego elementu  przetwarzanego przez gpu (wlacznie)
    int k2 = k1+m;    // indeks koncowego elementu  przetwarzanego przez gpu (wylacznie)



#pragma omp barrier

#pragma acc enter data create(x[0:m],b[0:m])

    int i1, i2;

    i1 = dn==0 ? 1 : n/2;
    i2 = dn==0 ? n/2-1 : n-2;



#pragma acc parallel loop independent present(x[0:m]) reduction(+:norm)
    for (int i=i1;i<=i2;i++){
      x[i-m*dn]=h*h*fx(i+1,h,p,q);
      norm+=x[i-m*dn]*x[i-m*dn];
    }

    if(dn==0){
#pragma acc parallel num_gangs(1) present(x[0:m]) reduction(+:norm)
      {
        x[0]=fx(1,h,p,q)*h2-A*(1+p*h/2);
        norm+=x[0]*x[0];
      }
    }else{
#pragma acc parallel num_gangs(1) present(x[0:m]) reduction(+:norm)
      {
        x[m-1]=fx(n,h,p,q)*h2-B*(1-p*h/2);
        norm+=x[m-1]*x[m-1];
      }

    }

    normB=norm;

  }

  normB=sqrt(normB);


  double wsp[3];


  t=omp_get_wtime();
  tridiagonal(n,r,s,l1,l2,l3,x,b,normB,wsp);
  t=omp_get_wtime()-t;



  printf("%.6lf",t);


#pragma omp parallel num_threads(ndevs) if(ndevs>1)
  {
    int dn=omp_get_thread_num();
#pragma acc set device_num(dn)

    int m = n/ndevs;  // rozmiar danych z tablicy x przetwarzany przez gpu
    int k1 = dn*m;    // indeks poczatkowego elementu  przetwarzanego przez gpu (wlacznie)
    int k2 = k1+m;    // indeks koncowego elementu  przetwarzanego przez gpu (wylacznie)


#pragma acc exit data copyout(x[0:m],b[0:m])
  }
  free(x);
  free(b);
  return 0;
}
