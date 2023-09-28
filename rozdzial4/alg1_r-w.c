#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<stdlib.h>
#include<openacc.h>



// double test_format(int n, int r, double t1, double t2, double  t3, double *x,double *b,double *res, double normB){
//     int s=n/r, index;
//     double sum=0;
// #pragma acc data copy(sum)
//     {
// #pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
//         {
// #pragma acc loop independent
//             for(int i=1;i<s-1;i++){
// #pragma acc loop independent private(index)
//                 for(int j=0;j<r;j++){
//                     index=i*r+j;
//                     res[index]=t1*x[index-r]+t2*x[index]+t3*x[index+r]-b[index];
// 		    sum+=res[index]*res[index];
//                 }
//             }
//         }
//
// #pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
// 	{
// #pragma acc loop independent private(index)
//             for(int i=1;i<r;i++){
//                 index=n-r+i;
//                 res[i]=t1*x[index-1]+t2*x[i]+t3*x[r+i]-b[i];
//                 res[index-1]=t1*x[index-r-1]+t2*x[index-1]+t3*x[i]-b[index-1];
//                 sum+=res[i]*res[i]+res[index-1]*res[index-1];
//             }
//
// 	}
// #pragma acc parallel num_gangs(1) present(b,x) deviceptr(res) reduction(+:sum)
//         {
//             res[0]=t2*x[0]+t3*x[r]-b[0];
//             res[n-1]=t1*x[n-1-r]+t2*x[n-1]-b[n-1];
//     	    sum+=res[0]*res[0]+res[n-1]*res[n-1];
//         }
//     }
//
//     return sqrt(sum)/normB;
// }





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
    //lz=b
    //prawa strona
    //

// #pragma acc parallel present(b,x)
//     {
// #pragma acc loop independent
//         for(int i=0;i<n;i++)
//             x[i]=b[i];
//
//     }
//lewa strona
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


    //ostatnie skladowe
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

    //calosc
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



    //r alfa = z//lewa strona
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




    //pierwsze skladowe
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

        b[0]/=(1+t3*r1*u[0]);
        tmp2=t3*r1*b[0];
    }
    //calosc
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
//mamy v i u
#pragma acc parallel present(b) deviceptr(u)
    {
#pragma acc loop  independent
        for(int i=1;i<n;i++){
            b[i]-=tmp2*u[i];
        }
    }
//     double *res;
//     res=(double*)acc_malloc(sizeof(double)*n);
//
//     int index;
//     double sum=0;
// #pragma acc data copy(sum)
//     {
// #pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
//         {
// #pragma acc loop independent
//             for(int i=1;i<s-1;i++){
// #pragma acc loop independent private(index)
//                 for(int j=0;j<r;j++){
//                     index=i*r+j;
//                     res[index]=t1*x[index-r]+t2*x[index]+t3*x[index+r]-b[index];
//                     sum+=res[index]*res[index];
//                 }
//             }
//         }
//
// #pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
//         {
// #pragma acc loop independent private(index)
//             for(int i=1;i<r;i++){
//                 index=n-r+i;
//                 res[i]=t1*x[index-1]+t2*x[i]+t3*x[r+i]-b[i];
//                 res[index-1]=t1*x[index-r-1]+t2*x[index-1]+t3*x[i]-b[index-1];
//                 sum+=res[i]*res[i]+res[index-1]*res[index-1];
//             }
//
//         }
// #pragma acc parallel num_gangs(1) present(b,x) deviceptr(res) reduction(+:sum)
//         {
//             res[0]=t2*x[0]+t3*x[r]-b[0];
//             res[n-1]=t1*x[n-1-r]+t2*x[n-1]-b[n-1];
//             sum+=res[0]*res[0]+res[n-1]*res[n-1];
//         }
//     }
//     acc_free(res);
//
//     return sqrt(sum)/normB;



/*
#pragma acc data  present(b,x) deviceptr(u,res,e0,es)
    {
	double test_value=test_format(n,r,t1,t2,t3,x,b,res, normB);
        printf("%.30lf ",test_value);
    }

*/


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
    //double l1=2,l2=13,l3=10, eps=pow(10,-10);
    double t;
    double *b;
    b=(double*)malloc(sizeof(double)*n);

double l1=-10, l2=11, l3=-1;

  b[0]=x[0]=l2;
  for (int i=1;i<n;i++){
    x[i]=b[i]=0;
  }

    acc_init(acc_device_nvidia);
#pragma acc data copyout(b[0:n]) copyin(x[0:n])
{
    t=omp_get_wtime();
#pragma acc parallel loop independent present(x,b)
    for(int i=0;i<n;i++)
        b[i]=x[i/r+(i%r)*s];

#pragma acc data present(b,x)
{

    tridiagonal(n,r,s,l1,l2,l3,b);

}
    //powrót
#pragma acc parallel loop independent present(x,b)
    for(int i=0;i<n;i++)
        b[i]=x[i/s+(i%s)*r];

    t=omp_get_wtime()-t;

    }

    printf("%.6lf",t);

    free(x);
    free(b);
    //free(dx);
    return 0;
}
