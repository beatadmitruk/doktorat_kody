#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<stdlib.h>
#include<openacc.h>



double test_format(int n, int r, double t1, double t2, double  t3, double *x,double *b,double *res, double normB){
    int s=n/r, index;
    double sum=0;
#pragma acc data copy(sum)
    {
#pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
        {
#pragma acc loop independent
            for(int i=1;i<s-1;i++){
#pragma acc loop independent private(index)
                for(int j=0;j<r;j++){
                    index=i*r+j;
                    res[index]=t1*x[index-r]+t2*x[index]+t3*x[index+r]-b[index];
                    sum+=res[index]*res[index];
                }
            }
        }

#pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
        {
#pragma acc loop independent private(index)
            for(int i=1;i<r;i++){
                index=n-r+i;
                res[i]=t1*x[index-1]+t2*x[i]+t3*x[r+i]-b[i];
                res[index-1]=t1*x[index-r-1]+t2*x[index-1]+t3*x[i]-b[index-1];
                sum+=res[i]*res[i]+res[index-1]*res[index-1];
            }

        }
#pragma acc parallel num_gangs(1) present(b,x) deviceptr(res) reduction(+:sum)
        {
            res[0]=t2*x[0]+t3*x[r]-b[0];
            res[n-1]=t1*x[n-1-r]+t2*x[n-1]-b[n-1];
            sum+=res[0]*res[0]+res[n-1]*res[n-1];
        }
    }

    return sqrt(sum)/normB;
}





double tridiagonal(int n,int r, int s, double t1, double t2, double t3, double *b, double *x, double normB){
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

#pragma acc parallel present(b,x)
    {
#pragma acc loop independent
        for(int i=0;i<n;i++)
            x[i]=b[i];

    }
//lewa strona
#pragma acc parallel present(x)
    {
        for(int i=1;i<s;i++){
#pragma acc loop  independent
            for(int j=0;j<r;j++)
                x[i*r+j]-=r1*x[(i-1)*r+j];
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
#pragma acc parallel num_gangs(1) present(x)
    {
        for(int j=1;j<r;j++){
            x[n-r+j]-=r1*x[n-r+j-1]*e0s;
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


#pragma acc parallel deviceptr(e0) present(x)
    {
#pragma acc loop independent
        for(int i=0;i<s-1;i++){
#pragma acc loop independent
            for(int j=1;j<r;j++){
                x[i*r+j]-=r1*x[(s-1)*r+j-1]*e0[i];
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

#pragma acc parallel present(x)
    {
#pragma acc loop independent
        for(int j=0;j<r;j++){
            x[n-r+j]/=r2;
        }
    }



    //r alfa = z//lewa strona
#pragma acc parallel deviceptr(u) present(x)
    {
        for(int i=s-2;i>=0;i--){
#pragma acc loop  independent
            for(int j=r-1;j>=0;j--){
                x[i*r+j]=(x[i*r+j]-t3*x[(i+1)*r+j])/r2;
                u[i*r+j]=(u[i*r+j]-t3*u[(i+1)*r+j])/r2;
            }
        }
    }




    //pierwsze skladowe
#pragma acc parallel num_gangs(1) present(x)
    {
        for(int j=r-2;j>=0;j--){
            x[j]-=t3*x[j+1]*es0;
        }
    }
#pragma acc parallel num_gangs(1) deviceptr(u)
    {
        for(int j=r-2;j>=0;j--){
            u[j]-=t3*u[j+1]*es0;
        }
    }

#pragma acc parallel num_gangs(1) deviceptr(u) present(x)
    {

        x[0]=x[0]/(1+t3*r1*u[0]);
        tmp2=t3*r1*x[0];
    }
    //calosc
#pragma acc parallel deviceptr(es) present(x)
    {
#pragma acc loop independent
        for(int i=1;i<s;i++ ){
#pragma acc loop independent
            for(int j=r-2;j>=0;j--){
                x[i*r+j]-=x[j+1]*t3*es[i];
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
#pragma acc parallel present(x) deviceptr(u)
    {
#pragma acc loop  independent
        for(int i=1;i<n;i++){
            x[i]=x[i]-tmp2*u[i];
        }
    }
    double *res;
    res=(double*)acc_malloc(sizeof(double)*n);

    int index;
    double sum=0;
#pragma acc data copy(sum)
    {
#pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
        {
#pragma acc loop independent
            for(int i=1;i<s-1;i++){
#pragma acc loop independent private(index)
                for(int j=0;j<r;j++){
                    index=i*r+j;
                    res[index]=t1*x[index-r]+t2*x[index]+t3*x[index+r]-b[index];
                    sum+=res[index]*res[index];
                }
            }
        }

#pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
        {
#pragma acc loop independent private(index)
            for(int i=1;i<r;i++){
                index=n-r+i;
                res[i]=t1*x[index-1]+t2*x[i]+t3*x[r+i]-b[i];
                res[index-1]=t1*x[index-r-1]+t2*x[index-1]+t3*x[i]-b[index-1];
                sum+=res[i]*res[i]+res[index-1]*res[index-1];
            }

        }
#pragma acc parallel num_gangs(1) present(b,x) deviceptr(res) reduction(+:sum)
        {
            res[0]=t2*x[0]+t3*x[r]-b[0];
            res[n-1]=t1*x[n-1-r]+t2*x[n-1]-b[n-1];
            sum+=res[0]*res[0]+res[n-1]*res[n-1];
        }
    }
    acc_free(res);

    return sqrt(sum)/normB;



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
    double t,t1,test;
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

#pragma omp parallel for reduction(+:normB)
    for (int i=1;i<n-1;i++){
        x[i]=h*h*fx(i+1,h,p,q);
        normB+=x[i]*x[i];
    }


    x[0]=fx(1,h,p,q)*h2-A*(1+p*h/2);
    x[n-1]=fx(n,h,p,q)*h2-B*(1-p*h/2);
    normB+=x[0]*x[0]+x[n-1]*x[n-1];

    normB=sqrt(normB);

    acc_init(acc_device_nvidia);
#pragma acc data copyout(b[0:n]) copyin(x[0:n])
{
    t1=omp_get_wtime();
#pragma acc parallel loop independent present(x,b)
    for(int i=0;i<n;i++)
        b[i]=x[i/r+(i%r)*s];

#pragma acc data present(b,x)
{
    t=omp_get_wtime();
    test=tridiagonal(n,r,s,l1,l2,l3,b,x,normB);
    t=omp_get_wtime()-t;
}
    //powrót
#pragma acc parallel loop independent present(x,b)
    for(int i=0;i<n;i++)
        b[i]=x[i/s+(i%s)*r];

    t1=omp_get_wtime()-t1;

    }

    printf("%.30lf %lf %lf",test,t,t1);

    free(x);
    free(b);
    //free(dx);
    return 0;
}
