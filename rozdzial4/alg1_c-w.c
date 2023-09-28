#include<stdio.h>
#include<math.h>
#include<mm_malloc.h>
#include<omp.h>
#include<stdlib.h>
#include<openacc.h>

void tridiagonal(int n,int r, double t1, double t2, double t3, double *b){
    const double r1=(t2>0) ? (t2-sqrt(t2*t2-4*t1*t3))/(2*t3) : (t2+sqrt(t2*t2-4*t1*t3))/(2*t3);
    const double r2=t2-t3*r1;
    int s=n/r;
    double tmp1=-t3/r2, tmp, es0, us;
    double last1, last2, last3, last4;

    double *u;
    u=(double*)acc_malloc((size_t)n*sizeof(double));
    double *es;
    es=(double*)acc_malloc((size_t)s*sizeof(double));
    double *e0;
    e0=(double*)acc_malloc((size_t)s*sizeof(double));

// #pragma acc parallel present(b,x)
//     {
// #pragma acc loop independent
//         for(int i=0;i<n;i++)
//             x[i]=b[i];
//     }

//lz=b
//lewa strona
#pragma acc parallel present(b)
{
#pragma acc loop  independent
for(int j=0;j<r;j++){
        for(int i=1;i<s;i++)
            b[j*s+i]-=r1*b[j*s+i-1];
    }
}


//prawa strona
#pragma acc parallel num_gangs(1) deviceptr(e0,u)
{
    e0[0]=u[0]=1;
    for(int i=1;i<s;i++){
        e0[i]=u[i]=-r1*u[i-1];
    }
    us=e0[s-1];
}


#pragma acc parallel num_gangs(1) deviceptr(es)
{
    es[s-1]=1./r2;
    for(int i=1;i<s;i++){
        es[s-i-1]=tmp1*es[s-i];
    }
    es0=es[0];
}




//ostatnie składowe
#pragma acc parallel num_gangs(1) present(b) //deviceptr(u)
{
    for(int j=1;j<r;j++){
       b[(j+1)*s-1]-=r1*b[j*s-1]*us;//u[s-1];
    }
}
#pragma acc parallel num_gangs(1) deviceptr(u)
{
    for(int j=1;j<r;j++){
       u[(j+1)*s-1]=-r1*u[j*s-1]*us;
    }
}



//całość
#pragma acc parallel present(b) deviceptr(e0)
{
#pragma acc loop independent
    for(int j=1;j<r;j++){

        //col=&x[j*s];
        last1=b[j*s-1]*r1;
#pragma acc loop  independent
       for(int i=0;i<s-1;i++)
            b[j*s+i]-=last1*e0[i];
    }

}

#pragma acc parallel deviceptr(u)
{
#pragma acc loop independent
    for(int j=1;j<r;j++){
        last3=u[j*s-1]*r1;
#pragma acc loop  independent
        for(int i=0;i<s-1;i++)
            u[j*s+i]=-last3*u[i];
    }
}


//r alfa = z
//lewa strona
#pragma acc parallel present(b)
{
#pragma acc loop  independent
    for(int j=r-1;j>=0;j--){
        b[j*s+s-1]/=r2;
	for(int i=s-2;i>=0;i--)
            b[j*s+i]=(b[j*s+i]-t3*b[j*s+i+1])/r2;
    }
}
//pierwsze składowe
#pragma acc parallel num_gangs(1) present(b)// deviceptr(es)
{
    for(int j=r-2;j>=0;j--){
       b[j*s]-=t3*b[(j+1)*s]*es0;
    }
}
//całość
#pragma acc parallel present(b) deviceptr(es)
{
#pragma acc loop independent
    for(int j=r-2;j>=0;j--){
        last2=t3*b[(j+1)*s];
#pragma acc loop  independent
        for(int i=1;i<s;i++)
            b[j*s+i]-=last2*es[i];
    }
}
//r beta = z
//lewa strona
#pragma acc parallel deviceptr(u)
{
#pragma acc loop  independent
    for(int j=r-1;j>=0;j--){
        u[j*s+s-1]/=r2;
        for(int i=s-2;i>=0;i--)
            u[j*s+i]=(u[j*s+i]-t3*u[j*s+i+1])/r2;
    }
}
//pierwsze składowe
#pragma acc parallel num_gangs(1) deviceptr(u)
{
    for(int j=r-2;j>=0;j--){
       u[j*s]-=t3*u[(j+1)*s]*es0;
    }
}
//całośc
#pragma acc parallel deviceptr(es,u)
{
#pragma acc loop independent
    for(int j=r-2;j>=0;j--){
        last4=t3*u[(j+1)*s];
#pragma acc loop  independent
       for(int i=1;i<s;i++)
            u[j*s+i]-=last4*es[i];
    }
}
    acc_free(es);
    acc_free(e0);

//x
#pragma acc parallel present(b[0:1]) deviceptr(u)
{
    b[0]/=(1+t3*r1*u[0]);
    tmp=t3*r1*b[0];
}
#pragma acc parallel present(b) deviceptr(u)
{
#pragma acc loop  independent
    for(int i=1;i<n;i++){
        b[i]-=tmp*u[i];
    }
}
    acc_free(u);

//     double *res = acc_malloc(sizeof(double)*n);
//     double sum=0;
// #pragma acc data copy(sum)
// {
// #pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
// {
// #pragma acc loop independent
//     for(int i=1;i<n-1;i++){
//         res[i]=(t1*x[i-1]+t2*x[i]+t3*x[i+1]-b[i]);
// 	sum+=res[i]*res[i];
//     }
// }
// #pragma acc parallel num_gangs(1) present(b,x)deviceptr(res) reduction(+:sum)
// {
//     res[0]=(t2*x[0]+t3*x[1]-b[0]);
//     res[n-1]=t1*x[n-2]+t2*x[n-1]-b[n-1];
//     sum+=res[0]*res[0]+res[n-1]*res[n-1];
// }
// }
//     acc_free(res);
//     return sqrt(sum)/normB;
}

float fx(int i,float h, float p, float q){
    return p*cos(i*h)+(q-1)*sin(i*h);
    //return expf(i*h)*(1+p+q);
}

int main(int argc, char **argv){
    int n=atoi(argv[1]);
    int r=atoi(argv[2]);
    //double t1=2,t2=10.1,t3=3;
    double *x=malloc(sizeof (double)*n);
    double *b=malloc(sizeof (double)*n);
    double t;
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

#pragma acc data copy(b[0:n])
{


#pragma acc data present(b)
{
    t=omp_get_wtime();
    tridiagonal(n,r,l1,l2,l3,b);
    t=omp_get_wtime()-t;
}
}
    printf("%.6lf",t);
    free(b);
    free(x);
    return 0;
}
