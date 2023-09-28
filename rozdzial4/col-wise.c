#include<stdio.h>
#include<math.h>
#include<mm_malloc.h>
#include<omp.h>
#include<stdlib.h>
#include<openacc.h>

double tridiagonal(int n,int r, double t1, double t2, double t3, double *b, double *x, double normB){
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

#pragma acc parallel present(b,x)
    {
#pragma acc loop independent
        for(int i=0;i<n;i++)
            x[i]=b[i];
    }

//lz=b
//lewa strona
#pragma acc parallel present(x)
{
#pragma acc loop  independent
for(int j=0;j<r;j++){
        for(int i=1;i<s;i++)
            x[j*s+i]-=r1*x[j*s+i-1];
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
#pragma acc parallel num_gangs(1) present(x) //deviceptr(u)
{
    for(int j=1;j<r;j++){
       x[(j+1)*s-1]-=r1*x[j*s-1]*us;//u[s-1];
    }
}
#pragma acc parallel num_gangs(1) deviceptr(u)
{
    for(int j=1;j<r;j++){
       u[(j+1)*s-1]=-r1*u[j*s-1]*us;
    }
}



//całość
#pragma acc parallel present(x) deviceptr(e0)
{
#pragma acc loop independent
    for(int j=1;j<r;j++){
        double *col;
        //col=&x[j*s];
        last1=x[j*s-1]*r1;
#pragma acc loop  independent
       for(int i=0;i<s-1;i++)
            x[j*s+i]-=last1*e0[i];
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
#pragma acc parallel present(x)
{
#pragma acc loop  independent
    for(int j=r-1;j>=0;j--){
        x[j*s+s-1]/=r2;
        for(int i=s-2;i>=0;i--)
            x[j*s+i]=(x[j*s+i]-t3*x[j*s+i+1])/r2;
    }
}
//pierwsze składowe
#pragma acc parallel num_gangs(1) present(x)// deviceptr(es)
{
    for(int j=r-2;j>=0;j--){
       x[j*s]-=t3*x[(j+1)*s]*es0;
    }
}
//całość
#pragma acc parallel present(x) deviceptr(es)
{
#pragma acc loop independent
    for(int j=r-2;j>=0;j--){
        last2=t3*x[(j+1)*s];
#pragma acc loop  independent
        for(int i=1;i<s;i++)
            x[j*s+i]-=last2*es[i];
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
#pragma acc parallel present(x[0:1]) deviceptr(u)
{
    x[0]/=(1+t3*r1*u[0]);
    tmp=t3*r1*x[0];
}
#pragma acc parallel present(x) deviceptr(u)
{
#pragma acc loop  independent
    for(int i=1;i<n;i++){
        x[i]=x[i]-tmp*u[i];
    }
}
    acc_free(u);

    double *res = acc_malloc(sizeof(double)*n);
    double sum=0;
#pragma acc data copy(sum)
{
#pragma acc parallel present(b,x) deviceptr(res) reduction(+:sum)
{
#pragma acc loop independent
    for(int i=1;i<n-1;i++){
        res[i]=(t1*x[i-1]+t2*x[i]+t3*x[i+1]-b[i]);
        sum+=res[i]*res[i];
    }
}
#pragma acc parallel num_gangs(1) present(b,x)deviceptr(res) reduction(+:sum)
{
    res[0]=(t2*x[0]+t3*x[1]-b[0]);
    res[n-1]=t1*x[n-2]+t2*x[n-1]-b[n-1];
    sum+=res[0]*res[0]+res[n-1]*res[n-1];
}
}
    acc_free(res);
    return sqrt(sum)/normB;
}

float fx(int i,float h, float p, float q){
    return p*cos(i*h)+(q-1)*sin(i*h);
    //return expf(i*h)*(1+p+q);
}

int main(int argc, char **argv){
    int n=atoi(argv[1]);
    int r=atoi(argv[2]);
    //double t1=2,t2=10.1,t3=3;
    double *b=malloc(sizeof (double)*n);
    double *x=malloc(sizeof (double)*n);
    double t, test;
    double dh=M_PI/(n+1);
    double dh2=dh*dh;
    float h2=(float)dh2;
    float h=(float)dh;
    double normB=0;
    float A=0,B=0;
    float p=100, q=pow(10,12);
    float l1=1+p*h/2,l2=-2-h*h*q,l3=1-p*h/2;
    if(fabs(l1)+fabs(l3)>=fabs(l2)){
        printf("data error\n");
        return 1;
    }

    acc_init(acc_device_nvidia);

#pragma acc data create(b[0:n])copyout(x[0:n])
{
#pragma acc parallel loop independent present(b) reduction(+:normB)
    for (int i=1;i<n-1;i++){
        b[i]=h*h*fx(i+1,h,p,q);
        normB+=b[i]*b[i];
    }

#pragma acc parallel num_gangs(1) present(b) reduction(+:normB)
{
    b[0]=fx(1,h,p,q)*h*h-A*(1+p*h/2);
    b[n-1]=fx(n,h,p,q)*h*h-B*(1-p*h/2);
    normB+=b[0]*b[0]+b[n-1]*b[n-1];
}

    normB=sqrt(normB);
#pragma acc data present(b,x)
{
    t=omp_get_wtime();
    test=tridiagonal(n,r,l1,l2,l3,b,x,normB);
    t=omp_get_wtime()-t;
}
}
    printf("%.30lf %lf",test,t);
    free(b);
    free(x);
    return 0;
}
