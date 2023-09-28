#include<stdio.h>
#include<math.h>
#include<omp.h>
#include<stdlib.h>


void tridiagonal_t1_t2_t3(int n, double t1, double t2, double t3, double *x, double *b){
    double *u=malloc(sizeof (double)*(n-1));
    const double r1=(t2>0) ? (t2-sqrt(t2*t2-4*t1*t3))/(2*t3) : (t2+sqrt(t2*t2-4*t1*t3))/(2*t3);
    const double r2=t2-t3*r1;
    double last;
//v
    for(int i=1;i<n;i++)
	b[i]-=r1*b[i-1];

    b[n-1]/=r2;
    for(int i=n-2;i>=0;i--)
	b[i]=(b[i]-t3*b[i+1])/r2;

//u
    u[0]=1;
    for(int i=1;i<n;i++)
	u[i]=-r1*u[i-1];

    u[n-1]/=r2;
    for(int i=n-2;i>=0;i--)
	u[i]=(u[i]-t3*u[i+1])/r2;

//x
    b[0]/=(1+t3*r1*u[0]);
    last=t3*r1*b[0];

    for(int i=1;i<n;i++)
        b[i]-=last*u[i];
    free(u);

}

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
float fx(int i,float h, float p, float q){
    return p*cos(i*h)+(q-1)*sin(i*h);
    //return expf(i*h)*(1+p+q);
}

int main(int argc, char **argv){
    //tridiagoanl test
    int n=atoi(argv[1]);
    double *b=malloc(sizeof (double)*n);
    double *x=malloc(sizeof (double)*n);

//     double c=1;
//     double l1=-1-c, l2=5+c,l3=-1;
    double l1=-10, l2=11, l3=-1;

  b[0]=x[0]=l2;
  for (int i=1;i<n;i++){
    x[i]=b[i]=0;
  }

    double t=omp_get_wtime();
    tridiagonal_t1_t2_t3(n,l1,l2,l3,x,b);
    t=omp_get_wtime()-t;
    //printf("%e",test(n,l1,l2,l3,b,x));
    printf("%.6lf",t);
    //puts("x");
    //for(int i=0;i<n;i++)
    //    printf("%.30lf\n", b[i]);
    free(b);
    free(x);
}

