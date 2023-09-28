#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double test(int n, float t1, float t2, float t3,double *x, double *b, double normB){
    double sum=0, tmp;

    for(int i=1;i<n-1;i++){
        tmp=(t1*x[i-1]+t2*x[i]+t3*x[i+1]-b[i]);
        sum+=tmp*tmp;
    }

    tmp=(t2*x[0]+t3*x[1]-b[0]);
    sum+=tmp*tmp;
    tmp=t1*x[n-2]+t2*x[n-1]-b[n-1];
    sum+=tmp*tmp;


    return sqrt(sum)/normB;
}


void thomas_algorithm(int n, double t1, double t2, double t3, double* f, double *x) {

    double *c=malloc(sizeof (double)*n);
    double *d=malloc(sizeof (double)*n);

    c[0] = t3 / t2;
    d[0] = f[0] / t2;

    for (int i=1; i<n; i++) {
        double m = 1.0 / (t2 - t1 * c[i-1]);
        c[i] = t3 * m;
        d[i] = (f[i] - t1 * d[i-1]) * m;
    }

    x[n-1]=d[n-1];
    for (int i=n-2; i>=0;i-- ) {
        x[i] = d[i] - c[i] * x[i+1];
    }
    free(c);
    free(d);

}

void thomas_algorithm2(int n, double t1, double t2, double t3, double* f, double *x) {

    double *b=malloc(sizeof (double)*n);
    double *d=malloc(sizeof (double)*n);

    double w;
    b[0]=t2;
    for (int i=1; i<n; i++) {
        w=t1/b[i-1];
        b[i] = t2-w*t3;
        d[i] = f[i] - w*d[i-1];
    }

    x[n-1]=d[n-1]/b[n-1];
    for (int i=n-2; i>=0;i-- ) {
        x[i] = (d[i] - t3 * x[i+1])/b[i];
    }
    free(b);
    free(d);

}
float fx(int i,float h, float p, float q){
    return p*cos(i*h)+(q-1)*sin(i*h);
    //return expf(i*h)*(1+p+q);
}

void run(int n, FILE *fp){
    n=pow(2,n);
    double *f=malloc(sizeof (double)*n);
    double *x=malloc(sizeof (double)*n);
    double t;
    double dh=M_PI/(n+1);
    float h=(float)dh;
    double norm=0;
    float A=0,B=0;
    float p=100, q=pow(10,12);
    float t1=1+p*h/2,t2=-2-h*h*q,t3=1-p*h/2;
    if(fabs(t1)+fabs(t3)>=fabs(t2)){
        printf("data error\n");
        return ;
    }

    for (int i=1;i<n-1;i++){
        f[i]=h*h*fx(i+1,h,p,q);
        norm+=f[i]*f[i];
    }


    f[0]=fx(1,h,p,q)*h*h-A*(1+p*h/2);
    f[n-1]=fx(n,h,p,q)*h*h-B*(1-p*h/2);
    norm+=f[0]*f[0]+f[n-1]*f[n-1];


    norm=sqrt(norm);


    t=omp_get_wtime();
    thomas_algorithm(n,t1,t2,t3,f,x);
    t=omp_get_wtime()-t;

    //    for (int i=0; i<n; i++) {
    //        printf("x[%d]=%lf\n",i,x[i]);
    //    }
    fprintf(fp,"%d\t%lf\t%e\n",n, t, test(n,t1,t2,t3,x,f,norm));
    //t=omp_get_wtime();
    //thomas_algorithm2(n,t1,t2,t3,f,x);
    //t=omp_get_wtime()-t;

    //    for (int i=0; i<n; i++) {
    //        printf("x[%d]=%lf\n",i,x[i]);
    //    }
    //fprintf(fp,"%lf\t%e\n", t, test(n,t1,t2,t3,x,f,norm));
    printf("%d\n",n);
    free(f);
    free(x);
}



int main() {
    FILE *fp;

    if((fp=fopen("thomas.csv","w"))!=NULL){
        for(int i=2;i<31;i++){
            run(i,fp);
        }
        fclose(fp);
    }



    return 0;
}
