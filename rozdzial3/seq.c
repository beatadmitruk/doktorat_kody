#include<stdio.h>
#include<math.h>
#include<mm_malloc.h>
#include<omp.h>
#include<time.h>

void method0(int n, double a, double d, double *b, double *u){
        double last;
        const double r2=(d>0)?((d+sqrt(d*d-4*a))/2):((d-sqrt(d*d-4*a))/2);
        const double r1=d-r2;

        b[0]=b[0]/r2;
        for(int i=1;i<n;i++)
                b[i]=(b[i]-b[i-1])/r2;

        for(int i=n-2;i>=0;i--)
                b[i]-=b[i+1]*r1;

        u[0]=1.0/r2;
        last=-1/r2;
        for(int i=1;i<n;i++)
                u[i]=u[i-1]*last;
        for(int i=n-2;i>=0;i--)
                u[i]-=u[i+1]*r1;

        b[0]/=(1+r1*u[0]);
        last=r1*b[0];
        for(int i=1;i<n;i++)
                b[i]-=last*u[i];
}


static void jmb_show_timestamp(char *s) {
    static struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    printf("%s: %10ld.%09ld \n", s,timestamp.tv_sec, timestamp.tv_nsec);
}


int main(int argc, char **argv){
        int n=atoi(argv[1]);
        double a=5, d=2;
        double *b=_mm_malloc(sizeof(double)*n,64);

        for(int i=0;i<n;i++) b[i]=1;
        double *u=_mm_malloc(sizeof(double)*n,64);

        double t=omp_get_wtime();
        //jmb_show_timestamp("start");
        method0(n,a,d,b,u);
        //jmb_show_timestamp("stop");
        printf("%.10lf",omp_get_wtime()-t);
        _mm_free(u);
        _mm_free(b);
        return 0;
}
