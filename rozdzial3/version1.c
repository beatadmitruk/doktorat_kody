#include<stdio.h>
#include<math.h>
#include<mm_malloc.h>
#include<omp.h>
//#include<time.h>

void version1(int n,int r, double a, double d, double *b, double *u){

    const double r2=(d>0)?((d+sqrt(d*d-4*a))/2):((d-sqrt(d*d-4*a))/2);
    const double r1=d-r2;

    double last1=-1/r2, last2=-r1;
    u[0]=1./r2;//wyliczam pierwszy dla v prawa strona

    int s=n/r;

    double *es=_mm_malloc(s*sizeof(double), 64);
    es[s-1]=1;

#pragma omp parallel
{
    //dla v
    //wzor (9):lewa strona
#pragma omp for nowait schedule(static)
    for(int j=0;j<r;j++){
        double *col;
        __assume_aligned(col,64);
        col=&b[j*s];
        col[0]/=r2;
        for(int i=1;i<s;i++){
            col[i]=(col[i]-col[i-1])/r2;
        }
    }

    //wzor (9):prawa strona u[0,s-1]
    //wyliczam s-1 elementow
#pragma omp single
{
    for(int i=1;i<s;i++){
        u[i]=u[i-1]*last1;//e_0=(1,0,...,0)
    }
}

    //wzor (9):calosc strona
    for(int j=1;j<r;j++){
        double *col;
        __assume_aligned(col,64);
        col=&b[j*s];
        double last=b[j*s-1];//col[-1];
#pragma omp for simd schedule(static)
        for(int i=0;i<s;i++){
            col[i]-=last*u[i];
        }
    }

    //wzor (12):lewa strona
#pragma omp for nowait schedule(static)
    for(int j=0;j<r;j++){
        double *col;
        __assume_aligned(col,64);
        col=&b[j*s];
        for(int i=s-2;i>=0;i--){
            col[i]-=r1*col[i+1];
        }
    }
    //wzor (12):prawa strona


#pragma omp single
{
    for(int i=s-2;i>=0;i--){
        es[i]=last2*es[i+1];
    }
}

    //wzor (12):calosc
    for(int j=r-2;j>=0;j--){
        double *col;
        __assume_aligned(col,64);
        col=&b[j*s];
        double last=b[(j+1)*s]*r1;
#pragma omp for simd schedule(static)
        for(int i=0;i<s;i++){
            col[i]-=last*es[i];
        }
    }
    //mamy v (zapisane w b)


    //to samo dla u
    //wzor (9):lewa strona
    //juz mamy w u[0,n-1]

    //wzor (9):prawa strona
    //taka sama jak dla v, korzystam z juz obliczonej(też w u[0,s-1]

    //wzor (9):calosc
    for(int j=1;j<r;j++){
        double *col;
        __assume_aligned(col,64);
        col=&u[j*s];

        double last=u[j*s-1];
#pragma omp for simd schedule(static)
        for(int i=0;i<s;i++){
            col[i]=-last*u[i];
        }
    }


    //wzor (12):lewa strona
#pragma omp for schedule(static)
    for(int j=0;j<r;j++){
        double *col;
        __assume_aligned(col,64);
        col=&u[j*s];

        for(int i=s-2;i>=0;i--){
            col[i]-=r1*col[i+1];
        }
    }

    //wzor (12):prawa strona
    //już obliczone w u[0,s]

    //wzor (12):calosc
    for(int j=r-2;j>=0;j--){
        double *col;
        __assume_aligned(col,64);
        col=&u[j*s];
        double last=u[(j+1)*s]*r1;
#pragma omp for simd schedule(static)
        for(int i=0;i<s;i++){
            col[i]-=last*es[i];
        }
    }



    //wzor (4) - dobrze
#pragma omp single
{
    b[0]/=(1+r1*u[0]);
    last1=r1*b[0];
}
#pragma omp for simd schedule(static)
    for(int i=1;i<n;i++){
        b[i]-=last1*u[i];
    }
}
    _mm_free(es);
}


double test(int n,double d, double a,double *b){
    double *res = malloc(sizeof(double)*n);
    res[0]=1-(d*b[0]+a*b[1]);
    for(int i=1;i<n-1;i++)
        res[i]=1-(b[i-1]+d*b[i]+a*b[i+1]);
    res[n-1]=1-(b[n-2]+d*b[n-1]);
    double s=0;
    for(int i=0;i<n;i++)
        s+=res[i]*res[i];
    s=sqrt(s);
    free(res);
    return s;
}

/*
static void jmb_show_timestamp(char *s) {
    static struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    printf("%s: %10ld.%09ld \n", s,timestamp.tv_sec, timestamp.tv_nsec);
}*/



int main(int argc, char **argv)
{
    int n=atoi(argv[1]);
    int r=atoi(argv[2]);
    double d=5, a=2;
    double *b=_mm_malloc(sizeof (double)*n,64);


    for (int i=0;i<n;i++)
        b[i]=1;
    double *u=_mm_malloc(sizeof (double)*n,64);


    double t0=omp_get_wtime();
    //jmb_show_timestamp("start");
    version1(n,r,a,d,b,u);
    printf("%e\n", test(n,d,a,b));
    //jmb_show_timestamp("stop");
    t0=omp_get_wtime()-t0;
    printf("%.10lf",t0);
    _mm_free(b);
    _mm_free(u);
    return 0;
}
