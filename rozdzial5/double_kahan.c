#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "immintrin.h"
#include <omp.h>
//#include <openacc.h>


#define NREP 1

// permutowanie elementow tablicy zadana liczbe razy
void shuff(int n, double *a, int nos)
{
    srand(1233);
    for (int i = 0; i < nos; i++)
    {
        int k1 = rand() % n;
        int k2 = rand() % n;
        double tmp = a[k1];
        a[k1] = a[k2];
        a[k2] = tmp;
        //    printf("%2d  %2d\n",k1,k2);
    }
}

// metoda 1 generowania elementow tablicy
void genrand1(int n, double *a, int chun)
{
    // a[0] = 0.5;
    for (int i = 0; i < n; i++)
    {
        int ii = i % chun;
        double p = (ii + 1) * (ii + 2);
        a[i] = 1.0f / p;
    }
    shuff(n, a,  2*n);
}

// metoda 1A generowania elementow tablicy
void genrand1A(int n, double *a, int chun)
{



    for (int i = 0; i < n/2; i++)
    {
        int ii = i % chun;
        double p = (ii + 1) * (ii + 2);
        a[i] = 1.0f / p;
    }

    for (int i = n/2; i < n; i++)
    {
        int ii = i % chun;
        double p = (ii + 1) * (ii + 2);
        a[i] = -1.0f / p;
    }

    for (int i = 0; i < chun; i++)
    {
        a[i] = chun;
    }

    for (int i = n/2; i < n/2+chun; i++)
    {
        a[i] = chun;
    }


    shuff(n, a,  2*n);
}


// suma elementow dla metody 1
double okrand1(int n, int chun)
{
    return ((double)(chun) / (chun + 1)) * (n / chun);
}

double okrand1A(int n, int chun)
{
    return (double)(2.0*chun*chun);
}

double sumk(int n, double *a)
{

    __assume_aligned(a, 64);

    double s = 0;
    double e = 0;

    for(int k=0;k<n;k++)
    {
      double temp=s;
      double y=a[k]+e;
      s=temp+y;
      e=(temp-s)+y;
    }


    return s;

}


// sumowanie rownoleglym algorytmem kompensacyjnym
double vsumk(int n, double *a)
{

    __assume_aligned(a, 64);

    double s = 0;
    double e = 0;


    __m512d vx,vs,ve,vy,vt;


    ve = _mm512_setzero_pd();
    vs = _mm512_setzero_pd();

    for (int k = 0; k < n; k=k+8)
    {
      vt = vs;
      vx = _mm512_load_pd(&a[k]);
      vy = _mm512_add_pd(vx,ve);
      vs = _mm512_add_pd(vt,vy);
      vt = _mm512_sub_pd(vt,vs);
      ve = _mm512_add_pd(vt,vy);
    }

//    s = _mm512_reduce_add_pd(vs);

    for(int k=0;k<8;k++)
    {
      double temp=s;
      double y=vs[k]+e;
      s=temp+y;
      e=(temp-s)+y;
    }


    return s;


/*
    ve = _mm512_setzero_pd();
    vs = _mm512_setzero_pd();

    for(int k=8;k>1;k=k/2){
      __mmask8 b=k/2;
      cmask = mask >> b;
      mask = mask - cmask;
      vt = _mm512_maskz_mov_pd(cmask,vs);

      vx    = _mm512_maskz_compress_pd(mask,vs);
      vs = _mm512_add_pd(vsold,vx);
      vt = _mm512_sub_pd(vs,vsold);
      vt = _mm512_sub_pd(vx,vt);
      vp = _mm512_add_pd(vp,vt);
      mask=cmask;
    }

    return vs[0]+vp[0];
*/


}

void tempvkadd(__m512d *vnew,__m512d *vold)
{
    __m512d vs,ve,vy,vt;

//    ve = _mm512_setzero_pd();

    vt = *vold;
    vy = *vnew;
    vs = _mm512_add_pd(vt,vy);
    vt = _mm512_sub_pd(vt,vs);
    ve = _mm512_add_pd(vt,vy);
    *vnew = _mm512_add_pd(vs,ve);
}

#pragma omp declare reduction(vkadd : __m512d : \
        tempvkadd(&omp_out,&omp_in) \
        initializer( omp_priv = _mm512_setzero_pd() )

#pragma omp declare reduction(vadd : __m512d : \
        omp_out=_mm512_add_pd(omp_in,omp_out) \
        initializer( omp_priv = _mm512_setzero_pd() )


double pvsumk(int n, double *a, double *s)
{

    __assume_aligned(a, 64);

    __m512d vsold;

    __m512d vx,vs,ve,vy,vt;


    ve = _mm512_setzero_pd();
    vs = _mm512_setzero_pd();


    vsold=_mm512_setzero_pd();

    double t1;


    t1=omp_get_wtime();


    #pragma omp parallel
    {


    #pragma omp for  firstprivate(vx,vt,vy,ve) reduction(vkadd:vs) schedule(static)
    for (int k = 0; k < n; k=k+8)
    {
      vt = vs;
      vx = _mm512_load_pd(&a[k]);
      vy = _mm512_add_pd(vx,ve);
      vs = _mm512_add_pd(vt,vy);
      vt = _mm512_sub_pd(vt,vs);
      ve = _mm512_add_pd(vt,vy);

    }

    }

    double ss=0;
    double e=0;

    for(int k=0;k<8;k++)
    {
      double temp=ss;
      double y=vs[k]+e;
      ss=temp+y;
      e=(temp-ss)+y;
    }

//    printf(">>>>> %lf\n",ss);

    t1=omp_get_wtime()-t1;

    *s= ss;

    return t1;
}






int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);

    double *a1 = (double *)_mm_malloc(n * sizeof(double),64);
    genrand1(n, a1, c);


    double t1,t2,t3,t4;

    double xsgm,xsrgm;

    double ss = 0;
    double suma;


    t1=omp_get_wtime();

//    for(int i=0;i<NREP;i++){
      suma = vsumk(n, a1);
//      suma = vsumk(n, a1);
//        pvsumk(n,a1,&suma);
//    }
    t1=omp_get_wtime()-t1;




    double ok = okrand1(n, c);



    printf("%lf %e\n",t1,fabs(suma-ok)/fabs(ok));


   return 0;
}
