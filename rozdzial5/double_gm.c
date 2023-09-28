#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "immintrin.h"
#include <omp.h>
//#include <openacc.h>


#define NREP 500

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


// suma elementow dla metody 1
double okrand1(int n, int chun)
{
    return ((double)(chun) / (chun + 1)) * (n / chun);
}


// sumowanie sekwencyjne
double sumord(int n, double *a)
{

    double s = 0;
    for (int i = 0; i < n; i++)
    {
        s += a[i];

    }
    return s;
}

double psumord(int n, double *a)
{
    double s = 0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (int i = 0; i < n; i++)
    {
        s += a[i];
    }
    return s;
}


// sumowanie algorytmem Gilla-Mollera
double sumgm(int n, double *a)
{
    __assume_aligned(a, 64);

    double s = 0;
    double p = 0;
    double sold = 0;
    for (int i = 0; i < n; i++)
    {
        s = (sold + a[i]);
        p = p +( (a[i] - (s - sold)) );
        sold = s;
    }
    return s + p;
}



double vsumgm(int n, double *a)
{

    __assume_aligned(a, 64);

    __m512d vx,vs,vp,vsold,vt;
    __mmask8 mask = 0xFF;
    __mmask8 cmask =0;


    vp = _mm512_setzero_pd();
    vs = _mm512_setzero_pd();
    vsold = _mm512_setzero_pd();

    for (int k = 0; k < n; k=k+8)
    {
      vx = _mm512_load_pd(&a[k]);
      vs = _mm512_add_pd(vsold,vx);
      vt = _mm512_sub_pd(vs,vsold);
      vt = _mm512_sub_pd(vx,vt);
      vp = _mm512_add_pd(vp,vt);
      vsold = vs;
    }

    vs = _mm512_add_pd(vs,vp);

    vp = _mm512_setzero_pd();
    vsold = _mm512_setzero_pd();


    for(int k=8;k>1;k=k/2){
      __mmask8 b=k/2;
      cmask = mask >> b;
      mask = mask - cmask;
      vx    = _mm512_maskz_compress_pd(mask,vs);
      vsold = _mm512_maskz_mov_pd(cmask,vs);
      vs = _mm512_add_pd(vsold,vx);
      vt = _mm512_sub_pd(vs,vsold);
      vt = _mm512_sub_pd(vx,vt);
      vp = _mm512_add_pd(vp,vt);
      mask=cmask;
    }

    return vs[0]+vp[0];
}




struct GMSum
{
  __m512d vs,vp;
};

typedef struct GMSum GMSum;


void tempvzero(GMSum *vnew)
{
  vnew->vp = _mm512_setzero_pd();
  vnew->vs = _mm512_setzero_pd();
}

void tempvgmadd(GMSum *vnew,GMSum *vold)
{

    __m512d tvs,tvp;

    tvs = _mm512_add_pd(vold->vs,vnew->vs);
    tvp = _mm512_sub_pd(tvs,vold->vs);
    tvp = _mm512_sub_pd(vnew->vs,tvp);
    tvp = _mm512_sub_pd(vnew->vp,tvp);
    vnew->vp = _mm512_add_pd(vold->vp,tvp);
    vnew->vs = tvs;

}



#pragma omp declare reduction(vgmadd : GMSum : \
        tempvgmadd(&omp_out,&omp_in) \
        initializer( tempvzero(&omp_priv ) )



#pragma omp declare reduction(vadd : __m512d : \
        omp_out=_mm512_add_pd(omp_in,omp_out) \
        initializer( omp_priv = _mm512_setzero_pd() )



double pvsumgm(int n, double *a, double *s)
{

    __assume_aligned(a, 64);

    __m512d vx,vs,vt,vp,old;
    __mmask8 mask = 0xFF;
    __mmask8 cmask =0;


    GMSum vsold;

    tempvzero(&vsold);

    double t1;


    t1=omp_get_wtime();


    #pragma omp parallel
    {

/*
    #pragma omp master
    {
    t1=omp_get_wtime();
    }
*/

    #pragma omp for  private(vx,vt,vs) reduction(vgmadd:vsold) schedule(static)
    for (int k = 0; k < n; k=k+8)
    {
      vx = _mm512_load_pd(&a[k]);
      vs = _mm512_add_pd(vsold.vs,vx);
      vt = _mm512_sub_pd(vs,vsold.vs);
      vt = _mm512_sub_pd(vx,vt);
      vsold.vp = _mm512_add_pd(vsold.vp,vt);
      vsold.vs = vs;
    }

    }

    vs = _mm512_add_pd(vsold.vs,vsold.vp);

    vp = _mm512_setzero_pd();
    old = _mm512_setzero_pd();


    for(int k=8;k>1;k=k/2){
      __mmask8 b=k/2;
      cmask = mask >> b;
      mask = mask - cmask;
      vx    = _mm512_maskz_compress_pd(mask,vs);
      old = _mm512_maskz_mov_pd(cmask,vs);
      vs = _mm512_add_pd(old,vx);
      vt = _mm512_sub_pd(vs,old);
      vt = _mm512_sub_pd(vx,vt);
      vp = _mm512_add_pd(vp,vt);
      mask=cmask;
    }

    t1=omp_get_wtime()-t1;

//    *t+=(t2-t1);
    //*t=fmin(*t,(t2-t1));


    *s= vs[0]+vp[0];

        return t1;
}



int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);

    __declspec(align(64))  double *a4 = (double *)_mm_malloc(n * sizeof(double),64);



    genrand1(n, a4, c);



    double t1,t2,t3,t4;

    double xsgm,xsrgm;



    double ss = 0;
    double suma;

    t4=omp_get_wtime();



    t4=pvsumgm(n,a4,&xsrgm);


    _mm_free(a4);
    double ok = okrand1(n, c);


    printf("procs=%i\n",omp_get_num_procs());
    printf("%.20lf", t4);
    printf(" %.30lf", fabs(xsrgm-ok)/fabs(ok));


    return 0;
}
