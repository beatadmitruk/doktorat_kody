#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "immintrin.h"
#include <omp.h>


#define NREP 1

// permutowanie elementow tablicy zadana liczbe razy
void shuff(int n, float *a, int nos)
{
    srand(1233);
    for (int i = 0; i < nos; i++)
    {
        int k1 = rand() % n;
        int k2 = rand() % n;
        float tmp = a[k1];
        a[k1] = a[k2];
        a[k2] = tmp;
        //    printf("%2d  %2d\n",k1,k2);
    }
}

// metoda 1 generowania elementow tablicy
void genrand1(int n, float *a, int chun)
{
    // a[0] = 0.5;
    for (int i = 0; i < n; i++)
    {
        int ii = i % chun;
        float p = (ii + 1) * (ii + 2);
        a[i] = 1.0f / p;
    }
    shuff(n, a,  2*n);
}

// metoda 1A generowania elementow tablicy
void genrand1A(int n, float *a, int chun)
{



    for (int i = 0; i < n/2; i++)
    {
        int ii = i % chun;
        float p = (ii + 1) * (ii + 2);
        a[i] = 1.0f / p;
    }

    for (int i = n/2; i < n; i++)
    {
        int ii = i % chun;
        float p = (ii + 1) * (ii + 2);
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
float okrand1(int n, int chun)
{
    return ((float)(chun) / (chun + 1)) * (n / chun);
}
double sumk(int n, float *a)
{

    //__assume_aligned(a, 64);

    float s = 0;
    float e = 0;

    for(int k=0;k<n;k++)
    {
      float temp=s;
      float y=a[k]+e;
      s=temp+y;
      e=(temp-s)+y;
    }


    return s;

}


// sumowanie rownoleglym algorytmem kompensacyjnym
float vsumk(int n, float *a)
{

    __assume_aligned(a, 64);

    float s = 0;
    float e = 0;


    __m512 vx,vs,ve,vy,vt;


    ve = _mm512_setzero_ps();
    vs = _mm512_setzero_ps();

    for (int k = 0; k < n; k=k+16)
    {
      vt = vs;
      vx = _mm512_load_ps(&a[k]);
      vy = _mm512_add_ps(vx,ve);
      vs = _mm512_add_ps(vt,vy);
      vt = _mm512_sub_ps(vt,vs);
      ve = _mm512_add_ps(vt,vy);
    }

//    s = _mm512_reduce_add_pd(vs);

    for(int k=0;k<16;k++)
    {
      float temp=s;
      float y=vs[k]+e;
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

void tempvkadd(__m512 *vnew,__m512 *vold)
{
    __m512 vs,ve,vy,vt;

//    ve = _mm512_setzero_pd();

    vt = *vold;
    vy = *vnew;
    vs = _mm512_add_ps(vt,vy);
    vt = _mm512_sub_ps(vt,vs);
    ve = _mm512_add_ps(vt,vy);
    *vnew = _mm512_add_ps(vs,ve);
}

#pragma omp declare reduction(vkadd : __m512 : \
        tempvkadd(&omp_out,&omp_in) \
        initializer( omp_priv = _mm512_setzero_ps() )

#pragma omp declare reduction(vadd : __m512 : \
        omp_out=_mm512_add_ps(omp_in,omp_out) \
        initializer( omp_priv = _mm512_setzero_ps() )


float pvsumk(int n, float *a, float *s)
{

    __assume_aligned(a, 64);

    __m512 vsold;

    __m512 vx,vs,ve,vy,vt;


    ve = _mm512_setzero_ps();
    vs = _mm512_setzero_ps();


    vsold=_mm512_setzero_ps();

    double t1;


    t1=omp_get_wtime();


    #pragma omp parallel
    {


    #pragma omp for  firstprivate(vx,vt,vy,ve) reduction(vkadd:vs) schedule(static)
    for (int k = 0; k < n; k=k+16)
    {
      vt = vs;
      vx = _mm512_load_ps(&a[k]);
      vy = _mm512_add_ps(vx,ve);
      vs = _mm512_add_ps(vt,vy);
      vt = _mm512_sub_ps(vt,vs);
      ve = _mm512_add_ps(vt,vy);

    }

    }

    float ss=0;
    float e=0;

    for(int k=0;k<16;k++)
    {
      float temp=ss;
      float y=vs[k]+e;
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

    float *a1 = (float *)_mm_malloc(n * sizeof(float),64);
    genrand1(n, a1, c);


    double t1,t2,t3,t4;

    float xsgm,xsrgm;

    float ss = 0;
    float suma = 0;


    t1=omp_get_wtime();

//    for(int i=0;i<NREP;i++){
//      suma = vsumk(n, a1);
        suma=sumk(n,a1);
//    }
    t1=omp_get_wtime()-t1;




    float ok = okrand1(n, c);



    printf("%lf %e\n",t1,fabs(suma-ok)/fabs(ok));


   return 0;
}
