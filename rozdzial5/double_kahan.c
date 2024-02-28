#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "immintrin.h"
#include <omp.h>



#define NREP 1


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

    }
}


void genrand1(int n, double *a, int chun)
{

    for (int i = 0; i < n; i++)
    {
        int ii = i % chun;
        double p = (ii + 1) * (ii + 2);
        a[i] = 1.0f / p;
    }
    shuff(n, a,  2*n);
}





double okrand1(int n, int chun)
{
    return ((double)(chun) / (chun + 1)) * (n / chun);
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



    for(int k=0;k<8;k++)
    {
      double temp=s;
      double y=vs[k]+e;
      s=temp+y;
      e=(temp-s)+y;
    }


    return s;





}

void tempvkadd(__m512d *vnew,__m512d *vold)
{
    __m512d vs,ve,vy,vt;



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


      suma = vsumk(n, a1);

    t1=omp_get_wtime()-t1;




    double ok = okrand1(n, c);



    printf("%lf %e\n",t1,fabs(suma-ok)/fabs(ok));
    free(a1);


   return 0;
}
