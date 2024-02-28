#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "immintrin.h"
#include <omp.h>

void shuff(int n, float *a, int nos)
{
    srand(123);
    for (int i = 0; i < nos; i++)
    {
        int k1 = rand() % n;
        int k2 = rand() % n;
        float tmp = a[k1];
        a[k1] = a[k2];
        a[k2] = tmp;

    }
}


void genrand1(int n, float *a, int chun)
{

    for (int i = 0; i < n; i++)
    {
        int ii = i % chun;
        float p = (ii + 1) * (ii + 2);
        a[i] = 1.0f / p;
    }
   shuff(n, a,  2*n);
}


float okrand1(int n, int chun)
{
    return ((double)(chun) / (double)(chun + 1)) * ((double)n / chun);
}

float sumord(int n, float *a)
{
    float s = 0;

    for (int i = 0; i < n; i++)
    {
        s += a[i];
    }
    return (double)s;
}



float msumgm(int n, float *a)
{
    float s = 0;
    double p = 0; // double
    float sold = 0;
    for (int i = 0; i < n; i++)
    {
        s = (sold + a[i]);
        double tmp = s-sold;
        p = p+ (a[i] -tmp) ; // double
        sold = s;
    }
    return (s+p);
}

float sumgm(int n, float *a)
{
    float s = 0;
    float p = 0; // double
    float sold = 0;
    for (int i = 0; i < n; i++)
    {
        s = (sold + a[i]);
        p = p+ (a[i] -(s-sold)) ; // double
        sold = s;
    }
    return (s+p);
}


float vsumgm(int n, float *a)
{

    __m512 vx,vs,vp,vsold,vt;

    vp = _mm512_setzero_ps();
    vs = _mm512_setzero_ps();
    vsold = _mm512_setzero_ps();

    for (int k = 0; k < n; k=k+16)
    {
      vx = _mm512_load_ps(&a[k]);
      vs = _mm512_add_ps(vsold,vx);
      vt = _mm512_sub_ps(vs,vsold);
      vt = _mm512_sub_ps(vx,vt);
      vp = _mm512_add_ps(vp,vt);
      vsold = vs;
    }

    vs = _mm512_add_ps(vs,vp);

    float s = 0;
    float p = 0; // double
    float sold = 0;
    for (int i = 0; i < 16; i++)
    {
        s = (sold + vs[i]);
        p = p+ (vs[i] - (s-sold)) ; // double
        sold = s;
    }
    return (s+p);

}

float mvsumgm(int n, float *a)
{

    __m512 vx,vs,vsold,vt;

    __m512d vp0,vp1;
    __m512d vx0,vx1;
    __m512d vt0,vt1;

    __mmask16 mask = 0xff00;

    vp0 = _mm512_setzero_pd();
    vp1 = _mm512_setzero_pd();

    vs = _mm512_setzero_ps();
    vsold = _mm512_setzero_ps();

    for (int k = 0; k < n; k=k+16)
    {
      // vx:=a[k]  FP32
      vx = _mm512_load_ps(&a[k]);
      // vs:=vsold+vx  FP32
      vs = _mm512_add_ps(vsold,vx);

      // vx:  FP32 -> FP64
      vx0= _mm512_cvtpslo_pd(vx);
      vx = _mm512_maskz_compress_ps(mask,vx);
      vx1= _mm512_cvtpslo_pd(vx);

      // vt:=vs-vsold FP32
      vt = _mm512_sub_ps(vs,vsold);

      // vt:  FP32 -> FP64
      vt0= _mm512_cvtpslo_pd(vt);
      vt = _mm512_maskz_compress_ps(mask,vt);
      vt1= _mm512_cvtpslo_pd(vt);

      // vt:=vx-vt   FP64
      vt0 = _mm512_sub_pd(vx0,vt0);
      vt1 = _mm512_sub_pd(vx1,vt1);

      // vp:=vp+vt   FP64
      vp0 = _mm512_add_pd(vp0,vt0);
      vp1 = _mm512_add_pd(vp1,vt1);

      // next step
      vsold = vs;
    }

   // vx:=vs  FP32 -> FP64
    vx0= _mm512_cvtpslo_pd(vs);
    vs = _mm512_maskz_compress_ps(mask,vs);
    vx1= _mm512_cvtpslo_pd(vs);

   // vp:=vp+vx   FP64
    vp0 = _mm512_add_pd(vp0,vx0);
    vp1 = _mm512_add_pd(vp1,vx1);


    double s = 0;
    double p = 0;
    double sold = 0;

    // first half of vp
    for (int i = 0; i < 8; i++)
    {
        s = (sold + vp0[i]);
        p = p+ (vp0[i] - (s-sold)) ;
        sold = s;
    }

    // second half of vp
    for (int i = 0; i < 8; i++)
    {
        s = (sold + vp1[i]);
        p = p+ (vp1[i] - (s-sold)) ;
        sold = s;
    }


    return (s+p);

}





struct GMSum
{
  __m512 vs,vp;
};

typedef struct GMSum GMSum;


void tempvzero(GMSum *vnew)
{
  vnew->vp = _mm512_setzero_ps();
  vnew->vs = _mm512_setzero_ps();
}

void fpvsumgm(GMSum *vnew,GMSum *vold)
{

    __m512 tvs,tvp;

    tvs = _mm512_add_ps(vold->vs,vnew->vs);
    tvp = _mm512_sub_ps(tvs,vold->vs);
    tvp = _mm512_sub_ps(vnew->vs,tvp);
    tvp = _mm512_sub_ps(vnew->vp,tvp);
    vnew->vp = _mm512_add_ps(vold->vp,tvp);
    vnew->vs = tvs;

}
#pragma omp declare reduction(vgmadd : GMSum : \
        tempvgmadd(&omp_out,&omp_in) \
        initializer( tempvzero(&omp_priv ) )


#pragma omp declare reduction(vadd : __m512 : \
        omp_out=_mm512_add_ps(omp_in,omp_out) \
        initializer( omp_priv = _mm512_setzero_ps() )



float tpvgmsum(int n,float  *a)
{


    __m512 vx,vs,vt,vp,old;
    __mmask16 mask = 0xFFFF;
    __mmask16 cmask =0;


    GMSum vsold;

    tempvzero(&vsold);



    #pragma omp parallel
    {



    #pragma omp for  private(vx,vt,vs) reduction(vgmadd:vsold) schedule(static)
    for (int k = 0; k < n; k=k+16)
    {
      vx = _mm512_load_ps(&a[k]);
      vs = _mm512_add_ps(vsold.vs,vx);
      vt = _mm512_sub_ps(vs,vsold.vs);
      vt = _mm512_sub_ps(vx,vt);
      vsold.vp = _mm512_add_ps(vsold.vp,vt);
      vsold.vs = vs;
    }

    }

    vs = _mm512_add_ps(vsold.vs,vsold.vp);

    vp = _mm512_setzero_ps();
    old = _mm512_setzero_ps();


    for(int k=16;k>1;k=k/2){
      __mmask16 b=k/2;
      cmask = mask >> b;
      mask = mask - cmask;
      vx    = _mm512_maskz_compress_ps(mask,vs);
      old = _mm512_maskz_mov_ps(cmask,vs);
      vs = _mm512_add_ps(old,vx);
      vt = _mm512_sub_ps(vs,old);
      vt = _mm512_sub_ps(vx,vt);
      vp = _mm512_add_ps(vp,vt);
      mask=cmask;
    }




    return  (double)vs[0]+(double)vp[0];



}


int main(int argc, char **argv)
{
    int n = atoi(argv[1]);
    int c = atoi(argv[2]);

    float *a1 = (float *)malloc(n * sizeof(float));


    genrand1(n, a1, c);

    double t1;





    t1=omp_get_wtime();
    float suma= mvsumgm(n, a1);
    t1=omp_get_wtime()-t1;




   free(a1);



    float ok = okrand1(n, c);
    printf("%.30f ", fabs(suma - ok) / fabs(ok) );

    printf("%.10f",t1);
    return 0;
}
