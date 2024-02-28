#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>

void s_bvp_set(float *u, int N)
{
    int k;
    float h=1.0/((float)N);
    float h2=h*h;

    for(k=0;k<N;k++){
        float x2=(h*(float)k)*(h*(float)k);
        u[k] = h2*20000.0*exp(-100*x2)*(1-200*x2);
    }
    u[0]*=0.5;
}


void s_bvp_solve(float *u, int N)
{
    int k;
    for(k=1;k<N;k++)
        u[k] += u[k-1];

    for(k=N-2;k>=0;k--){
        u[k] += u[k+1];
    }


}
void s_bvp_solve_Kahan(float *u, int N)
{
    int k;
    //  float s=0;
    float temp;
    float s=u[0];
    float e=0;
    float y;
    for(k=1;k<N;k++){
        //    u[k] += u[k-1];
        temp=s;
        y=u[k]+e;
        u[k]=s=temp+y;
        e=(temp-s)+y;
    }


    s=u[N-1];
    e=0;
    for(k=N-2;k>=0;k--){
        //    u[k] += u[k+1];
        temp=s;
        y=u[k]+e;
        u[k]=s=temp+y;
        e=(temp-s)+y;
    }



}



void v_bvp_set(float *u, int s, int r){

    int j,k,m;
    float h=1.0/((float)(s*r));
    float h2=h*h;

    for(j=0;j<r;j++) {
        m=j*s;
        for(k=0;k<s;k++){
            float x2=(h*(float)(m+k))*(h*(float)(m+k));
            u[m+k] = h2*20000.0*exp(-100*x2)*(1-200*x2);
        }
    }
    u[0]*=0.5;
}


void p_bvp_solve(float *u, int s, int r){

    //1A
    for(int j=0;j<r;j++) {
        for(int k=1;k<s;k++){
            u[j*s+k]+=u[j*s+k-1];
        }
    }

    //1B
    for (int k=1;k<r;k++) {
        u[(k+1)*s-1]+=u[k*s-1];
    }

    //1C
    for(int j=1;j<r;j++) {
        float a=u[j*s-1];
        for (int k=0;k<s-1;k++) {
            u[j*s+k]+=a;
        }

    }
    //2A
    for(int j=0;j<r;j++) {
        for(int k=s-2;k>=0;k--){
            u[j*s+k]+=u[j*s+k+1];

        }
    }
    //2B
    for (int k=r-2;k>=0;k--) {
        u[k*s]+=u[k*s+s];
    }

    //2C
    for(int j=0;j<r-1;j++) {
        float a=u[j*s+s];
        for (int k=1;k<s;k++) {
            u[j*s+k]+=a;
        }

    }


}

void p_bvp_solve_Kahan(float *u, int s, int r){
    float sum, y,e,tmp;
    //1A
    for(int j=0;j<r;j++) {
        sum=u[j*s],e=0;
        for(int k=1;k<s;k++){
            //u[j*s+k]= u[j*s+k]+ u[j*s+k-1];
            tmp=sum;
            y=u[j*s+k]+e;
            u[j*s+k]=sum=tmp+y;
            e=(tmp-sum)+y;

        }
    }


    //1B
    sum=u[s-1],e=0;
    for (int k=1;k<r;k++) {
        //u[(k+1)*s-1]+=u[k*s-1];
        tmp=sum;
        y=u[(k+1)*s-1]+e;
        u[(k+1)*s-1]=sum=tmp+y;
        e=(tmp-sum)+y;
    }



    //1C
    for(int j=1;j<r;j++) {
        float a=u[j*s-1];
        for (int k=0;k<s-1;k++) {
            u[j*s+k]+=a;

        }

    }
    //2A
    for(int j=0;j<r;j++) {
        sum=u[(j+1)*s-1],e=0;
        for(int k=s-2;k>=0;k--){
            //u[j*s+k]+=u[j*s+k+1];
            tmp=sum;
            y=u[j*s+k]+e;
            u[j*s+k]=sum=tmp+y;
            e=(tmp-sum)+y;

        }
    }
    //2B
    sum=u[(r-1)*s],e=0;
    for (int k=r-2;k>=0;k--) {
        //u[k*s]+=u[k*s+s];
        tmp=sum;
        y=u[k*s]+e;
        u[k*s]=sum=tmp+y;
        e=(tmp-sum)+y;
    }

    //2C
    for(int j=0;j<r-1;j++) {
        float a=u[j*s+s];
        for (int k=1;k<s;k++) {
            u[j*s+k]+=a;
        }

    }
    for(int i=0;i<100;i++)
        printf("u[%i]=%.30f\n",i,u[i]);


}


void acc_bvp_solve_Kahan(float *u, int s, int r){
    float sum, y,e,tmp;

// 1A
#pragma acc parallel present(u)
    {
#pragma acc loop  independent
        for(int j=0;j<r;j++) {
            sum=u[j*s],e=0;
            for(int k=1;k<s;k++){
                //u[j*s+k]= u[j*s+k]+ u[j*s+k-1];
                tmp=sum;
                y=u[j*s+k]+e;
                u[j*s+k]=sum=tmp+y;
                e=(tmp-sum)+y;

            }

        }
    }


// 1B

#pragma acc parallel num_gangs(1) present(u)
    {
        sum=u[s-1],e=0;
        for (int k=1;k<r;k++) {
            //u[(k+1)*s-1]+=u[k*s-1];
            tmp=sum;
            y=u[(k+1)*s-1]+e;
            u[(k+1)*s-1]=sum=tmp+y;
            e=(tmp-sum)+y;
        }
    }



// 1C
#pragma acc parallel present(u)
    {
        for(int j=1;j<r;j++) {
            float a=u[j*s-1];
#pragma acc loop  independent
            for (int k=0;k<s-1;k++) {
                u[j*s+k]+=a;

            }

        }
    }

// 2A
#pragma acc parallel present(u)
    {
#pragma acc loop  independent
        for(int j=0;j<r;j++) {
            sum=u[(j+1)*s-1],e=0;
            for(int k=s-2;k>=0;k--){
                //u[j*s+k]+=u[j*s+k+1];
                tmp=sum;
                y=u[j*s+k]+e;
                u[j*s+k]=sum=tmp+y;
                e=(tmp-sum)+y;

            }
        }
    }

// 2B
#pragma acc parallel num_gangs(1) present(u)
    {
        sum=u[(r-1)*s],e=0;
        for (int k=r-2;k>=0;k--) {
            //u[k*s]+=u[k*s+s];
            tmp=sum;
            y=u[k*s]+e;
            u[k*s]=sum=tmp+y;
            e=(tmp-sum)+y;
        }
    }
//2C
#pragma acc parallel present(u)
    for(int j=0;j<r-1;j++) {
        float a=u[j*s+s];
#pragma acc loop independent
        {
            for (int k=1;k<s;k++) {
                u[j*s+k]+=a;
            }
        }
    }





}
void acc_bvp_solve(float *u, int s, int r){

// 1A
#pragma acc parallel present(u)
    {

#pragma acc loop  independent
        for(int j=0;j<r;j++) {

            for(int k=1;k<s;k++){
                u[j*s+k]+=u[j*s+k-1];

            }

        }
    }

// 1B
#pragma acc parallel num_gangs(1) present(u)
    {
        for (int k=1;k<r;k++) {
            u[(k+1)*s-1]+=u[k*s-1];
        }
    }



// 1C
#pragma acc parallel present(u)
    {
        for(int j=1;j<r;j++) {
            float a=u[j*s-1];
#pragma acc loop  independent
            for (int k=0;k<s-1;k++) {
                u[j*s+k]+=a;

            }

        }
    }

// 2A
#pragma acc parallel present(u)
    {
#pragma acc loop  independent
        for(int j=0;j<r;j++) {
            for(int k=s-2;k>=0;k--){
                u[j*s+k]+=u[j*s+k+1];
            }
        }
    }

// 2B
#pragma acc parallel num_gangs(1) present(u)
    {
        for (int k=r-2;k>=0;k--) {
            u[k*s]+=u[k*s+s];
        }
    }
//2C
#pragma acc parallel present(u)
    for(int j=0;j<r-1;j++) {
        float a=u[j*s+s];
#pragma acc loop independent
        {
            for (int k=1;k<s;k++) {
                u[j*s+k]+=a;
            }
        }
    }

}



int main(int argc,char **argv){
    int N,r,s;
    N=atoi(argv[1]);
    r=atoi(argv[2]);
    s=N/r;

    size_t size = N * sizeof(float);

    float *u = (float*)malloc(size);
    double t;

    s_bvp_set(u, N);
    t = omp_get_wtime();
    s_bvp_solve_Kahan(u, N);
    t = omp_get_wtime()-t;

    /*v_bvp_set(u, s,r);
    t = omp_get_wtime();
    p_bvp_solve_Kahan(u, s,r);
    t = omp_get_wtime()-t;

    v_bvp_set(u, s,r);
    acc_init(acc_device_nvidia);

#pragma acc data copy(u[0:N])
    {
#pragma acc data present(u)
        {
            t = omp_get_wtime();
            acc_bvp_solve(u, s,r);
            t = omp_get_wtime()-t;


        }
    }*/


    float h=1.0/((float)N);
    float sum=0.0;
    float sumExact =0.0;

    for (int i=0; i<N; i++) {
        float x2=(h*(float)i)*(h*(float)i);
        float y=100*exp(-100*x2)-100*exp(-100.0);
        sumExact += y*y;
        sum += (y-u[i])*(y-u[i]);

    }
    free(u);
    printf("%.6lf %.30lf",t,(sum/sumExact));
    return 0;
}

