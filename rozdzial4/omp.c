#include<stdio.h>
#include<math.h>
#include<omp.h>


#define M_PI 3.14159265358979323846
void trg_omp(int n,int r, double t1,double t2,double t3, double *x, double *b){
    const double r1=(t2>0) ? (t2-sqrt(t2*t2-4*t1*t3))/(2*t3) : (t2+sqrt(t2*t2-4*t1*t3))/(2*t3);
    const double r2=t2-t3*r1;
    int s=n/r;
    double tmp;

    double *u=(double*)malloc((size_t)n*sizeof(double));
    double *es=(double*)malloc((size_t)s*sizeof(double));

    u[0]=1;//wyliczam pierwszy dla v prawa strona
    es[s-1]=1./r2;


#pragma omp parallel
    {
        //przepisanie do x - b zostawiamy do testu
#pragma omp for nowait schedule(static)
        for(int i=0;i<n;i++)
            x[i]=b[i];
#pragma omp barrier

#pragma omp single
        {
            for(int i=1;i<s;i++){
                u[i]=-r1*u[i-1];//e_0=(1,0,...,0)
            }
        }
#pragma omp single
        {
            for(int i=s-2;i>=0;i--){
                es[i]=-t3/r2*es[i+1];
            }
        }
        //dla v
        //wzor (9):lewa strona
#pragma omp for nowait schedule(static)
        for(int j=0;j<r;j++){
            for(int i=1;i<s;i++){
                x[j*s+i]-=r1*x[j*s+i-1];
            }
        }


        //ostatnie skĹ‚adowe
#pragma omp single
        {
            for(int j=1;j<r;j++){
                x[(j+1)*s-1]-=r1*x[j*s-1]*u[s-1];
                u[(j+1)*s-1]=-r1*u[j*s-1]*u[s-1];
            }
        }

        //wzor (9):calosc strona
        for(int j=1;j<r;j++){
            double last=x[j*s-1]*r1;
            double last1=r1*u[j*s-1];
#pragma omp for simd nowait schedule(static)
            for(int i=0;i<s-1;i++){
                x[j*s+i]-=last*u[i];
                u[j*s+i]=-last1*u[i];
            }
        }


#pragma omp barrier
        //wzor (12):lewa strona
#pragma omp for nowait schedule(static)
        for(int j=0;j<r;j++){
            x[j*s+s-1]/=r2;
            u[j*s+s-1]/=r2;
            for(int i=s-2;i>=0;i--){
                x[j*s+i]=(x[j*s+i]-t3*x[j*s+i+1])/r2;
                u[j*s+i]=(u[j*s+i]-t3*u[j*s+i+1])/r2;
            }
        }


        //wzor (12):prawa strona
#pragma omp barrier

        //pierwsze skĹ‚adowe
#pragma omp single
        {
            for(int j=r-2;j>=0;j--){
                x[j*s]-=t3*x[(j+1)*s]*es[0];
                u[j*s]-=u[(j+1)*s]*t3*es[0];
            }
        }



        //wzor (12):calosc
        for(int j=r-2;j>=0;j--){
            double last=x[(j+1)*s]*t3;
            double last1=u[(j+1)*s]*t3;
#pragma omp for simd  nowait schedule(static)
            for(int i=1;i<s;i++){
                x[j*s+i]-=last*es[i];
                u[j*s+i]-=last1*es[i];
            }
        }


#pragma omp barrier

free(es);

        //wzor (4) - dobrze
#pragma omp single
        {
            x[0]/=(1+t3*r1*u[0]);
            tmp=t3*r1*x[0];
        }
#pragma omp for simd schedule(static)
        for(int i=1;i<n;i++){
            x[i]-=tmp*u[i];
        }
    }

    free(u);
}


double test(int n,double t1, double t2, double t3, double *x, double *b, double norm){
    double tmp,s=0;
    for(int i=1;i<n-1;i++){
        tmp=(t1*x[i-1]+t2*x[i]+t3*x[i+1]-b[i]);
        s+=tmp*tmp;
    }
    tmp=(t2*x[0]+t3*x[1]-b[0]);
    s+=tmp*tmp;
    tmp=t1*x[n-2]+t2*x[n-1]-b[n-1];
    s+=tmp*tmp;
    return sqrt(s)/norm;
}

float fx(int i,float h, float p, float q){
    return p*cos(i*h)+(q-1)*sin(i*h);
    //return expf(i*h)*(1+p+q);
}



int main(int argc, char **argv)
{
    int n=atoi(argv[1]);
    int r=atoi(argv[2]);
    double *b=malloc(sizeof (double)*n);
    double *x=malloc(sizeof (double)*n);
    double dh=M_PI/(n+1);
    double dh2=dh*dh;
    float h2=(float)dh2;
    float h=(float)dh;
    double norm=0;
    float A=0,B=0;
    float p=100, q=pow(10,12);
    float l1=1+p*h/2,l2=-2-h2*q,l3=1-p*h/2;
    //double l1=2,l2=12,l3=3;
    if(fabs(l1)+fabs(l3)>=fabs(l2)){
        printf("data error\n");
        return 1;
    }
#pragma omp parallel for reduction(+:norm)
    for (int i=1;i<n-1;i++){
        b[i]=h*h*fx(i+1,h,p,q);
        norm+=b[i]*b[i];
    }

    b[0]=fx(1,h,p,q)*h*h-A*(1+p*h/2);
    b[n-1]=fx(n,h,p,q)*h*h-B*(1-p*h/2);
    norm+=b[0]*b[0]+b[n-1]*b[n-1];

    norm=sqrt(norm);



    double t0=omp_get_wtime();
    trg_omp(n,r,l1,l2,l3,x,b);

    t0=omp_get_wtime()-t0;
    printf("%.6lf %e",t0, test(n,l1,l2,l3,x,b, norm));
    //    for(int i=0;i<n;i++)
    //        printf("x%d=%.10lf\n",i,x[i]);

    free(b);
    free(x);

    return 0;
}

