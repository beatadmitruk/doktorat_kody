#include<stdio.h>
#include<math.h>
#include<mm_malloc.h>
#include<omp.h>
#include<stdlib.h>
#include<openacc.h>

#define BSIZE 128
#define VL 16
#define NC BSIZE/VL


void tridiagonal(int n,int r, double t1, double t2, double t3, double *x){
    const double r1=(t2>0) ? (t2-sqrt(t2*t2-4*t1*t3))/(2*t3) : (t2+sqrt(t2*t2-4*t1*t3))/(2*t3);
    const double r2=t2-t3*r1;
    int s=n/r;
    double tmp1=-t3/r2, tmp, es0, us;
    double last1, last2, last3, last4;

    double *u;
    u=(double*)acc_malloc((size_t)n*sizeof(double));
    double *es;
    es=(double*)acc_malloc((size_t)s*sizeof(double));



#pragma acc parallel present(x)
    {

      double xc[VL+1][BSIZE];
#pragma acc cache(xc)



         #pragma acc loop gang
      for(int j=0;j<r;j+=BSIZE){

	     #pragma acc loop vector
	for(int k=0;k<BSIZE;k++){
	  xc[0][k]=0.0;
	}


	for(int k=0;k<=s-VL;k+=VL){

	  for(int l=0;l<BSIZE;l+=NC){
	             #pragma acc loop vector
	    for(int i=0;i<BSIZE;i++){
	      xc[i%VL+1][l+i/VL]=x[(j+l+i/VL)*s+k+i%VL];
	    }
	  }

	         #pragma acc loop vector
	  for(int j=0;j<BSIZE;j++){
	    for(int i=0;i<VL;i++){
	      xc[i+1][j]-=r1*xc[i-1+1][j];
	    }
	  }

	  for(int l=0;l<BSIZE;l+=NC){
	             #pragma acc loop vector
	    for(int i=0;i<BSIZE;i++){
	      x[(j+l+i/VL)*s+k+i%VL]=xc[i%VL+1][l+i/VL];
	    }
	  }

	         #pragma acc loop vector
	  for(int kk=0;kk<BSIZE;kk++){
	    xc[0][kk]=xc[VL][kk];
	  }

	}

      }

    }



#pragma acc parallel num_gangs(1) deviceptr(u)
{
    u[0]=1;
    for(int i=1;i<s;i++){
	u[i]=-r1*u[i-1];
    }
    us=u[s-1];
}


#pragma acc parallel num_gangs(1) deviceptr(es)
{
    es[s-1]=1./r2;
    for(int i=1;i<s;i++){
        es[s-i-1]=tmp1*es[s-i];
    }
    es0=es[0];
}





#pragma acc parallel num_gangs(1) present(x)
{
    for(int j=1;j<r;j++){
       x[(j+1)*s-1]-=r1*x[j*s-1]*us;
    }
}
#pragma acc parallel num_gangs(1) deviceptr(u)
{
    for(int j=1;j<r;j++){
       u[(j+1)*s-1]=-r1*u[j*s-1]*us;
    }
}




#pragma acc parallel present(x) deviceptr(u)
{
#pragma acc loop independent
    for(int j=1;j<r;j++){
        last1=x[j*s-1]*r1;
#pragma acc loop  independent
       for(int i=0;i<s-1;i++)
            x[j*s+i]-=last1*u[i];
    }

}

#pragma acc parallel deviceptr(u)
{
#pragma acc loop independent
    for(int j=1;j<r;j++){
        last3=u[j*s-1]*r1;
#pragma acc loop  independent
        for(int i=0;i<s-1;i++)
            u[j*s+i]=-last3*u[i];
    }
}




#pragma acc parallel present(x)
    {

      double xc[VL+1][BSIZE];
#pragma acc cache(xc)


         #pragma acc loop gang
      for(int j=0;j<r;j+=BSIZE){

	     #pragma acc loop vector
	for(int k=0;k<BSIZE;k++){
	  xc[VL][k]=0.0;
	}


	for(int k=s-VL;k>=0;k-=VL){


	  for(int l=0;l<BSIZE;l+=NC){
	             #pragma acc loop vector
	    for(int i=0;i<BSIZE;i++){
	      xc[i%VL][l+i/VL]=x[(j+l+i/VL)*s+k+i%VL];
	    }
	  }



	  #pragma acc loop vector
	  for(int j=0;j<BSIZE;j++){
	    for(int i=VL-1;i>=0;i--){
	      xc[i][j]=(xc[i][j] -t3*xc[i+1][j])/r2;
	    }
	  }

	  for(int l=0;l<BSIZE;l+=NC){
	             #pragma acc loop vector
	    for(int i=0;i<BSIZE;i++){
	      x[(j+l+i/VL)*s+k+i%VL]=xc[i%VL][l+i/VL];
	    }
	  }

	   #pragma acc loop vector
	  for(int kk=0;kk<BSIZE;kk++){
	    xc[VL][kk]=xc[0][kk];
	  }


	}

      }

    }






#pragma acc parallel num_gangs(1) present(x)
{
    for(int j=r-2;j>=0;j--){
       x[j*s]-=t3*x[(j+1)*s]*es0;
    }
}

#pragma acc parallel present(x) deviceptr(es)
{
#pragma acc loop independent
    for(int j=r-2;j>=0;j--){
        last2=t3*x[(j+1)*s];
#pragma acc loop  independent
        for(int i=1;i<s;i++)
            x[j*s+i]-=last2*es[i];
    }
}



#pragma acc parallel deviceptr(u)
    {

      double uc[VL+1][BSIZE];
#pragma acc cache(uc)



         #pragma acc loop gang
      for(int j=0;j<r;j+=BSIZE){

	     #pragma acc loop vector
	for(int k=0;k<BSIZE;k++){
	  uc[VL][k]=0.0;
	}


	for(int k=s-VL;k>=0;k-=VL){


	  for(int l=0;l<BSIZE;l+=NC){
	             #pragma acc loop vector
	    for(int i=0;i<BSIZE;i++){
	      uc[i%VL][l+i/VL]=u[(j+l+i/VL)*s+k+i%VL];
	    }
	  }



	  #pragma acc loop vector
	  for(int j=0;j<BSIZE;j++){
	    for(int i=VL-1;i>=0;i--){
	      uc[i][j]=(uc[i][j] -t3*uc[i+1][j])/r2;
	    }
	  }

	  for(int l=0;l<BSIZE;l+=NC){
	             #pragma acc loop vector
	    for(int i=0;i<BSIZE;i++){
	      u[(j+l+i/VL)*s+k+i%VL]=uc[i%VL][l+i/VL];
	    }
	  }

	   #pragma acc loop vector
	  for(int kk=0;kk<BSIZE;kk++){
	    uc[VL][kk]=uc[0][kk];
	  }


	}

      }

    }



#pragma acc parallel num_gangs(1) deviceptr(u)
{
    for(int j=r-2;j>=0;j--){
       u[j*s]-=t3*u[(j+1)*s]*es0;
    }
}

#pragma acc parallel deviceptr(es,u)
{
#pragma acc loop independent
    for(int j=r-2;j>=0;j--){
        last4=t3*u[(j+1)*s];
#pragma acc loop  independent
       for(int i=1;i<s;i++)
            u[j*s+i]-=last4*es[i];
    }
}
    acc_free(es);

#pragma acc parallel present(x[0:1]) deviceptr(u)
{
    x[0]/=(1+t3*r1*u[0]);
    tmp=t3*r1*x[0];
}
#pragma acc parallel present(x) deviceptr(u)
{
#pragma acc loop  independent
    for(int i=1;i<n;i++){
        x[i]=x[i]-tmp*u[i];
    }
}
    acc_free(u);

}

float fx(int i,float h, float p, float q){
    return p*cos(i*h)+(q-1)*sin(i*h);

}

int main(int argc, char **argv){
    int n=atoi(argv[1]);
    int r=atoi(argv[2]);

    double *b=malloc(sizeof (double)*n);
    double *x=malloc(sizeof (double)*n);
    double t;
    double l1=-10, l2=11, l3=-1;

  b[0]=x[0]=l2;
  for (int i=1;i<n;i++){
    x[i]=b[i]=0;
  }
    acc_init(acc_device_nvidia);
    if(fabs(l1)+fabs(l3)>fabs(l2)){
        printf("data error\n");
        return 1;
    }

    acc_init(acc_device_nvidia);

#pragma acc data copy(b[0:n])
{


#pragma acc data present(b)
{
    t=omp_get_wtime();
    tridiagonal(n,r,l1,l2,l3,b);
    t=omp_get_wtime()-t;
}
}
    printf("%.6lf",t);
    free(b);
    free(x);

    return 0;
}
