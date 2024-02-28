#include <stdio.h>
#include <math.h>


float f1(float x) {
    return x*x;
}
float f2(float x) {
    return exp(x);
}

float f3(float x) {
    return sin(x);
}

float f4(float x) {
    return cos(x);
}

float rectangle(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;
    float sum = 0;
    float x = a + (b-a)/(2*n);
    for (int i = 0; i < n; i++) {
        sum += f(x);
        x += dx;
    }
    return sum*dx;
}
float rectangleMulti(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;
    float sum = 0;
    float x;

    for (int i = 0; i < n; i++) {

        x=a+i*dx;
        sum+=f(x);
    }
    return sum*dx;
}
float rectangleKahan(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;
    float x = a + (b-a)/(2*n);

    float sum = 0.0;
    float e2 = 0.0;
    float tmp,y;

    for (int i = 0; i < n; i++) {
        //Kahan
        sum+=f(x);


        //Kahan
        //x +=dx;
        tmp=x;
        y=dx +e2;
        x=tmp + y;
        e2 = (tmp - x) +y;
    }

    return sum*dx;
}
float rectangleKahanx2(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;
    float x = a + (b-a)/(2*n);

    float sum = 0.0;
    float e1 = 0.0, e2 = 0.0;
    float tmp,y;

    for (int i = 0; i < n; i++) {
        //Kahan
        //sum+=f(x);
        tmp = sum;
        y = f(x) + e1;
        sum = tmp +y;
        e1 = (tmp - sum) +y;

        //Kahan
        //x +=dx;
        tmp=x;
        y=dx +e2;
        x=tmp + y;
        e2 = (tmp - x) +y;
    }

    return sum*dx;
}
float rectangleGM(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;


    float x = a + (b-a)/(2*n);
    float sum=0, p=0, sold=0;

    for (int i = 0; i < n; i++) {
        //G-M
        //sum+=f(x);
        float term=f(x);
        sum=sold+term;
        float tmp=sum-sold;
        float tmp2=term-tmp;
        p=p+tmp2;
        sold=sum;

        //Kahan
        x +=dx;

    }

    return (sum+p)*(dx);
}
float rectangleGM_Kahan(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;


    float x = a + (b-a)/(2*n);
    float sum=0, p=0,p2=0, sold=0, tmp3,y;

    for (int i = 0; i < n; i++) {
        //G-M
        //sum+=f(x);
        float term=f(x);
        sum=sold+term;
        float tmp=sum-sold;
        float tmp2=term-tmp;
        p=p+tmp2;
        sold=sum;

        //Kahan
        //x +=dx;
        tmp3=x;
        y=dx +p2;
        x=tmp3 + y;
        p2 = (tmp3 - x) +y;
    }

    return (sum+p)*(dx);
}



float trapezoidal(int n, float a, float b, float(*f)(float)){
    float dx = (b-a)/n;
    float sum = 0;
    float a_base = f(a), b_base,x=a;
    for(int i=0;i<n;i++){
        x+=dx;
        b_base = f(x);
        sum += (a_base+b_base);
        a_base = b_base;
    }
    return 0.5*sum*dx;
}
float trapezoidalMulti(int n, float a, float b, float(*f)(float)){
    float dx = (b-a)/(float)n;
    float S = 0.0;
    float podstawa_a = f(a), podstawa_b,x=a;

    for(int i=1;i<=n;i++)
    {
        x=a+i*dx;
        podstawa_b = f(x);
        S += (podstawa_a+podstawa_b);
        podstawa_a = podstawa_b;
    }
    return S*0.5*dx;
}
float trapezoidalKahan(int n, float a, float b, float(*f)(float)){
    float h = (b-a)/(float)n;
    float S = 0.0;
    float podstawa_a = f(a), podstawa_b, x=a;

    float e=0.0, tmp,y;
    for(int i=1;i<=n;i++)
    {

        //x+=h;
        tmp=x;
        y=h+e;
        x=tmp+y;
        e=(tmp-x)+y;


        podstawa_b = f(x);

        S += (podstawa_a+podstawa_b);


        podstawa_a = podstawa_b;
    }
    return S*0.5*h;
}
float trapezoidalKahanx2(int n, float a, float b, float(*f)(float)){
    float h = (b-a)/(float)n;
    float S = 0.0;
    float podstawa_a = f(a), podstawa_b, x=a;

    float e=0.0, e1=0.0, tmp,y;
    for(int i=1;i<=n;i++)
    {

        //x+=h;
        tmp=x;
        y=h+e;
        x=tmp+y;
        e=(tmp-x)+y;


        podstawa_b = f(x);

        //S += (podstawa_a+podstawa_b);
        tmp=S;
        y = (podstawa_a+podstawa_b) + e1;
        S = tmp +y;
        e1 = (tmp - S) +y;

        podstawa_a = podstawa_b;
    }
    return S*0.5*h;
}
float trapezoidalGM(int n, float a, float b, float(*f)(float)){
    float h = (b-a)/(float)n;
    float S = 0.0;
    float podstawa_a = f(a), podstawa_b, x=a;

    float tmp,sold=0.,p=0.;
    for(int i=1;i<=n;i++)
    {

        x+=h;



        podstawa_b = f(x);

        //S += (podstawa_a+podstawa_b);
        float term =podstawa_a+podstawa_b;

        S = sold + term;
        tmp = S - sold;
        float tmp2= term - tmp;
        p = p +tmp2;
        sold = S;

        podstawa_a = podstawa_b;
    }
    return (S+p)*0.5*h;
}
float trapezoidalGM_Kahan(int n, float a, float b, float(*f)(float)){
    float h = (b-a)/(float)n;
    float S = 0.0;
    float podstawa_a = f(a), podstawa_b, x=a;

    float e=0.0,tmp,y,sold=0.,p=0.;
    for(int i=1;i<=n;i++)
    {

        //x+=h;
        tmp=x;
        y=h+e;
        x=tmp+y;
        e=(tmp-x)+y;


        podstawa_b = f(x);

        //S += (podstawa_a+podstawa_b);
        float term =podstawa_a+podstawa_b;
        S = sold + term;
        tmp=S - sold;
        float tmp2= term - tmp;
        p = p +tmp2;
        sold = S;

        podstawa_a = podstawa_b;
    }
    return (S+p)*0.5*h;
}


float simpson(int n, float a, float b, float(*f)(float)){
    float h = (b - a) / n;
    float sum1 = f(a+h/2), sum2=0.,x=a;

    for(int i=1;i<n;i++){
        x+=h;
        sum1+=f(x+h/2);
        sum2+=f(x);
    }
    return h/6*(f(a)+f(b) + 4*sum1+2*sum2);
}
float simpsonMulti(int n, float a, float b, float(*f)(float)){
    float h = (b - a) / n;
    float sum1 = f(a+h/2), sum2=0.,x=a;

    for(int i=1;i<n;i++){
        x=a+i*h;
        sum1+=f(x+h/2);
        sum2+=f(x);
    }
    return h/6*(f(a)+f(b) + 4*sum1+2*sum2);
}
float simpsonKahan(int n, float a, float b, float(*f)(float)){
    float h = (b - a) / n;
    float sum1 = f(a+h/2), sum2=0.,x=a;
    float tmp, e=0,y;

    for(int i=1;i<n;i++){
        //x+=h;
        tmp=x;
        y=h +e;
        x=tmp + y;
        e = (tmp - x) +y;

        sum1+=f(x+h/2);


        sum2+=f(x);
    }
    return h/6*(f(a)+f(b) + 4*sum1+2*sum2);

}
float simpsonKahanx2(int n, float a, float b, float(*f)(float)){
    float h = (b - a) / n;
    float sum1 = f(a+h/2), sum2=0.,x=a;
    float tmp, e0=0, e1=0,y;

    for(int i=1;i<n;i++){
        //x+=h;
        tmp=x;
        y=h +e0;
        x=tmp + y;
        e0 = (tmp - x) +y;

        //sum1+=f(x+h/2);
        float term = f(x+h/2);
        tmp = sum1;
        y = term + e1;
        sum1 = tmp  + y;
        e1 = (tmp - sum1) + y;


        sum2+=f(x);
    }
    return h/6*(f(a)+f(b) + 4*sum1+2*sum2);
}
float simpsonKahanx3(int n, float a, float b, float(*f)(float)){
    float h = (b - a) / n;
    float sum1 = f(a+h/2), sum2=0.,x=a;
    float tmp, e0=0, e1=0, e2=0,y;
    float term, term1;

    for(int i=1;i<n;i++){
        //x+=h;
        tmp=x;
        y=h +e0;
        x=tmp + y;
        e0 = (tmp - x) +y;

        //sum1+=f(x+h/2);
        term = f(x+h/2);
        tmp = sum1;
        y = term + e1;
        sum1 = tmp  + y;
        e1 = (tmp - sum1) + y;


        //sum2+=f(x);
        term1 = f(x);
        tmp = sum2;
        y = term1 + e2;
        sum2 = tmp  + y;
        e2 = (tmp - sum2) + y;

    }
    return h/6*(f(a)+f(b) + 4*sum1+2*sum2);
}
float simpsonGM(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;
    float sum = 0.0, st=0.,x=a;
    float sold=0,p=0;


    for(int i=1;i<=n;i++)
    {
        x+=dx;

        st+=f(x-dx/2);

        if(i<n){
            //sum+=f(x);
            float term=f(x);
            sum=sold+term;
            float tmp=sum-sold;
            float tmp2=term-tmp;
            p=p+tmp2;
            sold=sum;

        }
    }
    return dx/6 * (f(a)+f(b)+2*(sum+p) + 4*st);
}

float simpsonGMx2(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;
    float sum = 0.0, st=0.,x=a;
    float sold=0,p=0;
    float sold2=0,p2=0;


    for(int i=1;i<=n;i++)
    {
        x+=dx;


        //st+=f(x-dx/2);
        float term=f(x-dx/2);
        st=sold2+term;
        float tmp=st-sold2;
        float tmp2=term-tmp;
        p2=p2+tmp2;
        sold2=st;


        if(i<n){
            //sum+=f(x);
            float term=f(x);
            sum=sold+term;
            float tmp=sum-sold;
            float tmp2=term-tmp;
            p=p+tmp2;
            sold=sum;

        }
    }
    return dx/6 * (f(a)+f(b)+2*(sum+p) + 4*(st+p2));
}


float simpsonGM_Kahan(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;
    float sum = 0.0, st=0.,x=a;
    float tmp,e2=0.,y;
    float sold=0,p=0;


    for(int i=1;i<=n;i++)
    {
        //x+=dx;
        tmp=x;
        y=dx +e2;
        x=tmp + y;
        e2 = (tmp - x) +y;

        st+=f(x-dx/2);

        if(i<n){
            //sum+=f(x);
            float term=f(x);
            sum=sold+term;
            float tmp=sum-sold;
            float tmp2=term-tmp;
            p=p+tmp2;
            sold=sum;

        }
    }
    return dx/6 * (f(a)+f(b)+2*(sum+p) + 4*st);
}

float simpsonGMx2_Kahan(int n, float a, float b, float(*f)(float)){
    float dx = (b - a) / n;
    float sum = 0.0, st=0.,x=a;
    float tmp,e2=0.,y;
    float sold=0,p=0;
    float sold2=0,p2=0;


    for(int i=1;i<=n;i++)
    {
        //x+=dx;
        tmp=x;
        y=dx +e2;
        x=tmp + y;
        e2 = (tmp - x) +y;

        st+=f(x-dx/2);


        //st+=f(x-dx/2);
        float term=f(x-dx/2);
        st=sold2+term;
        float tmp=st-sold2;
        float tmp2=term-tmp;
        p2=p2+tmp2;
        sold2=st;


        if(i<n){
            //sum+=f(x);
            float term=f(x);
            sum=sold+term;
            float tmp=sum-sold;
            float tmp2=term-tmp;
            p=p+tmp2;
            sold=sum;

        }
    }
    return dx/6 * (f(a)+f(b)+2*(sum+p) + 4*(st+p2));
}





int main() {

    int n = 100;
    float (*f_arr[4])(float) ={f1,f2,f3,f4};
    float a_arr[4]={0.,0.,0.,0.};
    float b_arr[4]={1.,1.,2*M_PI,M_PI/2};
    float base_arr[4]={1./3,exp(1)-1,0,1};
    float sum;

    FILE *f = fopen("100.csv", "w");
    if (f == NULL) {
        perror("Nie udalo sie otworzyc pliku 'result.csv' do zapisu");
        return 1;
    }
    fprintf(f,"n=%d\n",n);
    fprintf(f,"\tMetoda_prostokątów\t\t\t\t\t\tMetoda_trapezów\t\t\t\t\t\tMetoda_Simpsona\n");
    fprintf(f,"Wynik_dokładny\tZwykłe_sumowanie\tSumowanie_mnożenie\tSumowanie_Kahanax1\tSumowanie_Kahanax2\tSumowanie_GM\tSumowanie_GM+Kahan\t");
    fprintf(f,"Zwykłe_sumowanie\tSumowanie_mnożenie\tSumowanie_Kahanax1\tSumowanie_Kahanax2\tSumowanie_GM\tSumowanie_GM+Kahan\t");
    fprintf(f,"Zwykłe_sumowanie\tSumowanie_mnożenie\tSumowanie_Kahanax1\tSumowanie_Kahanax2\tSumowanie_Kahanax3"
               "\tSumowanie_GM\tSumowanie_GMx2\tSumowanie_GM+Kahan\tSumowanie_GMx2+Kahan\n");
    for(int i=0;i<4;i++){

        float base = base_arr[i];
        fprintf(f,"%e\t",base_arr[i] );


        //fprintf(f,"\n-----------Metoda prostokątów---------------\n");
        sum = rectangle(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));

        sum = rectangleMulti(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));

        sum = rectangleKahan(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = rectangleKahanx2(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = rectangleGM(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = rectangleGM_Kahan(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));



        //fprintf(f,"\n-----------Metoda trapezów---------------\n");
        sum = trapezoidal(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));

        sum = trapezoidalMulti(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));

        sum = trapezoidalKahan(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = trapezoidalKahanx2(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = trapezoidalGM(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = trapezoidalGM_Kahan(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));



        //fprintf(f,"\n-----------Metoda Simpsona---------------\n");
        sum = simpson(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));

        sum = simpsonMulti(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));

        sum = simpsonKahan(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = simpsonKahanx2(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = simpsonKahanx3(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = simpsonGM(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = simpsonGMx2(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t", fabs(sum-base));

        sum = simpsonGM_Kahan(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\t",fabs(sum-base));

        sum = simpsonGMx2_Kahan(n,a_arr[i],b_arr[i],f_arr[i]);
        fprintf(f,"%e\n",fabs(sum-base));
    }
    return 0;
}
