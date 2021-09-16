#include<iostream>
#include<stdlib.h>
#include<vector>
#include<string>
#include<cmath>
#include<math.h>
#include<random>
#include<complex.h>
#include<omp.h>

//M_PI as pi

using namespace std;
int eps =1;

long long N=20000;



int index(long g){
    g = abs(g);
    if(g%2==1){return 0;}
    if(g%(2^20)==0){return 20;}
    if(g%(2^19)==0){return 19;}
    if(g%(2^18)==0){return 18;}
    if(g%(2^17)==0){return 17;}
    if(g%(2^16)==0){return 16;}
    if(g%(2^15)==0){return 15;}
    if(g%(2^14)==0){return 14;}
    if(g%(2^13)==0){return 13;}
    if(g%(2^12)==0){return 12;}
    if(g%(2^11)==0){return 11;}
    if(g%(2^10)==0){return 10;}
    if(g%(2^9)==0){return 9;}
    if(g%(2^8)==0){return 8;}
    if(g%(2^7)==0){return 7;}
    if(g%(2^6)==0){return 6;}
    if(g%(2^5)==0){return 5;}
    if(g%(2^4)==0){return 4;}
    if(g%(2^3)==0){return 3;}
    if(g%(4)==0){return 2;}
    if(g%(2)==0){return 1;}
}

int main(){
#pragma omp parallel

{
double w=M_PI;
double a =1/sqrt(2.0); complex<double> b =I/sqrt(2.0);
vector<complex<double>> psi_0(2*N+1,0);
vector<complex<double>> psi_1(2*N+1,0);
vector<complex<double>> disorder(2*N+1,1);
vector<long> pow_2(21,0);
//for(int i=0; i<21;i++){cout<<pow_2[i]<<endl;}
vector<long double> avg_disorder(20,0.0);
int j=0;
//long n,l,m,k;
psi_0[N]=a;psi_1[N]=b;
default_random_engine generator;
uniform_real_distribution<double> distr(-w,w);
double norm=0;
double sum_0,sum_1;

for(long l=0;l<2*N+1;l++){

    disorder[l]=cexp(I*distr(generator)*M_PI);
    
}
cout<<"Disorder Value is: "<<abs(disorder[N])<<endl;
for (int k=0;k<21;k++){
    pow_2[k]+=pow(2,(k));
    //cout<< pow_2[k]<<endl;

}


for(long n=1;n<N+1;n++){
   
    if(n==10000){cout<<"progress"<<endl;}
    if(n==20000){cout<<"progress"<<endl;}
    if(n==30000){cout<<"progress"<<endl;}
    if(n==40000){cout<<"progress"<<endl;}
    if(n==50000){cout<<"progress"<<endl;}
    if(n==60000){cout<<"progress"<<endl;}
    if(n==70000){cout<<"progress"<<endl;}
    if(n==80000){cout<<"progress"<<endl;}
    if(n==90000){cout<<"progress"<<endl;}
    if(n==100000){cout<<"done"<<endl;}

    for(int m=-n;m<n+1;m++){
        complex<double>help_0=(0,0);
        complex<double>help_1=(0,0);
        //cout<<psi_1[N+m]<<endl;
        // if (n==1){ cout<<disorder[N+m]*psi_1[N+m]<<endl;}
        help_0=disorder[N+m]*psi_0[N+m]*sin(pow(eps,index(m))*0.25*M_PI)+disorder[N+m]*psi_1[N+m]*cos(pow(eps,index(m))*0.25*M_PI);
        //cout<<abs(psi_0[N+m])<<endl;
        help_1=disorder[N+m]*psi_0[N+m]*cos(pow(eps,index(m))*0.25*M_PI)-disorder[N+m]*psi_1[N+m]*sin(pow(eps,index(m))*0.25*M_PI);

        psi_0[N+m]=help_0;
        psi_1[N+m]=help_1;
        //cout<<abs(psi_1[N+m])<<endl;
    }
    //if(n==10){
    //for(int m=-n;m<n+1;m++){
    //  cout<<psi_1[N+m]<<endl;
    //    cout<<psi_0[N+m]<<endl;
    //}
    //}
     
   
    for(int m=-N;m<N+1;m++){ complex<double> temp_0=(0,0),temp_1=(0,0);
        temp_0=psi_0[N-m-1];
        //cout<<temp_0<<endl;
        psi_0[N-m]=temp_0;
        //if(n==1&&m==0){cout<<temp_0<<endl;}
        //if(n==1&&m==0){cout<<psi_0[N+m-1]<<endl;}
        //if(n==1&&m==0){cout<<psi_0[N+m+1]<<endl;}
        temp_1= psi_1[N+m];
        psi_1[N+m-1]=temp_1;
        }
     
    if(n==pow_2[j]){
        //cout<<j<<endl;
        for(int m=-n;m<n;m++){
        norm=pow(abs(psi_0[N+m]),2)+pow(abs(psi_1[N+m]),2);  
        //cout<<abs(psi_0[N+m])<<endl;
        //cout<<abs(psi_1[N+m])<<endl;
        //cout<<pow(abs(psi_0[N+m]),2)<<endl;
        //cout<<pow(abs(psi_1[N+m]),2)<<endl;
        sum_0+=norm*(pow(m,2));
        //cout<<sum_0<<endl;
        sum_1+=(pow(norm,2)*pow(m,2));
        //cout<<sum_1<<endl;}
        }
    
    //cout<<sum_0<<endl;
    //cout<<sum_1<<endl;
    avg_disorder[j]+=(pow(abs(sum_0-sum_1),0.5));
    //cout<<avg_disorder[j]<<endl;
    sum_1=sum_0=0;
    norm=0;
    j+=1;}
    
}
for(int i=0;i<21;i++){
  cout<<avg_disorder[i]<<endl;
}
return  0;
}

}