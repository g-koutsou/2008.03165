import numpy as np
import datetime
import pystan

chains = 2
iter = 2000
warmup = 1000
thin = 5

start_t = datetime.datetime.now()
print(" START time: {}".format(start_t))

model_code = """
    functions{  

      real beta(int t, vector params, vector times, int N_par){
        int n;
        int ns;
        real c0;
        real cs[2];
        real ms[2];
        real acc;
        n=N_par;
        ns=2;   //(n-1)/2;
        c0=params[1];
        cs[1]=params[2]; 
        cs[2]=params[4];
        ms[1]=params[3];
        ms[2]=params[5];
        acc=0;
        for (k in 1:ns){
          acc = acc+cs[k]*tanh(ms[k]*(t-1-times[k]));
        } 
        acc = acc+c0;
        return acc;
      }

      real r(int t,vector T,vector rs){
        return 0.5*((rs[1]+rs[2])+(rs[2]-rs[1])*tanh(0.05*(t-1-T[2])));
      }
   
      real delta(real x1,real x2){
        if (x1==x2) return 1;
        else return 0;
      }
    }
  
    data {     
      int N;                     
      int th;
      int te;
      int N_data;
      vector[N_data] C;
      int N_T;
      vector[N_T] T;
      int N_par;
      vector[N_par] pri;
      vector[2] rs;
      int ts[N_data];
    }           
                                                                    
    parameters { 
      vector[N_par] params;  
      real<lower=0> sigma;
    } 
    

                                                                    
    transformed parameters {
      matrix[N_data+1,4] Y = rep_matrix(0,N_data+1,4);
      real r0;
      real Y_1;
      vector[N_par] s_par;

      s_par[1]=pri[1]/2;
      s_par[2]=fabs(pri[2])/2;
      s_par[3:5]=pri[3:5]/2;

      r0=r(1,T,rs);

      Y_1 = (C[2]-C[1])/(1-r0);
      Y[1,1] = (C[3]-C[2])/(1-r0);
      Y[1,2] = r0*C[1]/(1-r0);
      Y[1,3] = N-C[3]/(1-r0);
      Y[1,4] = C[1];
   
      for (i in ts[1]:(ts[N_data])){
        int t=ts[i];
        Y[i+1,1] = beta(t,params,T,N_par)*Y[i,2]*Y[i,3]/N;
        if (i+1-th>0){
          Y[i+1,2] = Y[i,2] + r(t,T,rs)*Y[i+1-te,1]-r(t-th+te,T,rs)*Y[i+1-th,1]; // infectious
          Y[i+1,3] = Y[i,3] - Y[i+1,1];                                          // susceptible
          Y[i+1,4] = Y[i,4] + (1-r(t,T,rs))*Y[i+1-te,1];                         // confirmed
        }
        else if (i+1-te>0){
          Y[i+1,2] = Y[i,2] + r(t,T,rs)*Y[i+1-te,1]-r(t-th+te,T,rs)*Y_1*delta(i+1,th);
          Y[i+1,3] = Y[i,3] - Y[i+1,1]; 
          Y[i+1,4] = Y[i,4] + (1-r(t,T,rs))*Y[i+1-te,1];     
        }
        else{
          Y[i+1,2] = Y[i,2] + r(t,T,rs)*Y_1*delta(i+1,te)-r(t-th+te,T,rs)*Y_1*delta(i+1,th);
          Y[i+1,3] = Y[i,3] - Y[i+1,1];
          Y[i+1,4] = Y[i,4] + (1-r(t,T,rs))*Y_1*delta(i+1,te);
        }
        if (i+1==th+1){
          Y[i+1,2]=Y[i+1,2]-Y[1,2];
        }
      }
    }
      
    model {
      sigma~normal(100,100);
      params[1:N_par]~normal(pri[1:N_par],s_par);
      C[ts[1]:N_data]~normal(Y[ts[1]:N_data,4],sigma); 
    }                       
                                                        
"""

#data
confirmed = np.loadtxt("../data/cyprus-confirmed.txt")
tfit = 128
priors = {
    84: [0.35039643,-0.26840509,0.08060385,0.06355486,0.01815248],
    114: [0.35313028,-0.26776621,0.0804449 ,0.0622391 ,0.02127642],
    128: [0.3582084,-0.27300769,0.07909936,0.07703217,0.01669354]
}[tfit]

N_pop=858_000
te = 2
th = 10 + 2

T0 = 15
T1 = 73
T2 = 145
T = [T0,T1,T2]
ts = np.arange(0, tfit)
da = confirmed[ts]
Nc = len(da)
N_par = len(priors)
rs = [0.9, 0.7]

#fit
data = dict(
    N=N_pop,
    th=th,
    te=te,
    C=da,
    N_data=Nc,
    N_T=len(T),
    T=T,
    pri=priors,
    rs=rs,
    ts=ts+1,
    N_par=N_par
)
model = pystan.StanModel(model_code=model_code)
fit = model.sampling(data=data, iter=iter, warmup=warmup, chains=chains, thin=thin)
end_t = datetime.datetime.now()
print(" END time: {}, took: {}".format(end_t, end_t-start_t))
