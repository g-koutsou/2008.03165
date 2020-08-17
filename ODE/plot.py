from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import numpy as np
import gvar as gv
import datetime

def sigmoids(params,times):
    n = len(params)
    ns = (n-1)//2
    c0 = params[0]
    cs = params[1::2]
    ms = params[2::2]
    def s(t):
        acc = 0
        for k in range(ns):
            acc += cs[k]*np.tanh(ms[k]*(t-times[k]))
        acc = acc + c0
        return acc
    return s

def model(params, C, ts, r, T):
    Nt = len(ts)
    beta = sigmoids(params, T)
    # Output array
    Y = np.zeros([Nt+1,4])
    r0 = r(ts[0])
    # Initial conditions
    Y[-1,0] = (C[1]-C[0])/(1-r0)
    Y[0,0] = (C[2]-C[1])/(1-r0)
    Y[0,1] = r0*C[0]/(1-r0)
    Y[0,2] = N-C[2]/(1-r0)
    Y[0,3] = C[0]
    # Time loop - all detected go into guarantine
    for i,t in enumerate(ts):
        Y[i+1,0] = beta(t)*Y[i,1]*Y[i,2]/N
        Y[i+1,1] = Y[i,1] + r(t)*Y[i+1-te,0]-r(t-th+te)*Y[i+1-th,0] # infectious
        Y[i+1,2] = Y[i,2] - Y[i+1,0]                                # susceptible
        Y[i+1,3] = Y[i,3] + (1-r(t))*Y[i+1-te,0]                    # confirmed
        if i+1==th:
            Y[i+1,1] -=Y[0,1]
    return Y

def twopop(Y0, ts, beta_mat, r):
    N = N1+N2
    T = len(ts)
    # Two-species
    Y = np.zeros([T+1,8])
    # Initial conditions
    Y[0,:] = Y0[-1,[0,0,1,1,2,2,3,3]]
    Y[-th:,:] = Y0[-th-1:-1,[0,0,1,1,2,2,3,3]]    
    Y[:,0::2] *= N1/N
    Y[:,1::2] *= N2/N
    for i,t in enumerate(ts):
        b = beta_mat(t)
        # Exposed
        Y[i+1,0] = (b[0,0]*Y[i,2]/N1 + b[0,1]*Y[i,3]/N2)*Y[i,4]
        Y[i+1,1] = (b[1,0]*Y[i,2]/N1 + b[1,1]*Y[i,3]/N2)*Y[i,5]
        #Infected
        Y[i+1,2] = Y[i,2] + r(t)*Y[i+1-te,0] - r(t+te-th)*Y[i+1-th,0]
        Y[i+1,3] = Y[i,3] + r(t)*Y[i+1-te,1] - r(t+te-th)*Y[i+1-th,1]
        # Susceptibles
        Y[i+1,4] = Y[i,4] - Y[i+1,0]
        Y[i+1,5] = Y[i,5] - Y[i+1,1]
        # Confirmed
        Y[i+1,6] = Y[i,6] + (1-r(t))*Y[i+1-te,0]
        Y[i+1,7] = Y[i,7] + (1-r(t))*Y[i+1-te,1]
    return Y

    
def R0(c):
    rho = c[1:] - c[:-1]
    t = len(rho)
    be = np.array([rho[i+1]/rho[i-th+1:i-te].sum() for i in range(th-1,t-1)])
    return be*(th-te)

# Return date as datetime
def getday(day,month):
    return datetime.datetime(2020,month,day)

#==============================================================================
#==============================================================================

# Get country data
N = 858_000
confirmed = np.loadtxt("data/cyprus-confirmed.txt")
Tc = len(confirmed)

day0 = getday(9,3) 
T0 = (getday(24,3) - day0).days # Beginning of lockdown
Tt = (getday(21,5) - day0).days # Testing inflection point
T1 = (getday(21,5) - day0).days # End of lockdown
T2 = (getday( 1,8) - day0).days # 1st of August. This is when we change scenarios

t_fit_ends = [128, 114, 84,]

te,th = 2,10+2
r0,r1,mr = 0.9,0.7,0.05
r = lambda x: 0.5*((r0+r1) + (r1-r0)*np.tanh(mr*(x-Tt)))

color_wheel = plt.rcParams['axes.prop_cycle'].by_key()['color']

fit_stan = dict()
for tf in t_fit_ends:
    tfit = (0, tf)
    fname = {
        84: "data/stan-params/stan-params-tfit084.txt",
        114:"data/stan-params/stan-params-tfit114.txt",
        128:"data/stan-params/stan-params-tfit128.txt",
    }[tf]
    fit_stan[tfit] = np.loadtxt(fname)
    pr = fit_stan[tfit]
    b2 = pr[:,0] + pr[:,1] + pr[:,3]
    b1 = b2 - 2*pr[:,3]
    b0 = b1 - 2*pr[:,1]
    xx = np.array([b0, b1, b2, pr[:,2], pr[:,4]]).T
    av = xx.mean(axis=0)
    hi = np.percentile(xx, 100-34, axis=0)
    lo = np.percentile(xx, 34, axis=0)
    up = hi-av
    dn = av-lo
    print(tfit)
    print(gv.gvar(av,up))
    print(gv.gvar(av,dn))
    print()

Tf = (getday(31, 12) - day0).days
fig = plt.figure(1, figsize=(6,6))
fig.clf()
gs = mpl.gridspec.GridSpec(12, 1, wspace=0.02, hspace=0.05, left=0.15, right=0.95)
ax = fig.add_subplot(gs[:7])
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: "{:4.1f}".format(x/1000)))
m0, = ax.plot(confirmed, marker="o", color="k", ms=4, mew=0.75, ls="", zorder=1, alpha=0.25)
labels = {m0: "Cyprus data"}
# for i,tf in enumerate(sorted(fit_stan.keys())):
for i,tf in enumerate(sorted([(0,128), (0,114), (0,84)])):
    t0,t1 = tf
    prms = fit_stan[tf]
    ts = np.arange(t0,Tf)
    da = confirmed[t0:t1]
    yi = np.array([model(x, da[:3], ts, r, (T0,T1)) for x in prms])
    m1, = ax.plot(ts[ts<t1], yi.mean(axis=0)[:-1,3][ts<t1], ls="-", color=color_wheel[i], zorder=-i-1)
    ax.plot(ts[ts>t1], yi.mean(axis=0)[:-1,3][ts>t1], ls="--", color=color_wheel[i], zorder=i+1)
    #.
    lo = np.percentile(yi, q=10, axis=0)
    hi = np.percentile(yi, q=90, axis=0)
    ax.fill_between(ts, lo[:-1,3], hi[:-1,3], color=color_wheel[i], alpha=0.2)    
    day = day0 + datetime.timedelta(t1)
    y,m,d = day.timetuple()[:3]
    labels[m1] = "Model, $t_{{f}}$ = {:02.0f}/{:02.0f}".format(d,m)
ax.set_xticklabels([])
ax.set_ylabel(r"Confirmed cases ($\times 10^3$)")
ax.legend(labels.keys(), labels.values(), loc="lower right", frameon=False)
ax.text(0.5*(T1 + T0), 1_190, "Lockdown", ha="center", va="top", fontsize=11)
ax.set_ylim(0, 1_330)
ax = fig.add_subplot(gs[7:10])
for i,tf in enumerate(sorted(fit_stan.keys())):
    t0,t1 = tf
    prms = fit_stan[tf]
    ts = np.arange(t0,Tf)
    bi = np.array([sigmoids(x, (T0, T1))(ts)*r(ts) for x in prms])
    ax.plot(ts, (bi*(th-te)).mean(axis=0), ls="-", color=color_wheel[i])    
    lo = np.percentile(bi*(th-te), q=10, axis=0)
    hi = np.percentile(bi*(th-te), q=90, axis=0)
    ax.fill_between(ts, lo, hi, color=color_wheel[i], alpha=0.2)    
    y = model(prms.mean(axis=0), da[:3], ts, r, (T0,T1))
    ax.plot(ts[th:], R0(y[:,3]), ls="--", color=color_wheel[i])
ax.plot([None],[None],ls="-",color="k",label=r"$R^{model}_0$")
ax.plot([None],[None],ls="--",color="k",label=r"$R^{integral}_0$")
ax.legend(loc="upper right", frameon=False, ncol=2)
ax.set_xticklabels([])
ax.set_ylim(-0.1, 1.9)
ax.set_ylabel("$R_0$")
ax = fig.add_subplot(gs[10:])
ax.plot(ts, r(ts), ls="-", color="k")
ax.set_ylim(0, 1.1)
ax.set_ylabel("r(t)")
for ax in fig.axes:
    ax.fill_betweenx([-0.1, 1_700], [T0, T0], [T1, T1], color=color_wheel[0], alpha=0.1, zorder=-2, lw=0)
    ax.set_xlim(0, Tf)
ax = fig.axes[-1]
xt = ax.get_xticks()
f = [(lambda x: "{2:02.0f}/{1:02.0f}\n{0}".format(*(x.timetuple()[:3])))(day0 + datetime.timedelta(d)) for d in xt]
ax.set_xticklabels(f)
fig.canvas.draw()
fig.show()

Tf = (getday(31, 12) - day0).days
ts = np.arange(Tf)
tfit = (0, 128)
scenarios = []
r = lambda x: 0.5*((r0+r1) + (r1-r0)*np.tanh(mr*(x-Tt)))
prms_1 = fit_stan[tfit]
scenarios.append(
    {
        "label": "A",
        "data": (ts, np.array([model(x, da[:3], ts, r, (T0,T1))[:-1,3] for x in prms_1])),
        "beta": [sigmoids(prms_1.mean(axis=0), (T0,T1,T2))],
        "r": r,
    }
)
r2 = 0.5
r = lambda x: 0.5*((r0+r2) + (r1-r0)*np.tanh(mr*(x-Tt)) + (r2-r1)*np.tanh(mr*(x-T2)))
scenarios.append(
    {
        "label": "B",
        "data": (ts, np.array([model(x, da[:3], ts, r, (T0,T1))[:-1,3] for x in prms_1])),
        "beta": [sigmoids(prms_1.mean(axis=0), (T0,T1,T2))],
        "r": r,
    }
)
b2 = prms_1[:,0]+prms_1[:,1]+prms_1[:,3]
m0,m1 = prms_1[:,2],prms_1[:,4]
b1 = b2 - 2*prms_1[:,3]
b0 = b1 - 2*prms_1[:,1]
# Used in third scenario
b3 = 0.8*b2
m2 = 0.1*m1**0
prms_2 = (np.array([(b3+b0)/2, (b1-b0)/2, m0, (b2-b1)/2, m1, (b3-b2)/2, m2])).T
r = lambda x: 0.5*((r0+r1) + (r1-r0)*np.tanh(mr*(x-Tt)))
scenarios.append(
    {
        "label": "C",
        "data": (ts, np.array([model(x, da[:3], ts, r, (T0,T1,T2))[:-1,3] for x in prms_2])),
        "beta": [sigmoids(prms_2.mean(axis=0), (T0,T1,T2))],
        "r": r,
    }
)
# Used in fourth scenario
r = lambda x: 0.5*((r0+r1) + (r1-r0)*np.tanh(mr*(x-Tt)))
N1 = np.round(0.2*N)
N2 = N-N1
b3 = 0.05*b2**0
m2 = 0.1*m1**0
prms_4a = np.array(prms_1)
prms_4b = np.array([(b3+b0)/2, (b1-b0)/2, m0, (b2-b1)/2, m1, (b3-b2)/2, m2]).T
b00 = [sigmoids(x, [T0, T1, T2]) for x in prms_4b]
b01 = [sigmoids(x, [T0, T1, T2]) for x in prms_4b]
b10 = [sigmoids(x, [T0, T1, T2]) for x in prms_4b]
b11 = [sigmoids(x, [T0, T1, T2]) for x in prms_4a]
beta_mat = [lambda x: np.array([[N1/N*b_00(x), N2/N*b_01(x)], [N1/N*b_10(x), N2/N*b_11(x)]]) for b_00,b_01,b_10,b_11 in zip(b00,b01,b10,b11)]
ts0 = np.arange(Tc)
y0 = [model(x, da[:3], ts0, r, (T0,T1)) for x in prms_1]
ts1 = np.arange(Tc,Tf)
y1 = [twopop(y, ts1, b, r) for y,b in zip(y0, beta_mat)]
scenarios.append(
    {
        "label": "D",
        "data": (np.concatenate((ts0,ts1)),
                 np.concatenate((np.array(y0)[:,:-1,3], np.array(y1)[:,:-1,6:8].sum(axis=2)),axis=1)),
        "beta": [sigmoids(prms_4a.mean(axis=0), [T0, T1, T2]), sigmoids(prms_4b.mean(axis=0), [T0, T1, T2])],
        "r": r,
    }
)

descr = ["Continue with fitted $R^{model}_0(t)$\n to end of year",
         "Reduce ratio of undetected \nto $\\sim50\%$",
         "Reduce $R^{model}_0(t)$ to $\\sim$1\non August 1$^{st}$",
         "Impose strict lockdown on\n20% of population"]

n_scen = len(scenarios)
fig = plt.figure(2, figsize=(1+4*n_scen//2,11.5))
fig.clf()
gs = mpl.gridspec.GridSpec(2*14, n_scen//2, wspace=0.075, left=0.075, right=0.95, bottom=0.12, top=0.95)
for i,sc in enumerate(scenarios):
    col = i%2
    row = i//2
    ax = fig.add_subplot(gs[row*14:row*14+8, col])
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x,_: "{:4.1f}".format(x/1000)))
    label = sc["label"]
    data = sc["data"]
    m0, = ax.plot(confirmed, marker="o", color="k", ms=4, ls="", mew=0, alpha=0.5, zorder=100)
    m1, = ax.plot(data[0], data[1].mean(axis=0), color=color_wheel[0], ls="-")
    hi = np.percentile(data[1], 10, axis=0)
    lo = np.percentile(data[1], 90, axis=0)
    ax.fill_between(data[0], lo, hi, color=color_wheel[0], alpha=0.1)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1_330)
    if col != 0:
        ax.set_yticklabels([])
    ax.fill_betweenx([0, 1_330], [tfit[0]]*2, [tfit[1]]*2, color="k", alpha=0.1, lw=0, zorder=-1)
    ax.text(np.mean(tfit), 1_220, "Fit", ha="center", va="top", fontsize=11)
    ax.text(np.mean([tfit[1], Tf]), 1_220, "Prediction", ha="center", va="top", fontsize=11)
    ax.set_xlim(0, Tf+1)
    if i == 0:
        ax.legend([m0, m1], ["Cyprus data", "Extended SEIQR model"], loc="lower right", frameon=False)
    if col == 0:
        ax.set_ylabel(r"Confirmed cases ($\times 10^3$)")
    ax.text(np.mean([tfit[1], Tf]), 800, "Scenario {}\n{}".format("ABCD"[i], descr[i]),
            ha="center", va="top", fontsize=11)
for i,sc in enumerate(scenarios):
    col = i%2
    row = i//2
    ax = fig.add_subplot(gs[row*14+8:row*14+11, col])
    label = sc["label"]
    ts = sc["data"][0]    
    beta = sc["beta"]
    for j,b in enumerate(beta):
        ls = ["-","-."][j]
        ax.plot(ts, r(ts)*b(ts)*(th-te), ls=ls, color=color_wheel[0], label=r"$R_0^{model}$")
    rr = R0(np.array(sc["data"][1]).mean(axis=0))
    ax.plot(ts[th+1:], rr, ls="--", color=color_wheel[1], label="$R_0^{integral}$")    
    ax.plot(ts, ts**0, ls="--", color="k")
    ax.set_ylim(-0.1, 2.9)
    if col != 0:
        ax.set_yticklabels([])
    ax.set_xlim(0, Tf+1)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel(r"$R_0$")
    if i == 0:
        ax.legend(loc="upper center", frameon=False, ncol=2)
for i,sc in enumerate(scenarios):
    col = i%2
    row = i//2
    ax = fig.add_subplot(gs[row*14+11:row*14+13, col])
    label = sc["label"]
    r = sc["r"]
    ts = sc["data"][0]    
    ax.plot(ts, sc["r"](ts), ls="-", color="k")    
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, Tf+1)
    if col != 0:
        ax.set_yticklabels([])
    xt = ax.get_xticks()
    f = [(lambda x: "{2:02.0f}/{1:02.0f}\n{0}".format(*(x.timetuple()[:3])))(day0 + datetime.timedelta(d)) for d in xt]
    f[0] = "" if i != 0 else f[0]
    ax.set_xticklabels(f)
    if col == 0:
        ax.set_ylabel("r(t)")
    if row == 0:
        ax.set_xticklabels([])
fig.canvas.draw()
fig.show()
