import numpy as np
from algorithms import LinUCB, RegretBalancingElim
from utils import FiniteContLinearRep, ContBanditProblem, make_random_rep, make_newlinrep_reshaped
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

def execute_regbalelim(T, true_rep, reps, reg_val, noise_std, delta, num):
    algo = RegretBalancingElim(reps, reg_val, noise_std, delta)
    problem = ContBanditProblem(true_rep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print(f"--- {num} finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()
    return reg

def execute_linucb(T, true_rep, rep, reg_val, noise_std, delta, num):
    algo = LinUCB(rep, reg_val, noise_std, delta)
    problem = ContBanditProblem(true_rep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print(f"--- {num} finished in {np.round(time.time() - start_time,2)} seconds ---")
    reg = problem.exp_instant_regret.cumsum()
    return reg

PARALLEL = True
NUM_CORES = 6
NRUNS = 5
nc, na, nd = 200, 20, 20
noise_std = 0.3
reg_val = 1
delta = 0.01
T = 50000

fig, ax = plt.subplots(1, 3, figsize=(15,5))

b = np.load('final_representation2.npz')
reps = []
for i in range(b['true_rep']+1):
    feat = b[f'reps_{i}']
    param = b[f'reps_param_{i}']
    reps.append(FiniteContLinearRep(feat, param))
linrep = reps[b['true_rep']]

print("Running algorithm RegretBalancingElim")
results = []
if PARALLEL:
    results = Parallel(n_jobs=NUM_CORES)(
        delayed(execute_regbalelim)(T, linrep, reps, reg_val, noise_std, delta, i) for i in range(NRUNS)
    )
else:
    for n in range(NRUNS):
        results.append(
            execute_regbalelim(T, linrep, reps, reg_val, noise_std, delta, n)
        )
regrets = []
for el in results:
    regrets.append(el.tolist())
regrets = np.array(regrets)
mean_regret = regrets.mean(axis=0)
std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])

ax[0].plot(mean_regret, label="RegBalElim")
ax[0].fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)

ax[1].plot(mean_regret, label="RegBalElim")
ax[1].fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)

print("Running algorithm LinUCB")
for nf, f in enumerate(reps):

    results = []
    if PARALLEL:
        results = Parallel(n_jobs=NUM_CORES)(
            delayed(execute_linucb)(T, linrep, f, reg_val, noise_std, delta, i) for i in range(NRUNS)
        )
    else:
        for n in range(NRUNS):
            results.append(
                execute_linucb(T, linrep, f, reg_val, noise_std, delta, n)
            )

    regrets = []
    for el in results:
        regrets.append(el.tolist())
    regrets = np.array(regrets)
    mean_regret = regrets.mean(axis=0)
    std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
    
    ax[0].plot(mean_regret, label=f"LinUCB - f{nf+1}")
    ax[0].fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)

    if nf == 7 or nf == 2:
        ax[1].plot(mean_regret, label=f"LinUCB - f{nf+1}")
        ax[1].fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)

        ax[2].plot(mean_regret, label=f"LinUCB - f{nf+1}")
        ax[2].fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)


ax[0].grid(True, which="both")
ax[1].grid(True, which="both")
ax[2].grid(True, which="both")

ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()
