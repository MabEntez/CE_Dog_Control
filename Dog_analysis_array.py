# SHEPARDD E. granulosus sensu lato transmission model 
# Author: M. Entezami
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

def run_model(                                                            # SHEPARD model parameters
    run_id=0,
    n_steps: int=300,                                                     #Used to scale the population and transmission dynamics (Mostly for fitting so dog population is more stable)
    n_dogs: int=1000,                                                        #Total number of dog agents
    init_dog_inf_prop: float=0.2,                                         #Total number of Sheep agents with 1/7th in each cohort (aged 0-6) with an addition 1/7 aged 0 as male lambs
    seed=None,
    egg: float=100000,                                                    #Initial egg count the the start of the run
    cyst: float = 45,
    dog_beta: float=0.012,                                           #Larva fertility variable (Torgerson larva model)
    egg_ex: int=400,                                           #From Torgeson age variation of larva paper per month
    egg_beta: float=0.047,                                     
    egg_life_exp: int=12,                                                 #Life expectancy of the egg in the environment (Poissoned 36 months)
    parasite_life_exp: int=12,                                            #CHECKK - Life expectancy of the parasite in dog intestines (Poissoned 6 - 22 months OLD)
    deworm : bool=True,
    deworm_freq : int = 4,
    alt_deworm_freq : int = 4,
    spray : bool=False,
    burn_in : int=1500,
    verbose: bool=False,                                                  #Set to True for diagnostic information
    plot_run: bool=True
  ):
    # Initialise data
    S = n_dogs * (1 - init_dog_inf_prop) - 200
    L_queue = [0.0, 0.0]
    L = 200
    I = n_dogs * init_dog_inf_prop
    
    E = egg
    G = np.zeros(12)
    C = cyst
    
    G = np.array([20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000, 20000])
    
    N = n_steps
    time_S = np.zeros(N)
    time_L = np.zeros(N)
    time_I = np.zeros(N)
    time_E = np.zeros(N)
    time_G = np.zeros(N)
    time_C = np.zeros(N)
    
    time_S[0], time_L[0], time_I[0], time_E[0], time_G[0], time_C[0] = S, L, I, E, G.sum(), C
  
    for t in range(1, n_steps):
        # Dog infection dynamics
        p_inf = 1.0 - math.exp(-dog_beta * C)      # probability a susceptible becomes latent this step
        p_inf = min(max(p_inf, 0.0), 1.0)
        
        new_infections = p_inf * S
        if deworm and t <= burn_in and t % max(1, int(round(12 / deworm_freq))) == 0:
            recoveries = (I / parasite_life_exp) + (I / 72) + (I * 0.85)   # deworming effect 80% coverage 100% efficacy
        elif t >= burn_in and t % max(1, int(round(12 / alt_deworm_freq))) == 0:
            recoveries = (I / parasite_life_exp) + (I / 72) + (I * 0.85)   # deworming effect 80% coverage 100% efficacy
        else:
            recoveries = I / parasite_life_exp + I / 72

        progressed = L_queue[-1]                   # those infected 2 steps ago enter I now
        L_queue = [new_infections, L_queue[0]]
        
        I = I + progressed - recoveries
        I = min(max(I, 0.0), n_dogs)
        L = sum(L_queue)
        L = max(0.0, min(L, n_dogs - I))
        
        S = n_dogs - I - L
        S = max(0.0, min(S, n_dogs))

        # Egg dynamics
        if spray and t >= burn_in:
            new_eggs = egg_ex * I * (1 - 0.85 * 0.35 * 0.55)  # 85% coverage, 35% efficacy, 55% days in effect
        else:
            new_eggs = egg_ex * I
        egg_deaths = E / egg_life_exp
        
        G = np.roll(G, 1)
        G[0] = new_eggs

        E = E + G[0] - egg_deaths
        E = max(E, 0.0)

        # Cyst dynamics
        new_cysts = egg_beta * G[-1]
        cyst_deaths = C / 60  # Assuming cysts have a life expectancy of 5 years

        C = C + new_cysts - cyst_deaths
        C = max(C, 0.0)

        time_S[t] = S
        time_L[t] = L
        time_I[t] = I
        time_E[t] = E
        time_G[t] = G.sum()
        time_C[t] = C
        if verbose:
            print(f"Step {t}: S={S}, L={L}, I={I}, E={E}, G={G}, C={C}")
            
    if plot_run:
        plt.clf()
        plt.cla()
        plt.close("all")
        plt.figure()
        plt.tight_layout()
        plt.plot(np.arange(n_steps), time_S, label="Susceptible", color="green")
        plt.plot(np.arange(n_steps), time_L, label="Latent", color="orange")
        plt.plot(np.arange(n_steps), time_I, label="Infected", color="red")
        #plt.plot(np.arange(n_steps), time_E, label="Eggs", color="blue")
        #plt.plot(np.arange(n_steps), time_C, label="Cyst", color="black")
        plt.legend()
        plt.show()
    return {
      "Egg": time_E,
      "S": time_S,
      "L": time_L,
      "I": time_I,
      "Growing Cysts": time_G,
      "Cysts": time_C,
      "N": N
    }

#run_model(verbose=True, plot_run=True, deworm=False)


      
# --- Calibration helpers ---

def prevalence_from_params_db_eb(dog_beta: float, egg_beta: float, *,
                                 use_tail_avg: bool=True, tail_avg_steps: int=50, **model_kwargs) -> float:
    res = run_model(dog_beta=dog_beta, egg_beta=egg_beta, plot_run=False, verbose=False, deworm=False, burn_in=2000, **model_kwargs)
    I = res["I"]
    Ndogs = model_kwargs.get("n_dogs", 1000)
    if use_tail_avg:
        k = min(tail_avg_steps, len(I))
        return float(I[-k:].mean() / Ndogs)
    else:
        return float(I[-1] / Ndogs)

def calibrate_dogbeta_eggbeta(
    target_prev: float = 0.16,
    *,
    dog_beta_range=(1e-8, 1e-1),
    egg_beta_range=(1e-8, 1e-2),
    n_grid: int = 25,
    use_tail_avg: bool = True,
    tail_avg_steps: int = 50,
    refine_on: str = "dog_beta",   # or "egg_beta" for 1D bisection refine
    tol: float = 1e-4,
    max_iter: int = 60,
    **model_kwargs
) -> dict:
    import numpy as np
    
    dog_betas = np.geomspace(max(dog_beta_range[0], 1e-12), dog_beta_range[1], n_grid)
    egg_betas = np.geomspace(max(egg_beta_range[0], 1e-12), egg_beta_range[1], n_grid)
    
    best = {"dog_beta": None, "egg_beta": None, "prev": None, "abs_err": float("inf"), "grid": []}
    
    def metric(db, eb):
        return prevalence_from_params_db_eb(db, eb, use_tail_avg=use_tail_avg,
                                            tail_avg_steps=tail_avg_steps, **model_kwargs)
    
    # coarse grid
    for db in dog_betas:
        for eb in egg_betas:
            prev = metric(db, eb)
            err = abs(prev - target_prev)
            best["grid"].append((float(db), float(eb), float(prev), float(err)))
            if err < best["abs_err"]:
                best.update({"dog_beta": float(db), "egg_beta": float(eb), "prev": float(prev), "abs_err": float(err)})

    # optional 1D refine by bisection on the chosen axis
    if refine_on == "dog_beta":
        fixed = best["egg_beta"]
        low, high = dog_beta_range
        # bracket target around current best
        low = max(low, best["dog_beta"]/5)
        high = min(high, best["dog_beta"]*5)
        pl = metric(low, fixed); ph = metric(high, fixed)
        # expand if needed
        expand = 0
        while not (min(pl, ph) <= target_prev <= max(pl, ph)) and expand < 20:
            if pl < target_prev and ph < target_prev:
                low, high = high, min(dog_beta_range[1], high*2)
            elif pl > target_prev and ph > target_prev:
                high, low = low, max(dog_beta_range[0], low/2)
            pl, ph = metric(low, fixed), metric(high, fixed)
            expand += 1
        it = 0
        while it < max_iter and abs(high - low) > 1e-12:
            it += 1
            mid = 0.5*(low+high)
            pm = metric(mid, fixed)
            if abs(pm - target_prev) < tol:
                best.update({"dog_beta": float(mid), "egg_beta": float(fixed),
                             "prev": float(pm), "abs_err": float(abs(pm-target_prev)),
                             "iterations_refine": it})
                break
            if pm < target_prev:
                low, pl = mid, pm
            else:
                high, ph = mid, pm
        else:
            mid = 0.5*(low+high); pm = metric(mid, fixed)
            best.update({"dog_beta": float(mid), "egg_beta": float(fixed),
                         "prev": float(pm), "abs_err": float(abs(pm-target_prev)),
                         "iterations_refine": it, "note": "refine hit limit"})
    elif refine_on == "egg_beta":
        fixed = best["dog_beta"]
        low, high = egg_beta_range
        low = max(low, best["egg_beta"]/5)
        high = min(high, best["egg_beta"]*5)
        def metric_eb(eb): return metric(fixed, eb)
        pl = metric_eb(low); ph = metric_eb(high)
        expand = 0
        while not (min(pl, ph) <= target_prev <= max(pl, ph)) and expand < 20:
            if pl < target_prev and ph < target_prev:
                low, high = high, min(egg_beta_range[1], high*2)
            elif pl > target_prev and ph > target_prev:
                high, low = low, max(egg_beta_range[0], low/2)
            pl, ph = metric_eb(low), metric_eb(high)
            expand += 1
        it = 0
        while it < max_iter and abs(high - low) > 1e-12:
            it += 1
            mid = 0.5*(low+high)
            pm = metric_eb(mid)
            if abs(pm - target_prev) < tol:
                best.update({"dog_beta": float(fixed), "egg_beta": float(mid),
                             "prev": float(pm), "abs_err": float(abs(pm-target_prev)),
                             "iterations_refine": it})
                break
            if pm < target_prev:
                low, pl = mid, pm
            else:
                high, ph = mid, pm
        else:
            mid = 0.5*(low+high); pm = metric_eb(mid)
            best.update({"dog_beta": float(fixed), "egg_beta": float(mid),
                         "prev": float(pm), "abs_err": float(abs(pm-target_prev)),
                         "iterations_refine": it, "note": "refine hit limit"})
    
    best.update({"target": float(target_prev), "use_tail_avg": use_tail_avg, "tail_avg_steps": tail_avg_steps})
    return best

best = calibrate_dogbeta_eggbeta(
    target_prev=0.415,
    n_steps=2000,
    dog_beta_range=(1e-8, 1e-1),
    egg_beta_range=(1e-8, 1e-2),
    n_grid=25,
    use_tail_avg=True,
    tail_avg_steps=50,
    refine_on="dog_beta"   # or "egg_beta"
)
print(best)
no_int = run_model(n_steps=1620, dog_beta=best["dog_beta"], egg_beta=best["egg_beta"],
                   plot_run=True, deworm =False, burn_in=1980)
print("Final:", no_int["I"][-1]/1000, "Tail:", no_int["I"][-12:].mean()/1000, "Tail_egg:", no_int["Egg"][-12:].mean())

#deworm_burn = run_model(n_steps=2000, dog_beta=best["dog_beta"], egg_beta=best["egg_beta"], deworm = False, alt_deworm_freq = 4, burn_in=0,
#                     plot_run=True)
#print("Final:", deworm_burn["I"][-1]/1000, "Tail:", deworm_burn["I"][-12:].mean()/1000, "Tail_egg:", deworm_burn["Egg"][-12:].mean())

deworm_4 = run_model(n_steps=1620, dog_beta=best["dog_beta"], egg_beta=best["egg_beta"], deworm = False, alt_deworm_freq = 4,
                     plot_run=True)
print("Final:", deworm_4["I"][-1]/1000, "Tail:", deworm_4["I"][-12:].mean()/1000, "Tail_egg:", deworm_4["Egg"][-12:].mean())

deworm_8 = run_model(n_steps=1620, dog_beta=best["dog_beta"], egg_beta=best["egg_beta"], deworm = False,
                     plot_run=True, alt_deworm_freq=8)
print("Final:", deworm_8["I"][-1]/1000, "Tail:", deworm_8["I"][-12:].mean()/1000, "Tail_egg:", deworm_8["Egg"][-12:].mean())

spray = run_model(n_steps=1620, dog_beta=best["dog_beta"], egg_beta=best["egg_beta"], deworm =False, alt_deworm_freq=8,
                  plot_run=True, spray=True)
print("Final:", spray["I"][-1]/1000, "Tail:", spray["I"][-12:].mean()/1000, "Tail_egg:", spray["Egg"][-12:].mean())


tail_len = 120
window = 6  # 6-month rolling mean

# Extract tails
I_deworm = pd.Series(deworm_4["I"][-tail_len:])
I_deworm_8 = pd.Series(deworm_8["I"][-tail_len:])
I_spray = pd.Series(spray["I"][-tail_len:])

# Compute 3-month rolling means
roll_deworm = I_deworm.rolling(window=window, center=True).mean()/1000
roll_deworm_8 = I_deworm_8.rolling(window=window, center=True).mean()/1000
roll_spray = I_spray.rolling(window=window, center=True).mean()/1000

# x-axis (last 60 months)
x = np.arange(tail_len) / 12

plt.figure(figsize=(8, 5))
plt.plot(x, roll_deworm, color="black", alpha=0.8, linewidth=2, label="Deworming (4 times annually)")
plt.plot(x, roll_deworm_8, color="blue", alpha=0.6, linewidth=2, label="Deworming (8 times annually)")
plt.plot(x, roll_spray, color="red", alpha=0.6, linewidth=2, label="Deworming (8 times annually) + Spraying")

plt.xlabel("Number of Years")
plt.ylabel("Dog prevalence (6-month rolling mean)")
plt.title("Control strategy comparison - Dog Prevalence (5 Years, 6-month rolling mean)")
plt.legend()
plt.tight_layout()
plt.savefig('Dog_prev_comp_8.png', dpi=300, bbox_inches='tight')
plt.show()


# Extract tails
I_deworm = pd.Series(deworm_4["Egg"][-tail_len:])
I_deworm_8 = pd.Series(deworm_8["Egg"][-tail_len:])
I_spray = pd.Series(spray["Egg"][-tail_len:])

ref_egg_count = np.average(deworm_4["Egg"][-tail_len])
# Compute 3-month rolling means
roll_deworm = I_deworm.rolling(window=window, center=True).mean()/I_deworm.rolling(window=window, center=True).mean()
roll_deworm_8 = I_deworm_8.rolling(window=window, center=True).mean()/I_deworm.rolling(window=window, center=True).mean()
roll_spray = I_spray.rolling(window=window, center=True).mean()/I_deworm.rolling(window=window, center=True).mean()


plt.figure(figsize=(8, 5))
plt.plot(x, roll_deworm, color="black", alpha=0.8, linewidth=2, label="Deworming (4 times annually)")
plt.plot(x, roll_deworm_8, color="blue", alpha=0.6, linewidth=2, label="Deworming (8 times annually)")
plt.plot(x, roll_spray, color="red", alpha=0.6, linewidth=2, label="Deworming (8 times annually) + Spraying")

plt.xlabel("Number of Years")
plt.ylabel("Relative Egg Count (6-month rolling mean)")
plt.title("Control strategy comparison - Environmental Egg Contamination (5 Years, 6-month rolling mean)")
plt.legend()
plt.tight_layout()
plt.savefig('Relative_egg_count_8.png', dpi=300, bbox_inches='tight')
plt.show()

results_df = pd.DataFrame({
    "Month": np.arange(deworm_4["N"]),
    "No_int_Prev": no_int["I"] / 1000,
    "Deworm_4_Prev": deworm_4["I"] / 1000,
    "Deworm_8_Prev": deworm_8["I"] / 1000,
    "Spray_Prev": spray["I"] / 1000
})


results_df.to_csv("shepard_dog_prevalence.csv", index=False)
