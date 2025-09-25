# Get 50-generation increments of fitness/comp plots 
import matplotlib
matplotlib.use("Agg")   # non-interactive backend to prevent GUI caching
import matplotlib.pyplot as plt
import numpy as np 
import pickle
import glob 
import os
from natsort import natsorted, ns
import evolutionMatrix
import gc 

# Load replicates 
test = ['F0', 'H0']
# test = ['F', 'H']

def load_data(test): 
    # pattern = "checkpoints/changing-environments/test"+test+"/checkpoints/long-evo-sexual0_gen*000.pkl"
    # pattern = "../checkpoints/variedParameters/testD-diversity/long-evo-sexual0_gen*000.pkl"
    # pattern = "../checkpoints/variedParameters/averaged-gini/test{0}/long-evo-sexual0_gen*000.pkl".format(test)
    pattern = f"for-submission/test{test}_gen*.pkl"
    files = list(filter(os.path.isfile, glob.glob(pattern)))
    files = natsorted(files, alg=ns.IGNORECASE)
    checkpoint = files[-1]
    results = pickle.load(open(checkpoint, 'rb'))
    
    # clean duplicate entries 
    log = results['logbook']
    genlist = log.select('gen')
    
    seen = set()
    dupes = [x for x in genlist if x in seen or seen.add(x)]
    for x in dupes:
        log.pop(x)
    log.pop(0)
    fits = log.select('meanFit')
    comps = log.select('meanComp')
    del results
    
    return fits, comps 

pixel_height = 400
dpi = 100
aspect_ratio = 4/3
height_in_inches = pixel_height/dpi
width_in_inches = height_in_inches * aspect_ratio


    
def plot(rec_fits, rec_comps, norec_fits, norec_comps, gen,
         ax_fitness=None, ax_complexity=None):
    """
    Draw time series up to generation `gen`.
    Inputs are NumPy arrays (or will be coerced).
    """

    # clamp gen to available data
    # idx = min(gen, rec_fits.shape[0], norec_fits.shape[0],
    #                 rec_comps.shape[0], norec_comps.shape[0])
    # if idx <= 0:
    #     return

    xs = np.arange(1, gen + 1)

    if ax_fitness is not None:
        ax_fitness.plot(xs, rec_fits[:gen],   color='#2ca02c', label='Recombination',    alpha=0.7)
        ax_fitness.plot(xs, norec_fits[:gen], color='#ffa500', label='No recombination', alpha=0.7)

    if ax_complexity is not None:
        ax_complexity.plot(xs, rec_comps[:gen],   color='#2ca02c', label='Recombination',    alpha=0.7)
        ax_complexity.plot(xs, norec_comps[:gen], color='#ffa500', label='No recombination', alpha=0.7)


rec_fits, rec_comps = load_data(test[1])
norec_fits, norec_comps = load_data(test[0])

plt.ioff()  # no interactive state

# After load_data(...):
rec_fits       = np.asarray(rec_fits, dtype=np.float32)
print(rec_fits)
rec_comps      = np.asarray(rec_comps, dtype=np.float32)
norec_fits     = np.asarray(norec_fits, dtype=np.float32)
norec_comps    = np.asarray(norec_comps, dtype=np.float32)


# Generations to plot:
gens = list(range(1, 501)) + list(range(510, 2001, 10)) + list(range(2050, 100001, 50))

from pathlib import Path
base_dir = Path("for-submission")
outdir_fitness    = base_dir / "frames_fitness"
outdir_complexity = base_dir / "frames_complexity"
outdir_fitness.mkdir(parents=True, exist_ok=True)
outdir_complexity.mkdir(parents=True, exist_ok=True)

for gen in gens:
    print(gen)
    # ----- FITNESS FRAME -----
    fig1, ax1 = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)  # no tight_layout
    ax1.set_ylabel('Fitness (% of maximum)')
    ax1.set_ylim(-2.22724586064651, 102.22724586064651)
    ax1.set_xscale('log')
    ax1.set_xlim((1, 100000))
    ax1.set_xlabel('Generations')

    plot(rec_fits, rec_comps, norec_fits, norec_comps, gen, ax_fitness=ax1, ax_complexity=None)
    fig1.savefig(outdir_fitness / f"generation{gen}.fitness.png", bbox_inches="tight")
    plt.close(fig1); del fig1, ax1

    # ----- COMPLEXITY FRAME -----
    fig2, ax2 = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=dpi)
    ax2.set_ylabel('Complexity (1 - Gini coefficient)')
    ax2.set_ylim(-0.0222724586064651, 1.0222724586064651)
    ax2.set_xscale('log')
    ax2.set_xlim((1, 100000))
    ax2.set_xlabel('Generations')

    plot(rec_fits, rec_comps, norec_fits, norec_comps, gen, ax_fitness=None, ax_complexity=ax2)
    ax2.legend(loc='best', frameon=False)
    fig2.savefig(outdir_complexity / f"generation{gen}.complexity.png", bbox_inches="tight")
    plt.close(fig2); del fig2, ax2

    # encourage Python to release large render buffers
    if (gen % 10) == 0:
        gc.collect()
