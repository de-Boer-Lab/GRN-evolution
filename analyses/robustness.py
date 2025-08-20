import evoHelpers as ev
from evolution import register_deap_tools, load_config
import numpy as np 
import random 
import pickle 
import os
import glob
from natsort import natsorted, ns
from configparser import ConfigParser
from deap import base, creator, tools 
import argparse
import scipy.stats as stats 
from pathlib import Path    

repo_root = Path(__file__).resolve().parents[1]
config_dir = repo_root / "config-files"
output_dir = Path("robustness-outputs")
output_dir.mkdir(exist_ok=True)

# Get directory and testname as CLI arguments
parser = argparse.ArgumentParser(description="Analyze robustness and complexity for GRN evolution checkpoints")
parser.add_argument("--directory", "-d", required=True, help="Checkpoint directory")
parser.add_argument("--testname", "-t", required=True, help="Simulation testname (e.g., testA, testB)")
args = parser.parse_args()

cp_dir = Path(args.directory)
testname = args.testname

output_log = output_dir / f"robustnessIndividual-change-{testname}.measures"
summary_log = output_dir / f"robustness-change-{testname}.measures"

replicates = [str(i) for i in range(10)]
generations = list(range(50000, 1000001, 50000)) + [1010000]
CHECKPOINT_TEMPLATE = "{test}{rep}_gen{gen}.pkl"
N_MUTANTS = 100

for rep in replicates: 
    ini = f"rep{rep}"
    config_path = f"/arc/project/st-cdeboer-1/mchapel/GRN-repository/configuration-files/{testname}/rep{rep}.ini"
    config_path = config_dir / testname / f"rep{rep}.ini"
    
    print(f"Initializing from {config_path}", file=open(output_log, 'a'), flush=True)
    config_parser = ConfigParser(converters={
        'intlist': lambda x: [int(i.strip()) for i in x.split(',')],
        'floatlist': lambda x: [float(i.strip()) for i in x.split(',')]
    })
    config_parser.read(config_path)
    config = load_config(config_parser)
    toolbox = register_deap_tools(config)
    
    population_expression_distances = []
    
    for gen in generations: 
        cp_filename = os.path.join(cp_dir, testname, CHECKPOINT_TEMPLATE.format(test=testname, rep=rep, gen=gen))
        
        if not os.path.exists(cp_filename):
            print(f"Checkpoint missing: {cp_filename}", file=open(output_log, 'a'), flush=True)
            continue

        print(f"Resuming from {cp_filename}", file=open(output_log, 'a'), flush=True)
        with open(cp_filename, "rb") as cp_file:
            cp = pickle.load(cp_file)

        pop = cp["population"]
        random.setstate(cp["rndstate"])
        
        fitnesses = [toolbox.evaluate((ind, ind.expression.values)) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit[0],
            ind.complexity.values = fit[1]
            ind.expression.values = fit[2]

        mutant_distances = []
        pop_complexity = []

        population = list(toolbox.map(toolbox.clone, pop))
        index_positions = list(range(len(pop[0])))
        
        for ind, current_ind in zip(pop, population):
            individual_dists = []
            baseline_expression = ind.expression.values 
            
            for _ in range(N_MUTANTS):
                current_ind[:] = ind[:]
                current_ind[random.choice(index_positions)] += random.choice([-4.0, 4.0])
                new_expr = toolbox.evaluate([current_ind, baseline_expression])[2]
                eucl_dist = np.linalg.norm(np.array(baseline_expression) - new_expr)
                individual_dists.append(1 - eucl_dist)
                
            mutant_distances.append(np.mean(individual_dists))
            pop_complexity.append(toolbox.evaluate([ind, baseline_expression])[1])
             
        population_expression_distances.append(mutant_distances)

        print(f"mutant_distances_gen{gen} = {mutant_distances}", file=open(output_log, 'a'), flush=True)
        correlation = np.corrcoef(pop_complexity, mutant_distances)
        print(f"correlation_gen{gen} = {correlation[0,1]:.4f}", file=open(output_log, 'a'), flush=True)

    # Final per-replicate summary
    mean_robustness = [np.mean(x) for x in population_expression_distances]
    print(f"mean_{rep} =", mean_robustness, file=open(summary_log, 'a'), flush=True)

    # Clean up
    del pop, cp