import argparse
import glob
import os
from pathlib import Path
import pickle
import random
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools
from natsort import natsorted, ns

import evoHelpers as ev

# Disable interactive plotting (for running on Sockeye)
plt.ioff()

def load_config(config_parser):
    """
    Extracts configuration values from an initialization file and returns a dictionary of parameters
    """
    return{
        "reg_genes": config_parser.getint('network', 'reg_genes'),
        "target_genes": config_parser.getint('network', 'target_genes'),
        "chromosome_endpoints": config_parser.getintlist('network', 'chromosome_endpoints'),
        "initial_affinity": config_parser.get('network', 'initial_affinity'),

        "number_of_generations": config_parser.getint('evolution', 'number_of_generations'),
        "population_size": config_parser.getint('evolution', 'population_size'),
        "sexual_reproduction": config_parser.getboolean('evolution', 'sexual_reproduction'),
        "reproduction_probability": config_parser.getfloat('evolution', 'reproduction_probability'),
        "recombination_probability": config_parser.getfloat('evolution', 'recombination_probability'),
        "mutation_probability": config_parser.getfloat('evolution', 'mutation_probability'),

        "track_fitness": config_parser.getboolean('tracking', 'track_fitness'),
        "track_complexity": config_parser.getboolean('tracking', 'track_complexity'),
        "track_diversity": config_parser.getboolean('tracking', 'track_diversity'),

        "cp_frequency": config_parser.getint('checkpointing', 'cp_frequency'),
        "cp_name": config_parser.get('checkpointing', 'cp_name'),
        "results_file": config_parser.get('checkpointing', 'results_file'),

        "activator_list": config_parser.getintlist('initializations', 'activator_list', fallback=None),
        "repressor_list": config_parser.getintlist('initializations', 'repressor_list', fallback=None),
        "target_list": config_parser.getintlist('initializations', 'target_list', fallback=None),
        "initial_expression": config_parser.getfloatlist('initializations', 'initial_expression', fallback=None),
        "opt_expression": config_parser.getfloatlist('initializations', 'opt_expression', fallback=None),
        "max_fitness": config_parser.getfloat('initializations', 'max_fitness', fallback = None)
    }
    
def initialize_missing_values(config, config_parser, ini_path):
    """
    Generates a GRN topology and expression goals when beginning a new simulation.
    Writes values back to the .ini file 

    Args:
        config (dict): Existing configuration dictionary
        config_parser (ConfigParser): ConfigParser instance
        ini_path (str): Path to the original .ini file

    Returns:
        dict: Config with all values initialized
    """
    num_genes = config["reg_genes"] + config["target_genes"]
    
    # Generate values
    activators, repressors, targets = ev.create_topography(config["reg_genes"], config["target_genes"])
    initial_expression = ev.set_random_expression(list(range(num_genes)))
    optimal_expression = ev.set_random_expression(list(range(num_genes)))
    
    config["activator_list"] = activators
    config["repressor_list"] = repressors
    config["target_list"] = targets
    config["initial_expression"] = initial_expression
    config["opt_expression"] = optimal_expression
    
    # Write back to ini file 
    config_parser.set("initializations", "activator_list", ",".join(map(str, activators)))
    config_parser.set("initializations", "repressor_list", ",".join(map(str, repressors)))
    config_parser.set("initializations", "target_list", ",".join(map(str, targets)))
    config_parser.set("initializations", "initial_expression", ",".join(map(str, initial_expression)))
    config_parser.set("initializations", "opt_expression", ",".join(map(str, optimal_expression)))

    with open(ini_path, "w") as f:
        config_parser.write(f)

    return config
    
def register_deap_tools(config):
    """
    Registers DEAP tools and objects using simulation parameters

    Args:
        config (dict): Simulation parameters

    Returns:
        Toolbox: a DEAP toolbox with all registrations necessary for simulations
    """
    
    # Type definitions
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Complexity", float) 
    creator.create("Expression", list)
    creator.create("Individual", list,
        fitness=creator.Fitness,
        complexity=creator.Complexity,
        expression=creator.Expression
    ) 
    
    # Config values 
    nreg = config["reg_genes"]
    ntarget = config["target_genes"]
    chromosome = config["chromosome_endpoints"]
    
    # Toolbox Registration
    toolbox = base.Toolbox()
    toolbox.register("affinity", ev.edges, nreg, config["initial_affinity"])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.affinity, n = nreg * (nreg + ntarget))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("select", tools.selRandom)
    toolbox.register("mate", ev.cxBio, nreg=nreg, ntarget=ntarget, chromosome=chromosome, recombination_prob=config["recombination_probability"])
    toolbox.register("clone", ev.lightCopy)
    toolbox.register("mutate", ev.mutGaussian, 
                    mu = 0.0, sigma = 2.0, indpb=config["mutation_probability"], 
                    max_binding_affinity = 5.0, min_binding_affinity = -5.0)
    
    # Evaluation wrapper
    def eval_wrapper(ind_exp_pair):
        return ev.unpack_pairs_and_eval(
            reg_genes=config["reg_genes"],
            target_genes=config["target_genes"],
            activator_list=config["activator_list"],
            repressor_list=config["repressor_list"],
            target_list=config["target_list"],
            opt_expression=config["opt_expression"],
            pair=ind_exp_pair,
        )
    
    toolbox.register("evaluate", eval_wrapper)
    
    return toolbox
    
def initialize_population(max_fitness, config, toolbox):
    """
    Creates and evaluates an initial population.

    Args:
        max_fitness (float): Maximum fitness based on number of targets, used to scale fitness to a percentage
        config (dict): simulation parameters
        toolbox (Toolbox): DEAP toolbox with registered functions

    Returns:
        tuple: (population, logbook) ready for evolution
    """
    
    # Create population
    population = toolbox.population(n=config["population_size"])
    if config['initial_affinity'] == 'asymmetrical':
        population = list(toolbox.map(ev.asymmetrical_affinities, population))
    
    # Evaluate each individual
    for ind in population:
        ind.expression.values = np.copy(config["initial_expression"])
        fit, comp, expr = toolbox.evaluate((ind, ind.expression.values))
        ind.fitness.values = (fit,)
        ind.complexity.values = comp
        ind.expression.values = expr
        
    #Initialize logbook with gen 0 stats
    logbook = tools.Logbook()
    logbook.header = ["gen", "meanFit", "maxFit", "meanComp", "maxComp", "diversity"]
    logbook = track_metrics(population, gen=0, max_fitness = max_fitness, config = config, logbook=logbook)
        
    return population, logbook

def run_evolution(config, toolbox, population, max_fitness, logbook=None, start_gen=0):
    """Performs the main evolution simulation loop

    Args:
        config (dict): configuration parameters
        toolbox (Toolbox): DEAP toolbox with registered functions
        population (list): the current population
        max_fitness (float): maximum fitness, based on the number of target genes
        logbook (Logbook): A DEAP Logbook 
        start_gen (int, optional): The generation to begin running the simulation from. Defaults to 0.
    """
    if logbook is None:
        logbook = tools.Logbook()
        logbook.header = ["gen", "meanFit", "maxFit", "meanComp", "maxComp", "diversity"]
    
    total_gens = config["number_of_generations"]
    
    for gen in range(start_gen + 1, total_gens + 1): 
        # Selection
        offspring = toolbox.select(population, len(population))
        offspring = list(toolbox.map(toolbox.clone, offspring))
        
        # Reproduction
        if config["sexual_reproduction"]:
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config["reproduction_probability"]:
                    toolbox.mate(ind1, ind2)
                    toolbox.mutate(ind1)
                    toolbox.mutate(ind2)
                    del ind1.fitness.values, ind2.fitness.values
        else:
            for mutant in offspring:
                if random.random() < config["reproduction_probability"]:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
        # Evaluate new individuals 
        for ind in offspring:
            if not ind.fitness.valid: 
                fit, comp, expr = toolbox.evaluate((ind, ind.expression.values))
                ind.fitness.values = (fit,)
                ind.complexity.values = comp
                ind.expression.values = expr
                
        population[:] = offspring
        logbook = track_metrics(population, gen, max_fitness, config, logbook)
        
def track_metrics(population, gen, max_fitness, config, logbook):
    """
    Tracks and records relevant statistics from the population as it evolves

    Args:
        population (list): current population
        gen (int): current generation
        max_fitness (float): maximum fitness that can be achieved by the network
        config (dict): configuration parameters
        logbook (Logbook): DEAP logbook to update

    Returns:
        Logbook: a Logbook with metrics added for the latest generation
    """
    population_metrics = compute_population_metrics(population, max_fitness)

    log_to_file(gen, population_metrics["meanFit"], config)
    
    logbook.record(
        gen=gen,
        meanFit=population_metrics["meanFit"],
        maxFit=population_metrics["maxFit"],
        meanComp=population_metrics["meanComp"],
        maxComp=population_metrics["maxComp"]
    )

    if gen % config["cp_frequency"] == 0:
        save_checkpoint(population, gen, config, logbook)

    return logbook

def compute_population_metrics(population, max_fitness):
    fits = [ind.fitness.values[0] for ind in population]
    fits_percent = [np.exp(fit - max_fitness) * 100 for fit in fits]
    comps = [ind.complexity.values for ind in population]
    
    population_metrics = {
        "meanFit": np.mean(fits_percent),
        "maxFit": np.max(fits_percent),
        "meanComp": np.mean(comps),
        "maxComp": np.max(comps)
    }

    return population_metrics

def log_to_file(gen, mean_fitness, config):
    """
    Tracks fitness and generation number by writing to a .log file

    Args:
        gen (int): Current generation
        mean_fitness (float): Mean fitness score (scaled to a percentage of max)
        config (dict): Simulation parameters
    """
    log_path = Path('logs') / f"{config['cp_name']}.log"
    with open(log_path, "a") as log_file:
        print(f"-- Generation {gen} -- (Sexual: {config['sexual_reproduction']})", file=log_file)
        print(f"Average Fitness: {mean_fitness}", file=log_file, flush=True)
        
def save_checkpoint(population, gen, config, logbook):
    """
    Saves a population checkpoint and thins out the previous one

    Args:
        population (list): current DEAP population
        gen (int): current generation
        config (dict): configuration parameters containing checkpointing info
        logbook (Logbook): DEAP logbook to include in checkpoint
    """
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    cp_path = checkpoint_dir / f"{config['cp_name']}_gen{gen}.pkl"

    with open(cp_path, "wb") as f:
        pickle.dump({
            "population": population,
            "generation": gen,
            "rndstate": random.getstate(),
            "logbook": logbook,
        }, f)

    # Thin previous checkpoint
    prev_gen = gen - config["cp_frequency"]
    if prev_gen > 0:
        prev_path = Path("checkpoints") / f"{config['cp_name']}_gen{prev_gen}.pkl"
        if os.path.exists(prev_path):
            with open(prev_path, "rb") as f:
                prev_cp = pickle.load(f)
            prev_cp.pop("logbook", None)
            with open(prev_path, "wb") as f:
                pickle.dump(prev_cp, f)    

def main():
    """
    Runs GRN evolution simulation.
    Loads a config, handles checkpointing, initializes population, and triggers main evolution loop.
    """
    parser = argparse.ArgumentParser(description="Run a simulation replicate.")
    parser = argparse.ArgumentParser(description="Run a simulation replicate.")
    parser.add_argument("test", help="Test name (e.g., testA, testB)")
    parser.add_argument("replicate", help="Config file ID (e.g., rep0, rep1)")
    args = parser.parse_args()
    replicate_id = args.replicate
    
    SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    #  Load .ini config 
    ini_path = SCRIPT_DIR / "config-files" / args.test/ f"{args.replicate}.ini"
    
    config_parser = ConfigParser(converters={
        'intlist': lambda x: [int(i.strip()) for i in x.split(',')],
        'floatlist': lambda x: [float(i.strip()) for i in x.split(',')]
    })
    config_parser.read(ini_path)
    config = load_config(config_parser)
    
    # Initialize any missing values on a new run
    if config["activator_list"] is None:
        config = initialize_missing_values(config, config_parser, ini_path)
        
    # Register DEAP operators 
    toolbox = register_deap_tools(config)
    max_fitness = ev.calculate_max_fitness(config["target_genes"])
    ev.init_offspring_arrays(config["reg_genes"],config["target_genes"])
    
    # Check for resume checkpoint 
    cp_pattern = str(Path("checkpoints") / f"{config['cp_name']}_gen*.pkl")
    checkpoint_files = natsorted(glob.glob(cp_pattern), alg=ns.IGNORECASE)

    if checkpoint_files:
        latest_cp = checkpoint_files[-1]
        print(latest_cp)
        with open(latest_cp, "rb") as f:
            cp_data = pickle.load(f)
        population = cp_data["population"]
        logbook = cp_data["logbook"]
        start_gen = cp_data["generation"] 
        random.setstate(cp_data["rndstate"])
        print(f"Resuming from checkpoint {latest_cp}, starting at generation {start_gen}")
    else:
        population, logbook = initialize_population(max_fitness, config, toolbox)
        start_gen = 0
        
    # === Run evolution ===
    run_evolution(
        config=config,
        toolbox=toolbox,
        population=population,
        max_fitness=max_fitness,
        logbook=logbook,
        start_gen=start_gen,
    )

if __name__ == "__main__":
    main()