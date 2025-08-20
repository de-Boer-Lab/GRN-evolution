# Functions required to run evolution simulation
import random
import numpy as np
import math
import pickle
from datetime import datetime
from collections.abc import Sequence
from itertools import repeat
from operator import attrgetter


# import matplotlib.pyplot as plt

random.seed(2)
# np.random.seed(2)

# Global pre-allocated arrays to improve performance of crossover step
offspring1 = None
offspring2 = None

def create_topography(nreg, ntarget): 
    """Randomly assigns chromosomal indices to activators, repressors, and targets.
    """
    geneList = list(range(nreg + ntarget))
    random.shuffle(geneList) 
    
    regulator_list = geneList[:nreg]
    activator_list = regulator_list[:int(nreg//2)]
    repressor_list = regulator_list[int(nreg//2):]
    target_list = geneList[nreg:]
    return activator_list,repressor_list,target_list

def calculate_max_fitness(target):
    """Calculates a maximum possible fitness score, assuming optimal expression across all targets
    Used to determine fitness as a % of maximum.
    """
    return target*np.log(normalDistScore(0,0)) 

def set_random_expression(geneList): 
    """Generates a list of random expression values (in log space: [-1, 1]) for each gene
    """
    return [random.uniform(-1, 1) for _ in geneList]

def edges(reg_genes, starting_affinity):
    """Returns initial binding affinities when generating a new population.

    Args:
        reg_genes (int): Number of TFs
        starting_affinity (str): Initial complexity. One of {"minimum", "moderate", "asymmetrical"}

    Raises:
        ValueError: If an unknown initial binding affinity condition is provided. 

    Returns:
        float: initial binding affinity
    """
    if starting_affinity == "minimum" or starting_affinity == 'asymmetrical':
        return -(0.517 * math.log(reg_genes) + 2.516)
    elif starting_affinity == "moderate":
        return 0.0
    else: 
        raise ValueError(f"Unknown initial complexity mode: {starting_affinity}")

def asymmetrical_affinities(individual):
    """Re-configures the GRN topology for asymmetric initial binding affinities

    """
    for col in range(220): 
        row = random.randint(0, 19)
        index = row*220 + col
        individual[index] = 0.517*math.log(20)+2.516
    return individual


def unpack_pairs_and_eval(reg_genes, target_genes, activator_list, repressor_list, target_list, opt_expression, pair):
    """
    Unpacks an (individual, initial_expression) pair and passes contents to evaluator.
    Intended for use with DEAP toolbox.map() for multiprocessing
    """
    ind = pair[0]
    initial_expression = pair[1]
    return evalIndividual(reg_genes, target_genes, activator_list, repressor_list, target_list, opt_expression, initial_expression, ind)
    
    
def evalIndividual(reg_genes, target_genes, activator_list, repressor_list, target_list, opt_expression, initial_expression, ind):
    """
    Evaluates an individual's fitness by simulating gene expression 
    until the system stabilizes or reaches a maximum iteration limit.

    Returns:
        tuple: (fitness_score, gini_score, final_expression)
    """
    # Convert individual genome into a regulatory matrix
    ind = np.reshape(ind,newshape=(reg_genes, (reg_genes+target_genes)))
    regulator_list = np.append(activator_list, repressor_list)
    
    stable = False
    iterations = 0 
    
    initial_expression = setExpression(regulator_list, activator_list, repressor_list, initial_expression, ind)
    
    # Track expression for limit-finding fallback
    exp = [[] for i in range(reg_genes + target_genes)]   
    
    while stable == False:
        # Record expression values in case expression doesn't stabilize
        for i in range(reg_genes + target_genes): 
            exp[i].append(initial_expression[i]) 
            
        # Update expression
        new_expression = setExpression(regulator_list, activator_list, repressor_list, initial_expression, ind) 
        
        # Check if expression has stabilized
        if np.allclose(initial_expression, new_expression, atol=1e-4, rtol=0.0):
            stable = True
        else:
            iterations += 1 
            initial_expression = new_expression
        
        # If expression hasn't stabilized, stop iterating and find an average
        if iterations == 5000:
            new_expression = findExpressionLimit(exp)
            stable = True
            
    # Evaluate expression profile
    fitness_score = fitnessScore(target_list, new_expression, opt_expression)
    gini_score = giniScore(new_expression, ind)
    return fitness_score, gini_score, new_expression
        

def setExpression(regulator_list, activator_list, repressor_list, initial_expression, ind):
    """
    Updates gene expression levels based on regulatory interactions. 
    Expression is updated in log space and mapped to [1, 1] using a sigmoid function.

    Args:
        regulator_list (list): All TF indices (activators + repressors)
        activator_list (list): Subset of TFs acting as activators
        repressor_list (list): Subset of TFs acting as repressors
        initial_expression (list): Current expression values (in log space)
        ind (np.ndarray): Regulatory matrix 

    Returns:
        np.ndarray: Updated expression levels in log space (range [1, 1])
    """
    # Get binding affinities and expression levels for activators and repressors
    act_affinities = ind[[i for i in range(len(regulator_list)) if regulator_list[i] in activator_list],:]
    rep_affinities = ind[[i for i in range(len(regulator_list)) if regulator_list[i] in repressor_list],:]
    act_expression = [initial_expression[gene] for gene in activator_list]
    rep_expressions = [initial_expression[gene] for gene in repressor_list]
        
    # Compute activation and repression strengths for each target
    activations = np.sum(np.exp(act_affinities.T)*np.exp(act_expression) / 
                         (np.exp(act_affinities.T)*np.exp(act_expression) + 1), axis = 1)
    repressions = np.sum(np.exp(rep_affinities.T)*np.exp(rep_expressions) / 
                         (np.exp(rep_affinities.T)*np.exp(rep_expressions) + 1), axis=1)
    expression = activations - repressions
    
    # Compress to range [-1, 1] with sigmoid function
    expression_sigmoid = ((1/(1+np.e**(-expression)))-0.5)*2
    
    return expression_sigmoid

def findExpressionLimit(expSeries, debug=False):
    """
    Estimates the final expression state for a non-converging gene by
    averaging its full expression history over time.

    Args:
        expSeries (list of lists): A time series of expression values for each gene.
        debug (bool): If True, generates a plot of expression dynamics for troubleshooting.

    Returns:
        list: Averaged final expression values for each gene.
    """
    expression = [sum(gene) / len(gene) for gene in expSeries]

    if debug: 
        import matplotlib.pyplot as plt
        from datetime import datetime
        for gene in expSeries:
            plt.plot(gene)
        plt.savefig(f"badOscillations/{datetime.now()}.png")
        plt.close()

    return expression

def fitnessScore(target_list, expression, optExpression):
    """
    Computes total fitness for an individual based on how close expression for each target 
    gene is to the optimum. TF expression levels are excluded.

    Args:
        target_list: List of target genes (non-TF indices)
        expression (list): current expression values (in log space)
        optExpression (list): optimum expression values (in log space)

    Returns:
        float: fitness score for this individual
    """
    
    fitness = 0
    for i in target_list:
        geneFitness = normalDistScore(optExpression[i], expression[i])
        geneFitness = max(geneFitness, 1e-5)  # Prevent log(0)
        fitness += np.log(geneFitness)
    return fitness

def normalDistScore(optimum, actual, sigma=0.75): 
    """
    Computes the probability density of a normal distribution centered at `optimum`,
    evaluated at `actual`. Used as a fitness proxy.
    """
    return (
        math.exp(-((actual - optimum) ** 2) / (2 * sigma ** 2))
        / (sigma * math.sqrt(2 * math.pi))
    )


def lightCopy(selected):
    """
    Creates a new individual by manually copying only relevant data.
    Avoids the time overhead of deepcopy by specifying exactly what to duplicate.

    This is equivalent to a partial deep copy: the genome and expression are fully copied,
    while fitness and complexity are shallowly assigned (assuming they will be overwritten).
    """
    # Create a new instance of same class
    offspring = selected.__class__()
    # Copy genome and expression
    offspring[:] = selected[:]
    offspring.expression.values = np.copy(selected.expression.values)
    
    # Shallow copy of fitness and complexity 
    offspring.fitness.values = selected.fitness.values 
    offspring.complexity.values = selected.complexity.values 
        
    return offspring

def init_offspring_arrays(nreg, ntarget):
    """
    Pre-allocates global offspring arrays for use in cxBio function. 
    This improves performance by avoiding repeatedly allocating memory for new offspring during the simulation

    Args:
        nreg (int): Number of regulatory genes
        ntarget (int): Number of target genes
    """
    global offspring1, offspring2
    offspring1 = np.empty(shape=(nreg, nreg + ntarget))
    offspring2 = np.empty(shape=(nreg, nreg + ntarget))

def cxBio(ind1, ind2, nreg, ntarget, chromosome, recombination_prob):
    """
    Performs biologically-inspired crossover between two individuals by recombining
    defined chromosomal segments with probabilistic crossover points.

    Args:
    nreg (int): Number of regulatory genes
    ntarget (int): Number of target genes
    chromosome (list of int): Start/end indices defining chromosomal segments
    recombination_prob (float): Probability of recombining each segment
    ind1 (list): First individual (modified in-place)
    ind2 (list): Second individual (modified in-place)

    Note:
    Uses pre-allocated global buffers `offspring1` and `offspring2`.
    Individuals are reshaped to (nreg x nreg+ntarget) matrices for crossover.
    """
    global offspring1, offspring2 
    # Reshape individual genomes into regulatory matrices
    ind1_edges = np.reshape(ind1, newshape=(nreg, nreg + ntarget))
    ind2_edges = np.reshape(ind2, newshape=(nreg, nreg + ntarget))
    
    for offspring in offspring1, offspring2:
        # Recombine each chromosomal segment
        for i in range(len(chromosome)-1): 
            start = chromosome[i]
            end = chromosome[i+1]
            if random.random() < recombination_prob:
                cxPoint = random.randint(start,end) # crossover point within segment
                if random.randint(0,1) == 0: 
                    # Inherit left from parent 1, right from parent 2
                    offspring[:,start:cxPoint] = ind1_edges[:, start:cxPoint]
                    offspring[:,cxPoint:end] = ind2_edges[:, cxPoint:end]
                else: 
                    # Inherit left from parent 2, right from parent 1
                    offspring[:,start:cxPoint] = ind2_edges[:, start:cxPoint]
                    offspring[:,cxPoint:end] = ind1_edges[:, cxPoint:end]
            else:
                # No recombination: copy entire segment from one parent 
                if random.randint(0,1) == 0: 
                    offspring[:] = ind1_edges[:]
                else: offspring[:] = ind2_edges[:]

    ind1[:] = offspring1.flatten()
    ind2[:] = offspring2.flatten()

    return   
    
def giniScore(expression, adjMatrix):
    """
    Calculates average regulatory complexity as 1 - Gini(binding probabilities),
    per gene, then returns the mean across all genes.

    Higher values reflect more evenly distributed regulation.

    Args:
        expression (array): expression values (in log space)
        adjMatrix (np.ndarray): binding affinities (in log space)

    Returns:
        float: Complexity score for an individual
    """
    # Exponentiate binding affinities and expression (back to linear scale)
    affinities = np.exp(adjMatrix)
    expression = np.reshape(np.exp(expression), newshape=(1, -1))  # Shape: (1, total_genes)
    
    # Compute activity matrix. Each cell: binding probability from TF to target
    activity_matrix = affinities * expression / (affinities * expression + 1)
    
    # Gini is computed per column and then averaged
    n = activity_matrix.shape[0]
    sorted_matrix = np.sort(activity_matrix, axis=0)  
    indices = np.arange(1, n + 1).reshape(-1, 1) 
    
    gini_numerators = np.sum((2 * indices - n - 1) * sorted_matrix, axis=0)
    gini_denominators = np.sum(sorted_matrix, axis=0)
    
    gini_values = 1 - gini_numerators / (n * gini_denominators)
    return np.mean(gini_values)

def selRoulette(individuals, k, fit_attr="fitness"):
    """
    Performs roulette selection using fitness values
    Similar to DEAP's roulette function, but uses random.choices for faster performance

    Args:
        individuals (list): _description_
        k (int): Number of individuals to select
        fit_attr (str, optional): Attribute to select on. Defaults to "fitness".

    Returns:
        list: selected individuals
    """
    fits = [getattr(ind, fit_attr).values[0] for ind in individuals]
    maxFit = np.amax(fits)  # For numerical stability
    normalized = [np.exp(x - maxFit) for x in fits]

    sum_fits = sum(normalized)
    if sum_fits == 0:
        fitness_proportions = [1 / len(individuals)] * len(individuals)
    else:
        fitness_proportions = [x / sum_fits for x in normalized]

    return random.choices(individuals, weights=fitness_proportions, k=k) 

def mutGaussian(individual, mu, sigma, indpb, max_binding_affinity, min_binding_affinity):
    """
    Applies Gaussian mutation to each gene of an individual with a given probability.
    Each gene is mutated by adding Gaussian noise and clipped to stay within the 
    specified binding affinity range.

    Args:
        individual (list): The individual to mutate.
        mu (float): Mutation mean.
        sigma (float): Mutation std dev.
        indpb (float): Probability of mutating each gene.
        max_binding_affinity (float): Max allowed value.
        min_binding_affinity (float): Min allowed value.

    Returns:
        tuple: A tuple containing the mutated individual.
    """
    size = len(individual)
    if not isinstance(mu, Sequence):
        mu = repeat(mu, size)
    elif len(mu) < size:
        raise IndexError(f"mu must be at least the size of individual: {len(mu)} < {size}")
    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError(f"sigma must be at least the size of individual: {len(sigma)} < {size}")

    for i, (m, s) in enumerate(zip(mu, sigma)):
        if random.random() < indpb:
            mutated_value = individual[i] + random.gauss(m, s)
            individual[i] = min(max(mutated_value, min_binding_affinity), max_binding_affinity)

    return individual,        
    
def get_binding_affinities(pop, target_genes, reg_genes, activators_repressors):
    """
    Determines the average binding matrix across the population, then extracts 
    the TF-TF and TF-Target edges. Used in generating binding probability heatmaps.
    """
    population_array = np.array(pop)
    mean_individual = np.mean(population_array, axis=0)
    mean_individual = np.reshape(mean_individual, newshape=(reg_genes, reg_genes+target_genes))
    
    non_tf_indices = [i for i in range(target_genes+reg_genes) if i not in activators_repressors]
    
    tf_tf_edges = mean_individual[:,activators_repressors]
    tf_target_edges = mean_individual[:, non_tf_indices]
    
    return tf_tf_edges, tf_target_edges

# === ARCHIVED FUNCTIONS === 
# These are not currently used, but are kept for reference or future re-use 

# def define_gene_module(targets, n): 
#     """Returns a group of n random genes from a list of target genes

#     Args:
#         targets (list): a list of target genes
#         n (int): number of genes to select

#     Returns:
#         list: a list of randomly-selected target genes
#     """
#     module = random.sample(targets, n)
#     return module

# def expressionIsStable(initial_expression, new_expression): 
#     """Returns True if two expression profiles are element-wise similar within tolerance."""
#     return all(abs(x - y) < 0.0001 for x, y in zip(initial_expression, new_expression))

# def mutTFfreeze(individual, mu, sigma, indpb, max_binding_affinity, min_binding_affinity, TFs): 
#     size = len(individual)
#     num_TFs = len(TFs)
#     targetsize = size // num_TFs

#     # Convert TFs to a set for faster membership testing
#     TFs_set = set(TFs)
    
#     # Pre-generate mutation probabilities and mutations
#     mutation_probs = np.random.random(size=size)
#     mutations = np.random.normal(mu, sigma, size=size)
    
#     for i in range(size):
#         row_index = i // targetsize   # Determine the row (TF)
#         column_index = i % targetsize # Determine the column (Target)
        
#         # Skip mutation if column is in TFs
#         if column_index in TFs_set:
#             continue
        
#         # Apply mutation based on indpb
#         if mutation_probs[i] < indpb:
#             mutated_value = individual[i] + mutations[i]
#             individual[i] = min(max(mutated_value, min_binding_affinity), max_binding_affinity)

#     return individual
    
    
# def mutTargetFreeze(individual, mu, sigma, indpb, max_binding_affinity, min_binding_affinity, TFs):
#     size = len(individual)
#     num_TFs = len(TFs) 
#     targetsize = size // num_TFs
    
#     TF_set = set(TFs)
    
#     # Generate mutation probabilities and sizes 
#     mutation_probs = np.random.random(size = size) 
#     mutations = np.random.normal(mu, sigma, size=size)
    
#     for i in range(size): 
#         row_index = i // targetsize # Determine the row (Target)
#         column_index = i % targetsize # Determine the column (Target)
        
#         if column_index not in TF_set: 
#             continue 
        
#         if mutation_probs[i] < indpb: 
#             mutated_value = individual[i] + mutations[i]
#             individual[i] = min(max(mutated_value, min_binding_affinity), max_binding_affinity)
            
#     return individual
    
# def get_max_diversity(max_binding_affinity, min_binding_affinity, reg_genes, target_genes):
#     min_individual = np.array([min_binding_affinity for x in range(reg_genes*(reg_genes+target_genes))])
#     max_individual = np.array([max_binding_affinity for x in range(reg_genes*(reg_genes+target_genes))])
    
#     tf_tf_indices = [i*(reg_genes+target_genes) + j for i in range(reg_genes) for j in range(reg_genes)]
#     tf_target_indices = [i*(reg_genes+target_genes) + j for i in range(reg_genes) for j in range(target_genes)]
    
#     min_tf_tf_interactions = min_individual[tf_tf_indices]
#     min_tf_target_interactions = min_individual[tf_target_indices]
#     max_tf_tf_interactions = max_individual[tf_tf_indices]
#     max_tf_target_interactions = max_individual[tf_target_indices]

#     max_euclidean = np.linalg.norm(max_individual - min_individual, axis =0)
#     max_euclidean_tf_tf = np.linalg.norm(max_tf_tf_interactions - min_tf_tf_interactions, axis = 0)
#     max_euclidean_tf_target = np.linalg.norm(max_tf_target_interactions - min_tf_target_interactions, axis = 0)
    
#     return max_euclidean, max_euclidean_tf_tf, max_euclidean_tf_target


# def get_diversity(max_diversity, population, TF_list, target_list):
#     population_array = np.array(population)
#     tf_tf_indices = [i*(len(TF_list) + len(target_list)) + j for i in range(len(TF_list)) for j in TF_list]
#     tf_target_indices = [i*(len(TF_list) + len(target_list)) + j for i in range(len(TF_list)) for j in target_list]
    
#     tf_tf_interactions = population_array[:, tf_tf_indices]
#     tf_target_interactions = population_array[:, tf_target_indices]
    
#     mean_individual = np.mean(population_array, axis = 0)
#     mean_tf_tf = np.mean(tf_tf_interactions, axis = 0)
#     mean_tf_target = np.mean(tf_target_interactions, axis=0)

#     euclidean = np.linalg.norm(population_array - mean_individual, axis =1)/max_diversity[0]
#     euclidean_tf_tf = np.linalg.norm(tf_tf_interactions - mean_tf_tf, axis = 1)/max_diversity[1]
#     euclidean_tf_target = np.linalg.norm(tf_target_interactions - mean_tf_target, axis = 1)/max_diversity[2]

#     diversity = np.mean(euclidean)
#     diversity_tf_tf = np.mean(euclidean_tf_tf)
#     diversity_tf_target = np.mean(euclidean_tf_target)
    
#     return diversity, diversity_tf_tf, diversity_tf_target