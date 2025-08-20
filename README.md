# GRN-evolution

This repository contains the simulation code, configuration files, and analysis scripts used in: 

> **Chapel & de Boer, 2025**. *Evolutionary simulations reveal role for genomic recombination in the evolution of gene regulatory network complexity and robustness.*

The code implements genetic algorithms using the DEAP Python framework to simulate the evolution of gene regulatory networks (GRNs). Individuals within the population consist of biochemically-informed GRNs that model transcription factor binding and target gene expression. The simulation tracks fitness, regulatory complexity, and optionally population diversity to explore the role of recombination in the evolution of regulatory complexity. 

## Simulation framework 
### GRN Model
In this simulation, there are two gene types: target genes, the expression of which determines fitness, and TFs, which regulate the target genes and each other, but do not directly impact fitness. The 200 target genes and 20 TFs are randomly arranged on a single linear “chromosome”, and the binding affinities between them are captured in a 20 x 220 matrix. A gene’s expression is the sum of TF binding probabilities (determined by the combined effect of their affinity and expression), each weighted by their effect on expression (+1 for activators, -1 for repressors; n=10 of each). Each target gene has a randomly initialized ‘optimal’ expression; its fitness is normally distributed with respect to the log(expression), centred at this optimum. An individual’s fitness is the product of the fitness measures for all target genes, simulating a system where all target genes are equal and essential. 

### Evolution simulations
Populations are evolved using a genetic algorithm. In each generation, individuals are selected with probability proportional to their fitness and allowed to reproduce. Without recombination, offspring are a clonal replicate of a parent genome. With recombination, offspring are a hybrid of both parents, with a single randomly selected recombination site determining which genotypes are passed on. The TF affinities for each gene are inherited as a unit from a single parent, simulating inheritance of a single cis-regulatory region for each gene. For both with and without recombination, the TF affinities of offspring are subject to mutation. A new generation then begins, and the process repeats.

Code for three simulations are provided: 
* **Static environment**: As described above. Simulations proceed for the specified number of generations (default: 1 million). 
* **Neutral evolution**: Simulations proceed as in static environments, but individuals are selected randomly rather than according to fitness.
* **Changing environment**: After resuming from a checkpoint saved under static conditions, environment shifts will begin. The fitness at the checkpoint is stored, and the environment changes whenever the population's fitness is within 1% of the original fitness. Environment shifts are modeled by changing the fitness optima for a subset (n=20) of randomly-selected target genes.


## Repository structure 

```
├── analyses
│   ├── figures                 # Generated figures for manuscript
│   ├── robustness-outputs      # Output from robustness analyses
│   │   ├── changing-environment
│   │   └── static-environment
│   ├── figures_script.py
│   └── robustness.py
├── config-files                # Parameter .ini files for each test condition
│   ├── testA
│   ├── testB
│   └── ...
├── simulation-code             # Simulation code
│   ├── changing-environment
│   │   ├── checkpoints         # Checkpoint save files
│   │   ├── evoHelpers.py       # Helper functions for network creation, mutation, evaluation, etc.
│   │   └── evoShiftingEnv.py   # Main evolution script
│   ├── neutral-evolution
│   │   ├── checkpoints
│   │   ├── evoHelpers.py
│   │   └── evolution.py
│   └── static-environment
│       ├── checkpoints
│       ├── evoHelpers.py
│       └── evolution.py
├── environment.yml
└── README.md

```

## Installation 
Clone repository:
```bash 
git clone https://github.com/mach-2/GRN-evolution.git 
cd GRN-evolution
```
Create conda environment: 
```
conda env create -f environment.yml
conda activate grn-evo
```

## Usage

### Simulations
Simulation parameters are stored in `.ini` files (one `.ini` per replicate) in the `config-files` directory, and grouped according to the following test conditions: 

| test ID | Initial binding affinity | Recombination probability  |  Mutation rate |
| ------- | ------------- | ----- | ----- |
| testA | Moderate | 0.0 | Low |
| testB | Moderate | 0.0 | High |
| testC | Moderate | 1.0 | Low |
| testD | Moderate | 1.0 | High |
| testE | Minimal | 0.0 | Low |
| testF | Minimal | 0.0 | High |
| testG | Minimal | 1.0 | Low |
| testH | Minimal | 1.0 | High |
| testI | Non-uniform | 0.0 | Low |
| testJ | Non-uniform | 0.0 | High |
| testK | Non-uniform | 1.0 | Low |
| testL | Non-uniform | 1.0 | High |

Simulations can be run from within the `static-environment`, `changing-environment` or `neutral-evolution` directories. To begin a simulation, navigate to the appropriate directory and specify the test ID and replicate number (`.ini` file name): 
```
python evolution.py testA rep0
```
Checkpoints will be written by default every 5000 generations to the simulation's `checkpoints` directory. Checkpoints are large `.pkl` files that contain population data, rndstate information, and logbooks. It is recommended to periodically delete extra checkpoint files to reduce disk usage, keeping only every 50,000th to generate figures. 

To run a `changing-environment` simulation, two checkpoint files per replicate will need to be copied from the `static-environment` directory: the most recent checkpoint file (which contains the logbook), and the checkpoint file corresponding to the generation environmental changes will begin (which contains the population data). For example: 

```
├── changing-environment
│   ├── checkpoints
│   │   ├── testA0_gen300000.pkl
│   │   └── log
│   │       └── testA0_gen345000.pkl
│   ├── evolution.py
│   └── evoHelpers.py
```

If a simulation is interrupted before completion, it can be resumed from the most-recent checkpoint by running `python evolution.py test-ID rep#` again.

### Robustness 

After simulations have completed, the mutational robustness analyses can be run from the `analyses` directory: 

```
python -d path/to/checkpoint/directory -t testID
```

### Visualizations 

Figures from the paper are provided in `analyses/figures`, and can be reproduced with `analyses/figure_script.py`: 

```
python analyses/figure_script.py all
``` 
or
```
python analyses/figure_script.py Fig3 Supp2
```

## Acknowledgements
Built using [DEAP](https://github.com/DEAP/deap) (Distributed Evolutionary Algorithms in Python)