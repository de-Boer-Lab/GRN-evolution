import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import matplotlib.cm as cm 
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import seaborn as sns
import pickle 
import glob, os, sys
import pandas as pd 
from natsort import natsorted, ns 
import numpy as np
import re
import math, random
from pathlib import Path
from configparser import ConfigParser

sys.path.append(str(Path(__file__).parent.parent / "simulation-code" / "static-environment"))
import evolution as ev

# List of replicates, doesn't change from Fig to fig 
replicates = [
    '0', 
              '1', '2', '3', '4', '5', '6', '7', '8', '9'
              ]

generations = [(x+1)*50000 for x in range(20)]

ALL_FIGS = ["Fig2", # Static env, fitness
            "Fig3", # Static env, complexity
            "Fig4", # Changing env
            "Fig5", # Robustness
            "Fig6", # Neutral drift
            "Supp1", # GRN examples
            # "Supp2", # This is now initializations, code is in its own script.
            "Supp3", # All fitness
            "Supp4", # All complexity
            "Supp5", # Changing env example
            "Supp6", # Generations to regain fitness
            "Supp7", # Changing env, no recombination
            "Supp8", # Changing env, recombination
            "Supp9", # Robustness, changing env
            "Supp10" # Correlations
            ]

low_mut_color = "#ff9896"
high_mut_color = "#d62728"
min_affinity_color = "#0E447B" 
mod_affinity_color = "#9ecae1"
sparse_affinity_color = "#3182bd"
recombination_color = '#2ca02c'
no_recombination_color = '#ffa500'
changing_env_color = "#7d33c2"
static_env_color = "#7d3607"

wspace = 0.2
hspace = 0.25

# Directories 
# Base repo root (two levels up if script is in analyses/)
repo_root = Path(__file__).resolve().parents[1]

# Directories (relative to repo root)
checkpoint_directory = repo_root / "checkpoints"
static_env_directory = checkpoint_directory / "static-environment"
changing_env_directory = checkpoint_directory / "changing-environment"
neutral_directory = checkpoint_directory / "neutral"
figure_directory = repo_root / "analyses" / "figures"
robustness_directory = repo_root / "analyses" / "robustness-outputs" 
example_config_path = repo_root / "config-files" / "testA" / "rep0.ini"

# Make sure figure directory exists
figure_directory.mkdir(parents=True, exist_ok=True)

# Figures to generate: command line args take priority
cli_selection = [a for a in sys.argv[1:] if not a.startswith("-")]

requested = cli_selection or [] # Enter fig names in list
requested_lower = [f.lower() for f in requested]

if 'all' in requested_lower:
    requested = ALL_FIGS
    
figures_to_generate = {k: (k in requested) for k in ALL_FIGS}

# Print settings before starting 
print("Generating figures: ", requested)
print("Loading checkpoints from: ", checkpoint_directory)
print("Loading robustness data from: ", robustness_directory)
print("Saving figures to: ", figure_directory)

# Register DEAP classes
config_parser = ConfigParser(converters={
    'intlist': lambda x: [int(i.strip()) for i in x.split(',')],
    'floatlist': lambda x: [float(i.strip()) for i in x.split(',')]
})
config_parser.read(example_config_path)
config = ev.load_config(config_parser)
ev.register_deap_tools(config)

def load_log_data(checkpoint):
    with open(checkpoint, 'rb') as f:
        results = pickle.load(f)

    log = results['logbook']
    df = pd.DataFrame({
        'gen': log.select('gen'),
        'meanFit': log.select('meanFit'),
        'meanComp': log.select('meanComp'),
        'diversity': log.select('diversity') # Not used in current plots for manuscript
    })

    # Drop duplicated generations (generated in older code versions when resuming simulations)
    df = df.drop_duplicates(subset='gen').set_index('gen').sort_index()
    # Limit results to 1 million generations for plotting 
    df = df[df.index <= 1000000]

    return df, results['population']

def auto_figsize(nrows, ncols, panel_size=3, aspect=1.0, wspace=wspace, hspace=hspace):
    fig_width = panel_size * ncols + wspace #* (ncols - 1)
    fig_height = panel_size * nrows * aspect + hspace #* (nrows - 1)
    return (fig_width, fig_height)


def plot_sem(tests, ax, directory, plot_comps = False, plot_fits = False, colors = [no_recombination_color, recombination_color]):
    for i, test in enumerate(tests):
        color = colors[i% len(colors)]
        
        comp_dfs = []
        fit_dfs = []
        
        for rep_name in replicates: 
            pattern = str(Path(directory) / test / f"{test}{rep_name}*.pkl")
            files = natsorted(
                filter(os.path.isfile, glob.glob(pattern)), alg = ns.IGNORECASE
            )
            if not files:
                
                print(f"File not found: {pattern}")
                continue

            checkpoint = files[-1]
            print(checkpoint)
            df, _ = load_log_data(checkpoint)
            
            comp_dfs.append(df['meanComp'])
            fit_dfs.append(df['meanFit'])

        if not comp_dfs:
            continue

        # Align all dataframes by index (generation), filling missing values with NaN
        comp_df = pd.concat(comp_dfs, axis=1)
        fit_df = pd.concat(fit_dfs, axis=1)

        # Compute mean and SEM with rolling average
        if plot_comps:
            comp_mean = comp_df.mean(axis=1)
            comp_sem = comp_df.sem(axis=1)
            ax.plot(comp_mean.index+1, comp_mean, color=color, alpha = 0.7)
            ax.fill_between(comp_mean.index+1, comp_mean - comp_sem, comp_mean + comp_sem, color=color, alpha=0.2)
            ax.set_ylim(-0.022, 1.022)

        if plot_fits:
            fit_mean = fit_df.mean(axis=1)
            fit_sem = fit_df.sem(axis=1)
            ax.plot(fit_mean.index+1, fit_mean, color=color, alpha = 0.7)
            ax.fill_between(fit_mean.index+1, fit_mean - fit_sem, fit_mean + fit_sem, color=color, alpha=0.2)
            ax.set_ylim(-2.23, 102.23)

        ax.set_xscale('log')
        ax.set_xlim(left=100)
        
def load_robustness_data(test, robustness_directory = robustness_directory):
    print(test)
    print(robustness_directory)
    filepath = Path(f"{robustness_directory}-{test}.measures")
    if not filepath.is_file():
        print(f"Robustness file not found: {filepath}")
        
    mean_robustness = []
    print("Reading: ", filepath)
    with open(filepath, 'r') as file: 
        for line in file:
            if re.match(r"^mean_", line):
                clean_line = re.sub(r"mean_\d+\s*=\s*", "", line).replace("[", "").replace("]", "").strip()
                values = [float(num) for num in clean_line.split(",") if num.strip()]
                mean_robustness.append(values)
    return mean_robustness

def robustness_dataframe(data, generations): 
    all_data = pd.DataFrame()
    for rep_idx, replicate in enumerate(data): 
        df = pd.DataFrame({
            "Replicate": str(rep_idx + 1),
            "Generation": generations,
            "Robustness": replicate
        })
        df = df[df.index <= 1000000]
        all_data = pd.concat([all_data, df], ignore_index=True)
    return all_data

def plot_correlation_data(tests, robustness_directory, output):
    conditions = pd.DataFrame({
        "Test": tests,
        "Initial binding affinity": ["Moderate", "Moderate", "Moderate", "Moderate",
                                     "Minimal", "Minimal", "Minimal", "Minimal",
                                     "Sparse", "Sparse", "Sparse", "Sparse"],
        "Recombination": ["No", "No", "Yes", "Yes",
                          "No", "No", "Yes", "Yes",
                          "No", "No", "Yes", "Yes",
                              ],
        "Mutation rate": ["Low", "High", "Low", "High",
                          "Low", "High", "Low", "High",
                          "Low", "High", "Low", "High",]
    })
    all_data = []
    for test in tests:
        filepath = os.path.join(robustness_directory, f"robustnessIndividual-{test}.measures")
        
        with open(filepath, 'r') as file:
            for line in file:
                match = re.match(fr"correlation_gen(\d+)\s*=\s*(-?\d+\.\d+)", line)
                if match:
                    generation = float(match.group(1))
                    correlation = float(match.group(2))
                    r2 = correlation*correlation
                    all_data.append({
                        'test': test,
                        'generation': generation,
                        'correlation': r2,
                    })
    df = pd.DataFrame(all_data)
    
    # Colour maps 
    row_names = ["Initial binding affinity", "Recombination", "Mutation rate"]
    row_colors = {
        "Initial binding affinity": {"Minimal": min_affinity_color, 
                                     "Moderate": mod_affinity_color, 
                                     "Sparse": sparse_affinity_color,}, 
        "Recombination": {"Yes": recombination_color, "No": no_recombination_color},
        "Mutation rate": {"Low": low_mut_color, "High": high_mut_color}
    }
    
    # Figure 
    fig = plt.figure(figsize=(8, 6))
    # No separate bottom legend row needed now
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 0.7], hspace=0.05)

    ax_main  = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1])

    # leave a right margin for legends
    fig.subplots_adjust(right=0.75)

    # main plot
    sns.violinplot(data=df, x="test", y="correlation", color='white', order=tests, inner=None, cut=0, ax=ax_main)
    sns.stripplot(data=df, x='test', y='correlation', size=2, order=tests, alpha=0.5, color='black', ax=ax_main)
    ax_main.set_ylabel("Pearson r² (complexity vs robustness)")
    ax_main.set_xlabel('')
    ax_main.set_xticks([])
    ax_main.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_main.set_ylim(0.0)

    # Category strip
    plot_category_strip(ax_strip, conditions, row_names, row_colors, tests)

    affinity_items = [
        ("moderate", row_colors["Initial binding affinity"]["Moderate"]),
        ("sparse",   row_colors["Initial binding affinity"]["Sparse"]),
        ("minimal",  row_colors["Initial binding affinity"]["Minimal"]),
    ]
    recomb_items = [
        ("yes", row_colors["Recombination"]["Yes"]),
        ("no",  row_colors["Recombination"]["No"]),
    ]
    mut_rate_items = [
        ("high", row_colors["Mutation rate"]["High"]),
        ("low",  row_colors["Mutation rate"]["Low"]),
    ]

    # stack the three legends top→bottom
    add_group_legend(ax_main, "Initial binding affinity", affinity_items, anchor_y=1.0)
    add_group_legend(ax_main, "Recombination",            recomb_items,  anchor_y=0.70)
    add_group_legend(ax_main, "Mutation rate",            mut_rate_items, anchor_y=0.46)
    
    plt.savefig(output, dpi=1200)
    plt.close()
    
def add_group_legend(ax, title, items, anchor_y):
        handles = [Patch(facecolor=c, edgecolor="none", label=lab) for lab, c in items]
        leg = ax.legend( handles=handles, title=title,
            loc="upper left",
            bbox_to_anchor=(1.02, anchor_y),
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.2,
            handletextpad=0.5,
        )
        ax.add_artist(leg)
        
def plot_category_strip(ax, conditions, row_names, row_colors, test_order, dot_size=150, barplot=False):
    # Order
    conditions = conditions.set_index("Test").loc[test_order].reset_index()

    mat, row_color_maps = [], []
    for row in row_names: 
        cats = pd.Categorical(conditions[row], categories=list(row_colors[row].keys()),
                              ordered=True)
        mat.append(cats.codes)
        code_to_color = dict(zip(range(len(cats.categories)),
                                 [row_colors[row][c] for c in cats.categories]))
        row_color_maps.append(code_to_color)
    mat = np.vstack(mat)
    
    # ax.set_xlim(-0.5, len(test_order)-0.5)
    ax.set_ylim(-0.5, len(row_names)-0.5)
    ax.invert_yaxis()
    
    # Gray bars
    for y in range(len(row_names)):
        ax.axhspan(y - 0.15, y +0.15, color = "#e0e0e0", zorder = 0)
    
    # Upset-style dots
    if barplot:
        x_positions = [None, 0.5, None, 2.5, None, 4.5, None, 6.5]
        for i_row, code_to_color in enumerate(row_color_maps):
            for i_col, x_pos in enumerate(x_positions):
                if i_col % 2 == 0:
                    continue
                else:
                    color = code_to_color[mat[i_row, i_col]]
                    ax.scatter(x_pos, i_row, s = dot_size, color=color, zorder=1)
    else:
        for i_row, code_to_color in enumerate(row_color_maps):
            for i_col in range(len(test_order)):
                color = code_to_color[mat[i_row, i_col]]
                ax.scatter(i_col, i_row, s=dot_size, color=color, zorder=1)     
    
    # Labels
    ax.yaxis.tick_right()
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)
    ax.set_xticks([])
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.tick_params(axis='y', length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)
        
def color_subplot_spines(axs, col_colors, col_labels, row_colors, row_labels, thick_bar=6, thin_bar=2):
    axs = np.atleast_2d(axs)
    nrows, ncols = axs.shape
    
    for row in range(nrows):
        for col in range(ncols):
            ax = axs[row,col].twinx()
            ax.spines['top'].set_color(col_colors[col])
            ax.spines['top'].set_linewidth(thick_bar if row == 0 else thin_bar)
            ax.spines['top'].set_position(('outward', thick_bar/2+2 if row == 0 else thin_bar/2 +2))
            ax.tick_params(right=False, labelright=False, length=0)
            
            # Right spine color based on row color
            lw = thick_bar if col == ncols - 1 else thin_bar
            ax.spines['right'].set_color(row_colors[row])
            ax.spines['right'].set_linewidth(lw)
            ax.spines['right'].set_position(('outward', lw/2+2))
            
            for side in ['top', 'right']:
                ax.spines[side].set_capstyle('butt')
                
            if col == ncols-1:
                ax.set_ylabel(row_labels[row],
                              color=row_colors[row],
                              fontsize='large',
                              rotation=270, labelpad=18)
        
    for col in range(ncols):
        axs[0, col].set_title(col_labels[col], fontsize='large', color=col_colors[col], pad=12)
        
def plot_robustness(data, ax, color):
    # Compute mean and SEM
    grouped = data.groupby("Generation")["Robustness"]
    mean = grouped.mean()
    sem = grouped.sem()

    ax.plot(mean.index.values, mean, color=color, alpha = 0.8)
    ax.fill_between(mean.index.values, mean - sem, mean + sem, alpha=0.2, color=color)

    ax.set_xticks([0,250000,500000,750000,1000000])

def axes_labels(labels, axs):
    for ax, label in zip(axs.flatten(), labels):
        ax.annotate(
            label,
            xy=(0, 1),                    # Top-left corner of the axes
            xycoords='axes fraction',     # Interpret xy as a fraction of axes size
            xytext=(-5, hspace*75),              # Offset in points: left (-x) and up (+y)
            textcoords='offset points',   # Interpret xytext as offset in points
            fontsize=12,
            fontweight='bold',
            va='top', 
            ha='right'
        )
    
# Code blocks for individual figures 
if figures_to_generate['Fig2']: 
    figsize = auto_figsize(nrows=1, ncols=2, panel_size=4, aspect=0.75)
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    axs[0].set_ylabel('Fitness (% of maximum)', size='large', labelpad=15)
    fig.supxlabel('Generations', size='large')

    plot_sem(['testE', 'testG'], axs[0], directory=static_env_directory, plot_fits=True)
    plot_sem(['testF', 'testH'], axs[1], directory=static_env_directory, plot_fits=True) 

    labels = ['A)', 'B)']
    axes_labels(labels, axs)
    
    color_subplot_spines(axs, col_colors=[low_mut_color, high_mut_color], col_labels=["Low mutation rate", "High mutation rate"],
                         row_colors=["#ffffff00"], row_labels=[""])
        
    custom_lines = [Line2D([0], [0], color=recombination_color, linewidth=2), Line2D([0], [0], color=no_recombination_color, linewidth=2)]
    plt.legend(custom_lines, ['Yes', 'No'], title='Recombination', loc='best')
    
    
    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=0.2)
    
    # Save
    fig.savefig(os.path.join(figure_directory, 'Fig2.png'), dpi=1200)
    plt.close()
    
if figures_to_generate['Fig3']:
    # Example: 1 row, 3 cols, each panel square
    figsize = auto_figsize(nrows=1, ncols=3, panel_size=3, aspect=1.0)
    fig, axs = plt.subplots(1, 3, figsize=(figsize[0], figsize[1]*0.95), sharex=True, sharey=True)
    axs[0].set_ylabel('Complexity (1 - Gini)', size='large', labelpad=15)
    
    plot_sem(['testJ', 'testL'], axs[0], directory=static_env_directory, plot_comps=True)
    plot_sem(['testF', 'testH'], axs[1], directory=static_env_directory, plot_comps=True) 
    plot_sem(['testB', 'testD'], axs[2], directory=static_env_directory, plot_comps=True)

    labels = ['A)', 'B)', 'C)']
    axes_labels(labels, axs)
    color_subplot_spines(axs, row_colors=["#ffffff00"], row_labels=[""], 
                         col_colors=[sparse_affinity_color, min_affinity_color, mod_affinity_color],
                         col_labels=["Sparse affinity", "Minimal affinity", "Moderate affinity"])
    
    custom_lines = [Line2D([0], [0], color=recombination_color, linewidth=2), Line2D([0], [0], color=no_recombination_color, linewidth=2)]
    plt.legend(custom_lines, ['Yes', 'No'], title='Recombination', loc='lower right')
    fig.supxlabel('Generations', size='large')

    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=0.2)

    # Save
    fig.savefig(os.path.join(figure_directory, 'Fig3.png'), dpi=1200)
    plt.close()
    
if figures_to_generate['Fig4']:
    figsize = auto_figsize(nrows=1, ncols=2, panel_size=4, aspect=0.75)
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    axs[0].set_ylabel('Complexity (1 - Gini)', size='large', labelpad=15)
    fig.supxlabel('Generations', size='large')

    
    plot_sem(['testE'], axs[0], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testE'], axs[0], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    plot_sem(['testG'], axs[1], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testG'], axs[1], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    
    axs[0].vlines(300000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[1].vlines(300000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[0].set_xlim(1e5, 1.1e6)
    axs[1].set_xlim(1e5, 1.1e6)
    axs[0].set_ylim([0.2,0.6])
    
    labels = ['A)', 'B)']
    axes_labels(labels, axs)
    color_subplot_spines(axs, col_colors=[no_recombination_color,recombination_color], col_labels=['No recombination', 'Recombination'],
                         row_colors=["#ffffff00"], row_labels=[""])
    
    custom_lines = [Line2D([0], [0], color=changing_env_color, linewidth=2), Line2D([0], [0], color=static_env_color, linewidth=2)]
    plt.legend(custom_lines, ['Changing', 'Static'], title='Environment', loc='lower right')
    
    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=0.2)

    # Save
    fig.savefig(os.path.join(figure_directory, 'Fig4.png'), dpi=1200)
    plt.close()
    
if figures_to_generate['Fig5']:
    # This is the robustness figure, need to write the code still
    figsize = auto_figsize(nrows=3, ncols=2, panel_size=4, aspect=0.75)
    fig, axs = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
    fig.text(x=0.5, y=0.6/figsize[1]/2, s= "Generations", ha='center', va='bottom', size='large')
    fig.text(x=0.6/figsize[0]/2, y=0.5, s="Robustness", ha='left', va='center', size='large', rotation=90)

    
    plot_robustness(robustness_dataframe(load_robustness_data('testA', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[2,0], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testC', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[2,0], color=recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testB', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[2,1], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testD', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[2,1], color=recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testE', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[1,0], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testG', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[1,0], color=recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testF', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[1,1], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testH', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[1,1], color=recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testI', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[0,0], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testK', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[0,0], color=recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testJ', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[0,1], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testL', robustness_directory=robustness_directory / "static-environment" / "robustness"), generations), axs[0,1], color=recombination_color)
    labels = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)']
    
    axes_labels(labels, axs)
    color_subplot_spines(axs=axs, col_colors=[low_mut_color, high_mut_color], col_labels=["Low mutation rate", "High mutation rate"],
                         row_colors=[sparse_affinity_color, min_affinity_color, mod_affinity_color], 
                         row_labels=["Sparse affinity", "Minimal affinity", "Moderate affinity"])
    
    custom_lines = [Line2D([0], [0], color=recombination_color, linewidth=2), Line2D([0], [0], color=no_recombination_color, linewidth=2)]
    plt.legend(custom_lines, ['Yes', 'No'], title='Recombination', loc='lower right')
    
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
        
    # Save
    fig.savefig(os.path.join(figure_directory, 'Fig5.png'), dpi=1200)
    plt.close()

if figures_to_generate['Fig6']:
    # Neutral evolution 
    figsize = auto_figsize(nrows=2, ncols=3, panel_size=3, aspect=1)
    fig, axs = plt.subplots(2, 3, figsize=(figsize[0], figsize[1]*0.95), sharex=True, sharey=True)
    fig.text(x=0.5, y=0.1/figsize[1], s= "Generations", ha='center', va='bottom', size='large')
    fig.text(x=0.6/figsize[0]/2, y=0.5, s="Complexity (1 - Gini)", ha='left', va='center', size='large', rotation=90)
    
    plot_sem(['testB'], axs[0,2], directory=neutral_directory, plot_comps=True, colors=['gray'])
    plot_sem(['testB'], axs[0,2], directory=static_env_directory, plot_comps=True, colors=[no_recombination_color])
    plot_sem(['testD'], axs[1,2], directory=neutral_directory, plot_comps=True, colors=['gray'])
    plot_sem(['testD'], axs[1,2], directory=static_env_directory, plot_comps=True, colors=[recombination_color])
    plot_sem(['testF'], axs[0,1], directory=neutral_directory, plot_comps=True, colors=['gray'])
    plot_sem(['testF'], axs[0,1], directory=static_env_directory, plot_comps=True, colors=[no_recombination_color])
    plot_sem(['testH'], axs[1,1], directory=neutral_directory, plot_comps=True, colors=['gray'])
    plot_sem(['testH'], axs[1,1], directory=static_env_directory, plot_comps=True, colors=[recombination_color])
    plot_sem(['testJ'], axs[0,0], directory=neutral_directory, plot_comps=True, colors=['gray'])
    plot_sem(['testJ'], axs[0,0], directory=static_env_directory, plot_comps=True, colors=[no_recombination_color])
    plot_sem(['testL'], axs[1,0], directory=neutral_directory, plot_comps=True, colors=['gray'])
    plot_sem(['testL'], axs[1,0], directory=static_env_directory, plot_comps=True, colors=[recombination_color])
    
    custom_lines = [Line2D([0], [0], color=no_recombination_color, linewidth=2), 
                Line2D([0], [0], color='gray', linewidth=2), ]
    axs[0,2].legend(custom_lines, ['Selection', 'Neutral drift'], title='', loc='lower right')
    custom_lines = [Line2D([0], [0], color=recombination_color, linewidth=2), 
                Line2D([0], [0], color='gray', linewidth=2), ]
    axs[1,2].legend(custom_lines, ['Selection', 'Neutral drift'], title='', loc='lower right')
    
    labels = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)']
    axes_labels(labels, axs)
    color_subplot_spines(axs=axs, col_colors=[sparse_affinity_color, min_affinity_color, mod_affinity_color],
                         col_labels=["Sparse affinity", "Minimal affinity", "Moderate affinity"],
                         row_colors=[no_recombination_color, recombination_color], row_labels=['No recombination', 'Recombination'])

    
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    
    plt.savefig(os.path.join(figure_directory, 'Fig6.png'), dpi=1200)
    
if figures_to_generate['Supp1']: # Example 
    n_TFs = 20
    n_targets = 200
    # Create a sparse individual 
    sparse = np.full((n_TFs,n_TFs+n_targets), -(0.517 * math.log(n_targets) + 2.516)) 
    for col in range(220):
        row = random.randint(0,n_TFs-1)
        sparse[row,col] = 0.517*math.log(20)+2.516
            
    # Create a minimal individual 
    minimal = np.full((n_TFs,n_TFs+n_targets), -(0.517 * math.log(n_targets) + 2.516)) 
    
    # Create a moderate individual
    moderate = np.full((n_TFs,n_TFs+n_targets), 0.0) 
    
    # Load an individual at the plateau 
    pattern = str(Path(changing_env_directory) / "testD" / f"testD0_gen1035000.pkl")
    _, population = load_log_data(pattern)
    individual = np.reshape(population[0], newshape=(n_TFs, n_TFs+n_targets))
    
    # Plot
    figsize = auto_figsize(nrows=4, ncols=1, panel_size=4, aspect=0.3)
    fig, axs = plt.subplots(4,1, figsize=figsize)
    vmin = -5.0
    vmax = 5.0
    sns.heatmap(minimal, ax=axs[0], vmin=vmin, vmax=vmax, cmap='Blues_r', cbar=False)
    sns.heatmap(moderate, ax=axs[1], vmin=vmin, vmax=vmax, cmap='Blues_r', cbar=False)
    sns.heatmap(sparse, ax=axs[2], vmin=vmin, vmax=vmax, cmap='Blues_r', cbar=False)
    sns.heatmap(individual, ax=axs[3], vmin=vmin, vmax=vmax, cmap='Blues_r',cbar=False)
    
    for ax, title, color in zip(axs.flat, ['Minimal', 'Moderate', 'Sparse', 'Converged'], [min_affinity_color, mod_affinity_color, sparse_affinity_color, 'black']):
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("TFs")
        ax.set_title(title, color=color)
        
    fig.subplots_adjust(right=0.7, hspace=hspace*2)
    cbar_ax = fig.add_axes([0.8,0.15,0.03,0.7])
    cbar = plt.colorbar(axs[0].collections[0], cax=cbar_ax)
    cbar.set_label("log binding affinity")          # Label
    cbar.outline.set_visible(False)     
    axs[3].set_xlabel("TFs and Targets")
    
    fig.savefig(os.path.join(figure_directory, 'Supp1.png'), dpi=1200)
    plt.close()

if figures_to_generate['Supp3']:  
    figsize = auto_figsize(nrows=3, ncols=2, panel_size=4, aspect=0.75)
    fig, axs = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
    fig.text(x=0.5, y=0.6/figsize[1]/2, s= "Generations", ha='center', va='bottom', size='large')
    fig.text(x=0.6/figsize[0]/2, y=0.5, s="Fitness (% of maximum)", ha='left', va='center', size='large', rotation=90)

    plot_sem(['testI', 'testK'], axs[0, 0], directory=static_env_directory, plot_fits=True)
    plot_sem(['testJ', 'testL'], axs[0, 1], directory=static_env_directory, plot_fits=True) 
    plot_sem(['testE', 'testG'], axs[1, 0], directory=static_env_directory, plot_fits=True) 
    plot_sem(['testF', 'testH'], axs[1, 1], directory=static_env_directory, plot_fits=True) 
    plot_sem(['testA', 'testC'], axs[2, 0], directory=static_env_directory, plot_fits=True) 
    plot_sem(['testB', 'testD'], axs[2, 1], directory=static_env_directory, plot_fits=True) 

    labels = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)']
    axes_labels(labels, axs)
    
    color_subplot_spines(axs=axs, col_colors=[low_mut_color, high_mut_color],
                         col_labels=["Low mutation rate", "High mutation rate"],
                         row_colors=[sparse_affinity_color, min_affinity_color, mod_affinity_color], 
                         row_labels=["Sparse affinity", "Minimal affinity", "Moderate affinity"])
        
    custom_lines = [Line2D([0], [0], color=recombination_color, linewidth=2), Line2D([0], [0], color=no_recombination_color, linewidth=2)]
    plt.legend(custom_lines, ['Yes', 'No'], title='Recombination', loc='best')

    # Save
    fig.savefig(os.path.join(figure_directory, 'Supp3.png'), dpi=1200)
    plt.close()
    
if figures_to_generate['Supp4']: 
    figsize = auto_figsize(nrows=3, ncols=2, panel_size=4, aspect=0.75)
    fig, axs = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
    fig.text(x=0.5, y=0.6/figsize[1]/2, s= "Generations", ha='center', va='bottom', size='large')
    fig.text(x=0.6/figsize[0]/2, y=0.5, s="Complexity (1 - Gini)", ha='left', va='center', size='large', rotation=90)

    plot_sem(['testI', 'testK'], axs[0, 0], directory=static_env_directory, plot_comps=True)
    plot_sem(['testJ', 'testL'], axs[0, 1], directory=static_env_directory, plot_comps=True) 
    plot_sem(['testE', 'testG'], axs[1, 0], directory=static_env_directory, plot_comps=True) 
    plot_sem(['testF', 'testH'], axs[1, 1], directory=static_env_directory, plot_comps=True) 
    plot_sem(['testA', 'testC'], axs[2, 0], directory=static_env_directory, plot_comps=True) 
    plot_sem(['testB', 'testD'], axs[2, 1], directory=static_env_directory, plot_comps=True) 

    labels = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)']
    axes_labels(labels, axs)
    
    color_subplot_spines(axs=axs, col_colors=[low_mut_color, high_mut_color],
                         col_labels=["Low mutation rate", "High mutation rate"],
                         row_colors=[sparse_affinity_color, min_affinity_color, mod_affinity_color], 
                         row_labels=["Sparse affinity", "Minimal affinity", "Moderate affinity"])
        
    custom_lines = [Line2D([0], [0], color=recombination_color, linewidth=2), Line2D([0], [0], color=no_recombination_color, linewidth=2)]
    plt.legend(custom_lines, ['Yes', 'No'], title='Recombination', loc='best')
    
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # Save
    fig.savefig(os.path.join(figure_directory, 'Supp4.png'), dpi=1200)
    plt.close()
    
if figures_to_generate['Supp5']:
    # Changing env demonstration for one sample, use testD, rep0 
    pattern = str(Path(changing_env_directory) / "testD" / f"testD0_gen1035000.pkl")
    df, _ = load_log_data(pattern)
    fig, ax = plt.subplots(1,1, figsize = (6,4))
    ax.plot(df.index, df['meanFit'], color=recombination_color, alpha = 0.8)
    ax.set_xlim(149000, 152500)
    ax.set_ylim(-2.23, 102.23)
    ax.set_xlabel('Generations', size='large')
    ax.set_ylabel('Fitness (% of maximum)', size='large')
    
    ax.vlines(150000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    
    fig.savefig(os.path.join(figure_directory, 'Supp5.png'), dpi=1200)
    
if figures_to_generate['Supp6']:
    # Similar violin plot for time to regain original fitness
    data = pd.DataFrame({
        "Test": ['testA', 'testB', 'testC', 'testD', 'testE', 'testF', 
                           'testG', 'testH'],
        "Initial binding affinity": ["Moderate", "Moderate", "Moderate", "Moderate",
                                     "Minimal", "Minimal", "Minimal", "Minimal",],
        "Recombination": ["No", "No", "Yes", "Yes",
                          "No", "No", "Yes", "Yes",
                              ],
        "Mutation rate": ["Low", "High", "Low", "High",
                          "Low", "High", "Low", "High",],
        "Generations": [3210, 1001, 1278, 440, 3837, 1107, 2143, 585]
    })
    
    test_order = ['testA', 'testC', 'testB', 'testD', 'testE', 'testG', 'testF', 'testH']
    
    fig = plt.figure(figsize=(6,6))
    gs = GridSpec(nrows = 2, ncols=1, height_ratios=[3, 0.5], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax_main)
    fig.subplots_adjust(right=0.6)
    
    # Barplot 
    sns.barplot(data=data, x='Test', y='Generations', hue='Recombination', 
                hue_order=['Yes', 'No'], palette={'Yes': recombination_color, 'No': no_recombination_color}, ax=ax_main,
                order=test_order)
    xcenters = dict(zip(test_order, ax_main.get_xticks()))
    ax_main.set_xlabel('')
    ax_main.set_xticks([])
    ax_main.tick_params(axis='x', bottom=False, labelbottom=False)
    ax_main.set_ylim(0.0)
    ax_main.set_ylabel("Mean generations between environment changes")
    
    
    row_names = ["Initial binding affinity", "Mutation rate"]
    row_colors = {
        "Initial binding affinity": {"Minimal": min_affinity_color, 
                                     "Moderate": mod_affinity_color, }, 
        "Recombination": {"Yes": recombination_color, "No": no_recombination_color},
        "Mutation rate": {"Low": low_mut_color, "High": high_mut_color}
    }
    
    plot_category_strip(ax_strip, conditions=data, row_names=row_names, row_colors=row_colors, 
                        test_order=test_order, barplot=True)
    affinity_items = [
        ("moderate", row_colors["Initial binding affinity"]["Moderate"]),
        ("minimal",  row_colors["Initial binding affinity"]["Minimal"]),
    ]
    recombination_items = [
        ("yes", row_colors['Recombination']['Yes']),
        ("no", row_colors['Recombination']['No'])
    ]
    mut_rate_items = [
        ("high", row_colors["Mutation rate"]["High"]),
        ("low",  row_colors["Mutation rate"]["Low"]),
    ]

    # stack the three legends top→bottom
    add_group_legend(ax_main, "Initial binding affinity", affinity_items, anchor_y=1.0)
    add_group_legend(ax_main, "Recombination",            recombination_items, anchor_y=0.7 )
    add_group_legend(ax_main, "Mutation rate",            mut_rate_items, anchor_y=0.46)
    
    plt.savefig(os.path.join(figure_directory, 'Supp6.png'), dpi=1200)
    plt.close()
        
if figures_to_generate['Supp7']:
    # Changing env, no recombination 
    figsize = auto_figsize(nrows=2, ncols=2, panel_size=4, aspect=0.75)
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    fig.text(x=0.6/figsize[0]/2, y=0.5, s="Complexity (1 - Gini)", ha='left', va='center', size='large', rotation=90)
    
    plot_sem(['testE'], ax=axs[0,0], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testE'], ax=axs[0,0], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    plot_sem(['testF'], ax=axs[0,1], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testF'], ax=axs[0,1], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    plot_sem(['testA'], ax=axs[1,0], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testA'], ax=axs[1,0], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    plot_sem(['testB'], ax=axs[1,1], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testB'], ax=axs[1,1], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    
    axs[0,0].vlines(300000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[1,0].vlines(300000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[0,1].vlines(150000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[1,1].vlines(150000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[0,0].set_xlim(1e5, 1.1e6)
    
    labels = ['A)', 'B)', 'C)', 'D)']
    axes_labels(labels, axs)
    
    color_subplot_spines(axs, col_colors=[low_mut_color,high_mut_color], col_labels=['Low mutation rate', 'High mutation rate'],
                         row_colors=[min_affinity_color, mod_affinity_color], row_labels=["Minimal affinity", "Moderate affinity"])
    
    custom_lines = [Line2D([0], [0], color=changing_env_color, linewidth=2), Line2D([0], [0], color=static_env_color, linewidth=2)]
    plt.legend(custom_lines, ['Changing', 'Static'], title='Environment', loc='best')
    fig.supxlabel('Generations')
    
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # Save
    fig.savefig(os.path.join(figure_directory, 'Supp7.png'), dpi=1200)
    plt.close()   
    
if figures_to_generate['Supp8']:
    #Changing env, recombination
    figsize = auto_figsize(nrows=2, ncols=2, panel_size=4, aspect=0.75)
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    fig.text(x=0.6/figsize[0]/2, y=0.5, s="Complexity (1 - Gini)", ha='left', va='center', size='large', rotation=90)
    
    plot_sem(['testG'], ax=axs[0,0], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testG'], ax=axs[0,0], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    plot_sem(['testH'], ax=axs[0,1], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testH'], ax=axs[0,1], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    plot_sem(['testC'], ax=axs[1,0], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testC'], ax=axs[1,0], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    plot_sem(['testD'], ax=axs[1,1], directory=changing_env_directory, plot_comps=True, colors=[changing_env_color])
    plot_sem(['testD'], ax=axs[1,1], directory=static_env_directory, plot_comps=True, colors=[static_env_color])
    
    axs[0,0].vlines(300000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[1,0].vlines(300000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[0,1].vlines(150000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[1,1].vlines(150000,ymin=-2.22724586064651, ymax= 102.22724586064651, linestyle='dashed', color='black', alpha=0.5)
    axs[0,0].set_xlim(1e5, 1.1e6)
    
    labels = ['A)', 'B)', 'C)', 'D)']
    axes_labels(labels, axs)
    
    color_subplot_spines(axs, col_colors=[low_mut_color,high_mut_color], col_labels=['Low mutation rate', 'High mutation rate'],
                         row_colors=[min_affinity_color, mod_affinity_color], row_labels=["Minimal affinity", "Moderate affinity"])
    
        
    custom_lines = [Line2D([0], [0], color=changing_env_color, linewidth=2), Line2D([0], [0], color=static_env_color, linewidth=2)]
    plt.legend(custom_lines, ['Changing', 'Static'], title='Environment', loc='best')
    fig.supxlabel('Generations')
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # Save
    fig.savefig(os.path.join(figure_directory, 'Supp8.png'), dpi=1200)
    plt.close()    
    
if figures_to_generate['Supp9']:
    # Robustness, changing env
    figsize = auto_figsize(nrows=2, ncols=2, panel_size=4, aspect=0.75)
    fig, axs = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    fig.text(x=0.6/figsize[0]/2, y=0.5, s="Robustness", ha='left', va='center', size='large', rotation=90)
        
    plot_robustness(robustness_dataframe(load_robustness_data('testA', robustness_directory=robustness_directory / "changing-environment" / 'robustness-change'), generations[5:]), axs[1,0], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testC', robustness_directory=robustness_directory / "changing-environment" / 'robustness-change'), generations[5:]), axs[1,0], color=recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testB', robustness_directory=robustness_directory / "changing-environment" / 'robustness-change'), generations[2:]), axs[1,1], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testD', robustness_directory=robustness_directory / "changing-environment" / 'robustness-change'), generations[2:]), axs[1,1], color=recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testE', robustness_directory=robustness_directory / "changing-environment" / 'robustness-change'), generations[5:]), axs[0,0], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testG', robustness_directory=robustness_directory / "changing-environment" / 'robustness-change'), generations[5:]), axs[0,0], color=recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testF', robustness_directory=robustness_directory / "changing-environment" / 'robustness-change'), generations[2:]), axs[0,1], color=no_recombination_color)
    plot_robustness(robustness_dataframe(load_robustness_data('testH', robustness_directory=robustness_directory / "changing-environment" / 'robustness-change'), generations[2:]), axs[0,1], color=recombination_color)
    
    # Vertical lines
    ymin, ymax = axs[0,0].get_ylim()
    axs[0,0].vlines(300000,ymin=ymin, ymax= ymax, linestyle='dashed', color='black', alpha=0.5)
    axs[1,0].vlines(300000,ymin=ymin, ymax= ymax, linestyle='dashed', color='black', alpha=0.5)
    axs[0,1].vlines(150000,ymin=ymin, ymax= ymax, linestyle='dashed', color='black', alpha=0.5)
    axs[1,1].vlines(150000,ymin=ymin, ymax= ymax, linestyle='dashed', color='black', alpha=0.5)
    
     # Column titles + colored spines
    color_subplot_spines(axs, col_colors=[low_mut_color, high_mut_color], col_labels=["Low mutation rate", "High mutation rate"],
                         row_colors=[min_affinity_color, mod_affinity_color], row_labels=["Minimal affinity", "Moderate affinity"], )
    
    labels = ['A)', 'B)', 'C)', 'D)']
    axes_labels(labels, axs)
    
    custom_lines = [Line2D([0], [0], color=recombination_color, linewidth=2), Line2D([0], [0], color=no_recombination_color, linewidth=2)]
    plt.legend(custom_lines, ['Yes', 'No'], title='Recombination', loc='best')
    fig.supxlabel('Generations')
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    
    # Save
    fig.savefig(os.path.join(figure_directory, 'Supp9.png'), dpi=1200)
    plt.close()
    
if figures_to_generate['Supp10']:
    # Violin plot for correlation data
    plot_correlation_data(['testA', 'testB', 'testC', 'testD', 'testE', 'testF', 
                           'testG', 'testH', 'testI', 'testJ', 'testK', 'testL'],
                          robustness_directory / "static-environment", figure_directory / 'Supp10.png')
    
