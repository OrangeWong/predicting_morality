# Modify from Udacity finding donor visuals

import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
import os

def evaluate(results, scores):
    """
    Visualization code to display results of various learners.
    
    inputs:
      - results: a dict that the results of learner stored
      - f1: the f1 score for the benchmark predictor
      - mc: the matthews coefficient score for the benchmark predictor
    """
  
    # Create figure
    fig, ax = plt.subplots(2, 3, figsize = (13,8))

    # Constants
    bar_width = 0.2
    colors = ['#6ca659','#ab62c0','#c2843c', '#648ace'] #ca556a
    ylabels = ["Time (in seconds)", "F-score", "MCC-score"]
    titles = ["Training Set", "Testing Set"]
    benchmark_line_style = {'xmin': -0.1, 'xmax': 3.0, 'linewidth': 2, 'color': 'k', 'linestyle': 'dashed'}
    benchmark_score = [scores['f1_train'], scores['mcc_train'], scores['f1_test'], scores['mcc_test']]
    
    # Super loop to plot four panels of data
    for k, learner in enumerate(results.keys()): # first level: learner type
        for j, metric in enumerate(['train_time', 'f1_train', 'mcc_train',
                                    'pred_time', 'f1_test', 'mcc_test']):
            for i in np.arange(3):
                # Creative plot code
                ax[j//3, j%3].bar(bar_width/2+i+k*bar_width, results[learner][i][metric], width = bar_width, color = colors[k])
                ax[j//3, j%3].set_xticks([0.4, 1.4, 2.4])
                ax[j//3, j%3].set_xticklabels(["1%", "10%", "100%"])
                ax[j//3, j%3].set_xlabel("Training Set Size")
                ax[j//3, j%3].set_xlim((-0.1, 3.0))
    
    # Add unique y-labels
    for j in range(6):
        ax[j//3, j%3].set_ylabel(ylabels[j%3])
    
    # Add titles
    ax[0, 0].set_title("Model Training")
    ax[1, 0].set_title("Model Predicting")
    for j in range(4):
        ax[j//2, j%2+1].set_title(titles[j//2])
    
    # Add horizontal lines for naive predictors
    for j in range(4):
        ax[j//2, j%2+1].axhline(y = benchmark_score[j], **benchmark_line_style)
    
    # Set y-limits for score panels
    for j in range(4):
        ax[j//2, j%2+1].set_ylim((0, 1))
    
    # Create patches for the legend
    patches = []
    for i, learner in enumerate(results.keys()):
        patches.append(mpatches.Patch(color = colors[i], label = learner))
    plt.legend(handles = patches, bbox_to_anchor = (-0.85, 2.25),
               loc = 'upper center', borderaxespad = 0., ncol = 4, fontsize = 'large')
    
    # Aesthetics
    #plt.suptitle("Performance Metrics for Four Supervised Learning Models", fontsize = 16, y = 1.10)
    plt.tight_layout()
    if not os.path.exists('plot'):
        os.makedirs('plot')
    plt.savefig('plot/performance_metrics')
    plt.show()
    
def feature_plot(importances, X_train, y_train, n=10):
    """
    Display the first n most predictive features.
    
    inputs:
      - importances: The feature importances generated from Sklearn learner
      - X_train: Training data
      - y_train: Training label
      - n: number of importances
    """
    # Display the first n most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:n]]
    values = importances[indices][:n]

    # Creat the plot
    fig = plt.figure(figsize = (1.5*n,5))
    plt.title("Normalized Weights for First {} Most Predictive Features".format(n), fontsize = 16)
    plt.bar(np.arange(len(values)), values, width = 0.6, align="center", color = '#648ace', \
          label = "Feature Weight")
    plt.bar(np.arange(len(values)) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#ca556a', \
          label = "Cumulative Feature Weight")
    plt.xticks(np.arange(len(values)), columns)
    plt.xlim((-0.5, len(values)-0.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    if not os.path.exists('plot'):
        os.makedirs('plot')
    plt.savefig('plot/feature_plot')
    
    plt.show()   