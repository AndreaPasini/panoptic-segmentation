import os

import numpy as np
from matplotlib import pyplot as plt, ticker
from matplotlib.ticker import MaxNLocator


def plot_comparison_both(mink, maxk, sims_df, sims_df_agg, kmed_df, suffix2, output_path):
    fig, ax = plt.subplots(1, 2, figsize=[9, 3], sharey=True)
    # Coverage of the two methods
    ax[0].plot(np.arange(mink, maxk + 1),
               sims_df.loc[sims_df['N. graphs'] >= mink]['Coverage'], label='SImS',
               marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
    if sims_df_agg is not None:
        ax[0].plot(np.arange(mink, maxk + 1),
                   sims_df_agg.loc[sims_df_agg['N. graphs'] >= mink]['Coverage'], label='SImS (agg)',
                   marker='o', markersize='4', color='#ff7f00', markerfacecolor='#fc8d62')
    ax[0].plot(np.arange(mink, maxk + 1), kmed_df['Coverage'], label='KMedoids',
               marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')

    # Overlap of the two methods
    # ax[0].plot(np.arange(RUN_CONFIG.mink, RUN_CONFIG.maxk + 1),
    #            sims_df.loc[sims_df['N. graphs'] >= RUN_CONFIG.mink]['Coverage-overlap'], label='SImS-ov',
    #            marker='o', markersize='4', color='#005a32', markerfacecolor='#b2df8a')
    # ax[0].plot(np.arange(RUN_CONFIG.mink, RUN_CONFIG.maxk + 1), kmed_df['Coverage-overlap'], label='KMedoids-ov',
    #            marker='o', markersize='4', color='#0c2c84', markerfacecolor='#a6cee3')

    ax[0].set_xlabel('# graphs (k)')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylabel('coverage')
    ax[0].grid(axis='y', which="major", linewidth=1)
    ax[0].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')

    ax[1].plot(np.arange(mink, maxk + 1),
               sims_df.loc[sims_df['N. graphs'] >= mink]['Diversity'], label='SImS',
               marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
    if sims_df_agg is not None:
        ax[1].plot(np.arange(mink, maxk + 1),
                   sims_df_agg.loc[sims_df_agg['N. graphs'] >= mink]['Diversity'], label='SImS (agg)',
                   marker='o', markersize='4', color='#ff7f00', markerfacecolor='#fc8d62')
    ax[1].plot(np.arange(mink, maxk + 1), kmed_df['Diversity'], label='KMedoids',
               marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')

    # Diversity edge:
    # ax[1].plot(np.arange(RUN_CONFIG.mink, RUN_CONFIG.maxk + 1), sims_df.loc[sims_df['N. graphs'] >= RUN_CONFIG.mink]['Diversity-ne'], label='SImS-edge',
    #            marker='o', markersize='4', color='#005a32', markerfacecolor='#b2df8a')
    # ax[1].plot(np.arange(RUN_CONFIG.mink,RUN_CONFIG.maxk + 1), kmed_df['Diversity-ne'], label='KMedoids-edge',
    #            marker='o', markersize='4', color='#0c2c84', markerfacecolor='#a6cee3')

    ax[1].set_xlabel('# graphs (k)')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel('diversity')
    ax[1].grid(axis='y', which="major", linewidth=1)
    ax[1].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[1].legend(bbox_to_anchor=(1.6, 0.4), loc="lower right")
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'evaluationBoth{suffix2}.eps'), bbox_inches='tight')


def plot_comparison_coverage(mink, maxk, sims_df, sims_df_agg, kmed_df, suffix2, output_path):
    fig, ax = plt.subplots(1, 2, figsize=[9, 3], sharey=True)
    # Coverage of the two methods
    ax[0].plot(np.arange(mink, maxk + 1),
               sims_df.loc[sims_df['N. graphs'] >= mink]['Coverage'], label='SImS',
               marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
    if sims_df_agg is not None:
        ax[0].plot(np.arange(mink, maxk + 1),
                   sims_df_agg.loc[sims_df_agg['N. graphs'] >= mink]['Coverage'], label='SImS (agg)',
                   marker='o', markersize='4', color='#ff7f00', markerfacecolor='#fc8d62')
    ax[0].plot(np.arange(mink, maxk + 1), kmed_df['Coverage'], label='KMedoids',
               marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')

    # Overlap of the two methods
    ax[1].plot(np.arange(mink, maxk + 1),
               sims_df.loc[sims_df['N. graphs'] >= mink]['Coverage-overlap'], label='SImS',
               marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
    if sims_df_agg is not None:
        ax[1].plot(np.arange(mink, maxk + 1),
                   sims_df_agg.loc[sims_df_agg['N. graphs'] >= mink]['Coverage-overlap'], label='SImS (agg)',
                   marker='o', markersize='4', color='#ff7f00', markerfacecolor='#fc8d62')
    ax[1].plot(np.arange(mink, maxk + 1), kmed_df['Coverage-overlap'], label='KMedoids',
               marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')

    ax[0].set_xlabel('# graphs (k)')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylabel('coverage')
    ax[0].grid(axis='y', which="major", linewidth=1)
    ax[0].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[1].set_xlabel('# graphs (k)')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel('coverage degree')
    ax[1].grid(axis='y', which="major", linewidth=1)
    ax[1].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[1].legend(bbox_to_anchor=(1.6, 0.4), loc="lower right")
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    ax[1].yaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'evaluationCoverage{suffix2}.eps'), bbox_inches='tight')


def plot_comparison_diversity(mink, maxk, sims_df, sims_df_agg, kmed_df, suffix2, output_path):
    fig, ax = plt.subplots(1, 2, figsize=[9, 3], sharey=True)

    ax[0].set_xlabel('# graphs (k)')
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].set_ylabel('node diversity')
    ax[0].grid(axis='y', which="major", linewidth=1)
    ax[0].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[0].plot(np.arange(mink, maxk + 1),
               sims_df.loc[sims_df['N. graphs'] >= mink]['Diversity'], label='SImS',
               marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
    if sims_df_agg is not None:
        ax[0].plot(np.arange(mink, maxk + 1),
                   sims_df_agg.loc[sims_df_agg['N. graphs'] >= mink]['Diversity'], label='SImS (agg)',
                   marker='o', markersize='4', color='#ff7f00', markerfacecolor='#fc8d62')
    ax[0].plot(np.arange(mink, maxk + 1), kmed_df['Diversity'], label='KMedoids',
               marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')

    # Diversity edge:
    ax[1].plot(np.arange(mink, maxk + 1),
               sims_df.loc[sims_df['N. graphs'] >= mink]['Diversity-ne'], label='SImS',
               marker='o', markersize='4', color='#33a02c', markerfacecolor='#b2df8a')
    if sims_df_agg is not None:
        ax[1].plot(np.arange(mink, maxk + 1),
                   sims_df_agg.loc[sims_df_agg['N. graphs'] >= mink]['Diversity-ne'], label='SImS (agg)',
                   marker='o', markersize='4', color='#ff7f00', markerfacecolor='#fc8d62')
    ax[1].plot(np.arange(mink, maxk + 1), kmed_df['Diversity-ne'], label='KMedoids',
               marker='o', markersize='4', color='#1f78b4', markerfacecolor='#a6cee3')

    ax[1].set_xlabel('# graphs (k)')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[1].set_ylabel('edge diversity')
    ax[1].grid(axis='y', which="major", linewidth=1)
    ax[1].grid(axis='y', which="minor", linewidth=0.5, color='#cccccc')
    ax[1].legend(bbox_to_anchor=(1.6, 0.4), loc="lower right")
    ax[0].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[1].yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[0].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax[1].yaxis.set_tick_params(labelbottom=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'evaluationDiversity{suffix2}.eps'), bbox_inches='tight')