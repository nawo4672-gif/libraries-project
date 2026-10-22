#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument("--input", default="../10.13-15/METABRIC_RNA_Mutation.csv",
                    help="Kaggle breast cancer dataset")
parser.add_argument("--output", default="../10.13-15/img/breast-cancer-rna-analysis.png")
args = parser.parse_args()

def main():
    df = pd.read_csv(args.input) # read data into pandas DataFrame
    ### therapy and survival summaries
    therapy_cols = ['chemotherapy', 'hormone_therapy', 'radio_therapy']
    therapy_names = ['Chemotherapy', 'Hormone therapy', 'Radiotherapy']
    therapy_rates = df[therapy_cols].mean() * 100
    df['treatment_type'] = df[therapy_cols].apply(
        lambda row: ' + '.join(
            name for col, name in zip(therapy_cols, therapy_names)
            if row[col] == 1
        ) or 'No therapy',
        axis=1
    )
    df['survival_rate_by_treatment'] = (
        df.groupby('treatment_type')['overall_survival'].transform('mean') * 100
    )
    treatment_summary = (
        df.groupby('treatment_type')
        .agg(
            survival_rate=('survival_rate_by_treatment', 'first'),
            patients=('overall_survival', 'size')
        )
        .sort_values('survival_rate')
    )
    death_counts = df['death_from_cancer'].value_counts(dropna=True)
    ### visualization
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
    ax=ax.flatten() # collapses from 2d array into 1d array, makes indexing subplots simpler
    ax[0].bar(x=therapy_names,height=therapy_rates.values,color='black')
    ax[0].set_ylabel('Patients treated (%)')
    ax[0].set_title('Therapy use',loc='left')
    ax[0].tick_params(axis='x',rotation=20)
    treatment_labels = [
        treatment.replace('Chemotherapy', 'Chemo')
        .replace('Hormone therapy', 'Hormone')
        .replace('Radiotherapy', 'Radio') + f" (n={patients})"
        for treatment, patients in zip(
            treatment_summary.index, treatment_summary['patients']
        )
    ]
    bars = ax[1].barh(
        treatment_labels,
        treatment_summary['survival_rate'].values,
        color='teal'
    )
    ax[1].bar_label(bars, fmt='%.1f%%', padding=3, fontsize=8)
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel('Living patients (%)')
    ax[1].set_title('Survival rate by treatment type',loc='left')
    ax[1].tick_params(axis='y',labelsize=8)
    ax[2].bar(x=death_counts.index,height=death_counts.values,color='black')
    ax[2].set_ylabel('Count')
    ax[2].set_title('Death outcomes',loc='left')
    ax[2].tick_params(axis='x',rotation=20)
    for i in range(len(ax)):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(args.output)

if __name__ == "__main__":
    main()