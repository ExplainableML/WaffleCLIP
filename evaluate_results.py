#%% Select relevant files.
import numpy as np
import pandas as pd

files = [
    'results/baselines.csv',
    'results/baselines_concept.csv',
    'results/baselines_gpt.csv',
    'results/shared_randomized_descriptions.csv',
    'results/shared_randomized_descriptions_2xbudget.csv',
    'results/swapped_descriptions.csv',
    'results/scrambled_descriptions.csv',
    'results/randomized_descriptions.csv',
    'results/randomized_descriptions_5xbudget.csv',
    'results/waffleclip.csv',
    'results/waffleclip_concepts.csv',
    'results/waffleclip_gpt.csv',
    'results/waffleclip_gpt_concepts.csv',
]

#%% Extract classification accuracies from files.
table_data = []
for file in files:
    out = pd.read_csv(file, header=None)
    out = np.array(out)

    # Per Dataset Context
    # datasets = np.array(['flowers102', 'fgvcaircraft', 'cars'])
    datasets = np.array(['imagenetv2', 'imagenet', 'cub', 'eurosat', 'places365', 'food101', 'pets', 'dtd'])
    fixed_cols = []
    resp_dataset = []
    resp_dataset_row_idcs = []
    resp_dataset_col_idcs = []
    for row in out:
        coln = row[0]
        for dataset in datasets:
            coln = coln.replace(f'dataset={dataset}; ', '')
        if 'label_before_text' in coln:
            coln = coln.split('label_before_text')[0] + ''.join(''.join(coln.split('label_before_text')[1:]).split(';')[1:])        
        fixed_cols.append(coln)
    unique_cols, idcs = np.unique(fixed_cols, return_index=True)
    unique_cols = np.array(unique_cols)[np.argsort(idcs)]

    for i, row in enumerate(out):
        dataset = row[0].split('dataset=')[-1].split(';')[0]
        resp_dataset_col_idcs.append(np.where(datasets == dataset)[0][0])
        resp_dataset_row_idcs.append(np.where(unique_cols == fixed_cols[i])[0][0])
        resp_dataset.append(dataset)


    # Per Dataset Context
    mean_results_top1 = np.ones((len(unique_cols), len(datasets))) * np.nan
    std_results_top1 = np.ones((len(unique_cols), len(datasets))) * np.nan
    for k in range(len(out)):
        i = resp_dataset_row_idcs[k]
        j = resp_dataset_col_idcs[k]
        mean_results_top1[i, j] = np.round(out[k][1], 2)
        std_results_top1[i, j] = np.round(out[k][2], 2)

    mean_avgs_top1 = np.nanmean(mean_results_top1, axis=-1)
    norm = np.sum(~np.isnan(mean_results_top1), axis=-1).reshape(-1, 1)
    std_avgs_top1 = np.sqrt(np.sum(std_results_top1 ** 2 / norm, axis=-1))

    result_str = np.ones((len(unique_cols)+1, len(datasets) + 1)).astype(str)
    result_str[0, :] = [str(x) for x in datasets] + ['Avg']
    for i in range(mean_results_top1.shape[0]):
        for j in range(mean_results_top1.shape[1]):
            result_str[i+1, j] = '{0:2.2f} ({1:2.2f})'.format(mean_results_top1[i, j], std_results_top1[i, j])
        result_str[i+1, -1] = '{0:2.2f} ({1:2.2f})'.format(mean_avgs_top1[i], std_avgs_top1[i])
        

    # Per Dataset Context
    table_data.append(file)
    for i in range(len(unique_cols)):
        subcoll = []
        for sub in result_str[i+1]:
            res = sub.split(' ')
            subcoll.append('{0} \\small$\\pm{1}$'.format(res[0], res[1].replace('(', '').replace(')', '')))
        table_data.append(' & '.join(subcoll) + '\\\\')
    table_data.append('--------')
    


#%% Print final table.
print('\n'.join(table_data))

