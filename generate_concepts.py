#%%
import numpy as np
import openai
import json

openai.api_key = 'your_key'

#%% Create GPT-3 prompts with classname lists.
np.random.seed(0)
import numpy as np
datasets = ['imagenet', 'cub', 'eurosat', 'places365', 'food101', 'pets', 'dtd', 'fgvcaircraft', 'cars', 'flowers102', ]
single_prompts = []
long_prompts = []
for dataset in datasets:
    out = json.load(open(f'descriptors/descriptors_{dataset}.json'))
    lab_list = np.random.choice(list(out.keys()), np.min([25, len(out)]), replace=False)
    lab_list = ', '.join([x.replace('_', ' ') for x in lab_list])
    if dataset == 'places365':
        lab_list = ', '.join([x.replace('-', ' ') for x in lab_list])
    long_prompt = "Q: Tell me in five words or less what " + lab_list + " have in common. It may be nothing. A: They are all "
    long_prompts.append(long_prompt)


#%% Query GPT-3
out = openai.Completion.create(
    model="text-davinci-003", prompt=long_prompts,
    temperature=0., max_tokens=300
)

#%% Generated Concepts:
classes = []
for elem in out['choices']:
    concept = elem['text'].replace('\n', '').replace('.', '').replace('?', '').lower()
    if concept[-1] == 's':
        concept = concept[:-1]
    classes.append(concept)

print('Predicted Classes for each dataset: ')
for dataset, classname in zip(datasets, classes):
    print(f'Dataset: {dataset} - {classname}')