########################################################## Baselines & Motivation

######## Baselines
python base_main.py --savename='baselines' --dataset=imagenetv2 --mode=clip --model_size=ViT-L/14
python base_main.py --savename='baselines' --dataset=imagenet --mode=clip --model_size=ViT-L/14
python base_main.py --savename='baselines' --dataset=cub --mode=clip --model_size=ViT-L/14
python base_main.py --savename='baselines' --dataset=eurosat --mode=clip --model_size=ViT-L/14
python base_main.py --savename='baselines' --dataset=places365 --mode=clip --model_size=ViT-L/14
python base_main.py --savename='baselines' --dataset=food101 --mode=clip --model_size=ViT-L/14
python base_main.py --savename='baselines' --dataset=pets --mode=clip --model_size=ViT-L/14
python base_main.py --savename='baselines' --dataset=dtd --mode=clip --model_size=ViT-L/14

python base_main.py --savename='baselines' --dataset=imagenetv2 --mode=clip --model_size=ViT-B/32
python base_main.py --savename='baselines' --dataset=imagenet --mode=clip --model_size=ViT-B/32
python base_main.py --savename='baselines' --dataset=cub --mode=clip --model_size=ViT-B/32
python base_main.py --savename='baselines' --dataset=eurosat --mode=clip --model_size=ViT-B/32
python base_main.py --savename='baselines' --dataset=places365 --mode=clip --model_size=ViT-B/32
python base_main.py --savename='baselines' --dataset=food101 --mode=clip --model_size=ViT-B/32
python base_main.py --savename='baselines' --dataset=pets --mode=clip --model_size=ViT-B/32
python base_main.py --savename='baselines' --dataset=dtd --mode=clip --model_size=ViT-B/32

python base_main.py --savename='baselines' --dataset=imagenetv2 --mode=clip --model_size=RN50
python base_main.py --savename='baselines' --dataset=imagenet --mode=clip --model_size=RN50
python base_main.py --savename='baselines' --dataset=cub --mode=clip --model_size=RN50
python base_main.py --savename='baselines' --dataset=eurosat --mode=clip --model_size=RN50
python base_main.py --savename='baselines' --dataset=places365 --mode=clip --model_size=RN50
python base_main.py --savename='baselines' --dataset=food101 --mode=clip --model_size=RN50
python base_main.py --savename='baselines' --dataset=pets --mode=clip --model_size=RN50
python base_main.py --savename='baselines' --dataset=dtd --mode=clip --model_size=RN50


######## Baselines + GPTX
python base_main.py --savename='baselines_concept' --dataset=cub --mode=clip --model_size=ViT-L/14 --label_before_text='A photo of a bird: a '
python base_main.py --savename='baselines_concept' --dataset=eurosat --mode=clip --model_size=ViT-L/14 --label_before_text='A photo of a land use: a '
python base_main.py --savename='baselines_concept' --dataset=places365 --mode=clip --model_size=ViT-L/14 --label_before_text='A photo of a place: a '
python base_main.py --savename='baselines_concept' --dataset=food101 --mode=clip --model_size=ViT-L/14 --label_before_text='A photo of a food: a '
python base_main.py --savename='baselines_concept' --dataset=pets --mode=clip --model_size=ViT-L/14 --label_before_text='A photo of a breed: a '

python base_main.py --savename='baselines_concept' --dataset=cub --mode=clip --model_size=ViT-B/32 --label_before_text='A photo of a bird: a '
python base_main.py --savename='baselines_concept' --dataset=eurosat --mode=clip --model_size=ViT-B/32 --label_before_text='A photo of a land use: a '
python base_main.py --savename='baselines_concept' --dataset=places365 --mode=clip --model_size=ViT-B/32 --label_before_text='A photo of a place: a '
python base_main.py --savename='baselines_concept' --dataset=food101 --mode=clip --model_size=ViT-B/32 --label_before_text='A photo of a food: a '
python base_main.py --savename='baselines_concept' --dataset=pets --mode=clip --model_size=ViT-B/32 --label_before_text='A photo of a breed: a '

python base_main.py --savename='baselines_concept' --dataset=cub --mode=clip --model_size=RN50 --label_before_text='A photo of a bird: a '
python base_main.py --savename='baselines_concept' --dataset=eurosat --mode=clip --model_size=RN50 --label_before_text='A photo of a land use: a '
python base_main.py --savename='baselines_concept' --dataset=places365 --mode=clip --model_size=RN50 --label_before_text='A photo of a place: a '
python base_main.py --savename='baselines_concept' --dataset=food101 --mode=clip --model_size=RN50 --label_before_text='A photo of a food: a '
python base_main.py --savename='baselines_concept' --dataset=pets --mode=clip --model_size=RN50 --label_before_text='A photo of a breed: a '



######## GPT-Descriptions
python base_main.py --savename='baselines_gpt' --dataset=imagenetv2 --mode=gpt_descriptions --model_size=ViT-L/14
python base_main.py --savename='baselines_gpt' --dataset=imagenet --mode=gpt_descriptions --model_size=ViT-L/14
python base_main.py --savename='baselines_gpt' --dataset=cub --mode=gpt_descriptions --model_size=ViT-L/14
python base_main.py --savename='baselines_gpt' --dataset=eurosat --mode=gpt_descriptions --model_size=ViT-L/14
python base_main.py --savename='baselines_gpt' --dataset=places365 --mode=gpt_descriptions --model_size=ViT-L/14
python base_main.py --savename='baselines_gpt' --dataset=food101 --mode=gpt_descriptions --model_size=ViT-L/14
python base_main.py --savename='baselines_gpt' --dataset=pets --mode=gpt_descriptions --model_size=ViT-L/14
python base_main.py --savename='baselines_gpt' --dataset=dtd --mode=gpt_descriptions --model_size=ViT-L/14

python base_main.py --savename='baselines_gpt' --dataset=imagenetv2 --mode=gpt_descriptions --model_size=ViT-B/32
python base_main.py --savename='baselines_gpt' --dataset=imagenet --mode=gpt_descriptions --model_size=ViT-B/32
python base_main.py --savename='baselines_gpt' --dataset=cub --mode=gpt_descriptions --model_size=ViT-B/32
python base_main.py --savename='baselines_gpt' --dataset=eurosat --mode=gpt_descriptions --model_size=ViT-B/32
python base_main.py --savename='baselines_gpt' --dataset=places365 --mode=gpt_descriptions --model_size=ViT-B/32
python base_main.py --savename='baselines_gpt' --dataset=food101 --mode=gpt_descriptions --model_size=ViT-B/32
python base_main.py --savename='baselines_gpt' --dataset=pets --mode=gpt_descriptions --model_size=ViT-B/32
python base_main.py --savename='baselines_gpt' --dataset=dtd --mode=gpt_descriptions --model_size=ViT-B/32

python base_main.py --savename='baselines_gpt' --dataset=imagenetv2 --mode=gpt_descriptions --model_size=RN50
python base_main.py --savename='baselines_gpt' --dataset=imagenet --mode=gpt_descriptions --model_size=RN50
python base_main.py --savename='baselines_gpt' --dataset=cub --mode=gpt_descriptions --model_size=RN50
python base_main.py --savename='baselines_gpt' --dataset=eurosat --mode=gpt_descriptions --model_size=RN50
python base_main.py --savename='baselines_gpt' --dataset=places365 --mode=gpt_descriptions --model_size=RN50
python base_main.py --savename='baselines_gpt' --dataset=food101 --mode=gpt_descriptions --model_size=RN50
python base_main.py --savename='baselines_gpt' --dataset=pets --mode=gpt_descriptions --model_size=RN50
python base_main.py --savename='baselines_gpt' --dataset=dtd --mode=gpt_descriptions --model_size=RN50


######## Randomized Descriptions
python base_main.py --savename='randomized_descriptions' --dataset=imagenetv2 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions' --dataset=imagenet --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions' --dataset=cub --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions' --dataset=eurosat --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions' --dataset=places365 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions' --dataset=food101 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions' --dataset=pets --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions' --dataset=dtd --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14

python base_main.py --savename='randomized_descriptions' --dataset=imagenetv2 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions' --dataset=imagenet --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions' --dataset=cub --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions' --dataset=eurosat --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions' --dataset=places365 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions' --dataset=food101 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions' --dataset=pets --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions' --dataset=dtd --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32

python base_main.py --savename='randomized_descriptions' --dataset=imagenetv2 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions' --dataset=imagenet --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions' --dataset=cub --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions' --dataset=eurosat --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions' --dataset=places365 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions' --dataset=food101 --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions' --dataset=pets --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions' --dataset=dtd --mode=random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50


######## Randomized Descriptions
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=imagenetv2 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=imagenet --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=cub --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=eurosat --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=places365 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=food101 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=pets --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=dtd --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-L/14

python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=imagenetv2 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=imagenet --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=cub --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=eurosat --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=places365 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=food101 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=pets --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=dtd --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=ViT-B/32

python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=imagenetv2 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=imagenet --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=cub --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=eurosat --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=places365 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=food101 --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=pets --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=RN50
python base_main.py --savename='randomized_descriptions_5xbudget' --dataset=dtd --mode=random_descriptions --randomization_budget=5 --reps=7 --model_size=RN50



######## Shared Randomized Descriptions
python base_main.py --savename='shared_randomized_descriptions' --dataset=imagenetv2 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions' --dataset=imagenet --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions' --dataset=cub --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions' --dataset=eurosat --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions' --dataset=places365 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions' --dataset=food101 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions' --dataset=pets --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions' --dataset=dtd --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-L/14

python base_main.py --savename='shared_randomized_descriptions' --dataset=imagenetv2 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions' --dataset=imagenet --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions' --dataset=cub --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions' --dataset=eurosat --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions' --dataset=places365 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions' --dataset=food101 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions' --dataset=pets --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions' --dataset=dtd --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=ViT-B/32

python base_main.py --savename='shared_randomized_descriptions' --dataset=imagenetv2 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions' --dataset=imagenet --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions' --dataset=cub --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions' --dataset=eurosat --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions' --dataset=places365 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions' --dataset=food101 --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions' --dataset=pets --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions' --dataset=dtd --mode=shared_random_descriptions --randomization_budget=1 --reps=7 --model_size=RN50



######## Shared Randomized Descriptions
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=imagenetv2 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=imagenet --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=cub --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=eurosat --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=places365 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=food101 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=pets --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=dtd --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-L/14

python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=imagenetv2 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=imagenet --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=cub --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=eurosat --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=places365 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=food101 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=pets --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=dtd --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=ViT-B/32

python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=imagenetv2 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=imagenet --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=cub --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=eurosat --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=places365 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=food101 --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=pets --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=RN50
python base_main.py --savename='shared_randomized_descriptions_2xbudget' --dataset=dtd --mode=shared_random_descriptions --randomization_budget=2 --reps=7 --model_size=RN50




######## Swapped Descriptions
python base_main.py --savename='swapped_descriptions' --dataset=imagenetv2 --mode=swapped_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='swapped_descriptions' --dataset=imagenet --mode=swapped_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='swapped_descriptions' --dataset=cub --mode=swapped_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='swapped_descriptions' --dataset=eurosat --mode=swapped_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='swapped_descriptions' --dataset=places365 --mode=swapped_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='swapped_descriptions' --dataset=food101 --mode=swapped_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='swapped_descriptions' --dataset=pets --mode=swapped_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='swapped_descriptions' --dataset=dtd --mode=swapped_descriptions --reps=7 --model_size=ViT-L/14

python base_main.py --savename='swapped_descriptions' --dataset=imagenetv2 --mode=swapped_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='swapped_descriptions' --dataset=imagenet --mode=swapped_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='swapped_descriptions' --dataset=cub --mode=swapped_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='swapped_descriptions' --dataset=eurosat --mode=swapped_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='swapped_descriptions' --dataset=places365 --mode=swapped_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='swapped_descriptions' --dataset=food101 --mode=swapped_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='swapped_descriptions' --dataset=pets --mode=swapped_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='swapped_descriptions' --dataset=dtd --mode=swapped_descriptions --reps=7 --model_size=ViT-B/32

python base_main.py --savename='swapped_descriptions' --dataset=imagenetv2 --mode=swapped_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='swapped_descriptions' --dataset=imagenet --mode=swapped_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='swapped_descriptions' --dataset=cub --mode=swapped_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='swapped_descriptions' --dataset=eurosat --mode=swapped_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='swapped_descriptions' --dataset=places365 --mode=swapped_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='swapped_descriptions' --dataset=food101 --mode=swapped_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='swapped_descriptions' --dataset=pets --mode=swapped_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='swapped_descriptions' --dataset=dtd --mode=swapped_descriptions --reps=7 --model_size=RN50



######## Scrambled Descriptions
python base_main.py --savename='scrambled_descriptions' --dataset=imagenetv2 --mode=scrambled_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='scrambled_descriptions' --dataset=imagenet --mode=scrambled_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='scrambled_descriptions' --dataset=cub --mode=scrambled_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='scrambled_descriptions' --dataset=eurosat --mode=scrambled_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='scrambled_descriptions' --dataset=places365 --mode=scrambled_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='scrambled_descriptions' --dataset=food101 --mode=scrambled_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='scrambled_descriptions' --dataset=pets --mode=scrambled_descriptions --reps=7 --model_size=ViT-L/14
python base_main.py --savename='scrambled_descriptions' --dataset=dtd --mode=scrambled_descriptions --reps=7 --model_size=ViT-L/14

python base_main.py --savename='scrambled_descriptions' --dataset=imagenetv2 --mode=scrambled_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='scrambled_descriptions' --dataset=imagenet --mode=scrambled_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='scrambled_descriptions' --dataset=cub --mode=scrambled_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='scrambled_descriptions' --dataset=eurosat --mode=scrambled_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='scrambled_descriptions' --dataset=places365 --mode=scrambled_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='scrambled_descriptions' --dataset=food101 --mode=scrambled_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='scrambled_descriptions' --dataset=pets --mode=scrambled_descriptions --reps=7 --model_size=ViT-B/32
python base_main.py --savename='scrambled_descriptions' --dataset=dtd --mode=scrambled_descriptions --reps=7 --model_size=ViT-B/32

python base_main.py --savename='scrambled_descriptions' --dataset=imagenetv2 --mode=scrambled_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='scrambled_descriptions' --dataset=imagenet --mode=scrambled_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='scrambled_descriptions' --dataset=cub --mode=scrambled_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='scrambled_descriptions' --dataset=eurosat --mode=scrambled_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='scrambled_descriptions' --dataset=places365 --mode=scrambled_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='scrambled_descriptions' --dataset=food101 --mode=scrambled_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='scrambled_descriptions' --dataset=pets --mode=scrambled_descriptions --reps=7 --model_size=RN50
python base_main.py --savename='scrambled_descriptions' --dataset=dtd --mode=scrambled_descriptions --reps=7 --model_size=RN50













########################################################## Method

######## WaffleCLIP
python base_main.py --savename='waffleclip' --dataset=imagenetv2 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip' --dataset=imagenet --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip' --dataset=cub --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip' --dataset=eurosat --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip' --dataset=places365 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip' --dataset=food101 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip' --dataset=pets --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip' --dataset=dtd --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14

python base_main.py --savename='waffleclip' --dataset=imagenetv2 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip' --dataset=imagenet --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip' --dataset=cub --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip' --dataset=eurosat --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip' --dataset=places365 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip' --dataset=food101 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip' --dataset=pets --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip' --dataset=dtd --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32

python base_main.py --savename='waffleclip' --dataset=imagenetv2 --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip' --dataset=imagenet --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip' --dataset=cub --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip' --dataset=eurosat --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip' --dataset=places365 --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip' --dataset=food101 --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip' --dataset=pets --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip' --dataset=dtd --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50


######### Shared Shuffled + DCLIP
python base_main.py --savename='waffleclip_gpt' --dataset=imagenetv2 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip_gpt' --dataset=imagenet --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip_gpt' --dataset=cub --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip_gpt' --dataset=eurosat --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip_gpt' --dataset=places365 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip_gpt' --dataset=food101 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip_gpt' --dataset=pets --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14
python base_main.py --savename='waffleclip_gpt' --dataset=dtd --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14

python base_main.py --savename='waffleclip_gpt' --dataset=imagenetv2 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip_gpt' --dataset=imagenet --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip_gpt' --dataset=cub --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip_gpt' --dataset=eurosat --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip_gpt' --dataset=places365 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip_gpt' --dataset=food101 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip_gpt' --dataset=pets --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32
python base_main.py --savename='waffleclip_gpt' --dataset=dtd --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32

python base_main.py --savename='waffleclip_gpt' --dataset=imagenetv2 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip_gpt' --dataset=imagenet --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip_gpt' --dataset=cub --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip_gpt' --dataset=eurosat --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip_gpt' --dataset=places365 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip_gpt' --dataset=food101 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip_gpt' --dataset=pets --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50
python base_main.py --savename='waffleclip_gpt' --dataset=dtd --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50


########## WaffleCLIP + Concepts
python base_main.py --savename='waffleclip_concepts' --dataset=cub --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a bird: a '
python base_main.py --savename='waffleclip_concepts' --dataset=eurosat --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a land use: a '
python base_main.py --savename='waffleclip_concepts' --dataset=places365 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a place: a '
python base_main.py --savename='waffleclip_concepts' --dataset=food101 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a food: a '
python base_main.py --savename='waffleclip_concepts' --dataset=pets --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a breed: a '

python base_main.py --savename='waffleclip_concepts' --dataset=cub --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a bird: a '
python base_main.py --savename='waffleclip_concepts' --dataset=eurosat --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a land use: a '
python base_main.py --savename='waffleclip_concepts' --dataset=places365 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a place: a '
python base_main.py --savename='waffleclip_concepts' --dataset=food101 --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a food: a '
python base_main.py --savename='waffleclip_concepts' --dataset=pets --mode=waffle --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a breed: a '

python base_main.py --savename='waffleclip_concepts' --dataset=cub --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a bird: a '
python base_main.py --savename='waffleclip_concepts' --dataset=eurosat --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a land use: a '
python base_main.py --savename='waffleclip_concepts' --dataset=places365 --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a place: a '
python base_main.py --savename='waffleclip_concepts' --dataset=food101 --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a food: a '
python base_main.py --savename='waffleclip_concepts' --dataset=pets --mode=waffle --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a breed: a '


########### WaffleCLIP + GPT + Concepts
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=cub --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a bird: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=eurosat --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a land use: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=places365 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a place: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=food101 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a food: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=pets --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-L/14 --label_before_text='A photo of a breed: a '

python base_main.py --savename='waffleclip_gpt_concepts' --dataset=cub --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a bird: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=eurosat --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a land use: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=places365 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a place: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=food101 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a food: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=pets --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=ViT-B/32 --label_before_text='A photo of a breed: a '

python base_main.py --savename='waffleclip_gpt_concepts' --dataset=cub --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a bird: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=eurosat --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a land use: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=places365 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a place: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=food101 --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a food: a '
python base_main.py --savename='waffleclip_gpt_concepts' --dataset=pets --mode=waffle_and_gpt --waffle_count=15 --reps=7 --model_size=RN50 --label_before_text='A photo of a breed: a '