from datasets import Dataset

gen= Dataset.load_from_disk('generated_dataset_highq_examples')
for g in gen:
    print(g['input_col'])
    print(g['output_col'])
    print("======================")