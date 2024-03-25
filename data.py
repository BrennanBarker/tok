"""Get and reformat the data"""

import datasets

print('Loading dataset')
ds = datasets.load_dataset('wikitext', name="wikitext-103-raw-v1")

for split in ["test", "train", "validation"]:
    print(f'Working on split {split}')
    with open(f'data/wikitext-103-raw/wiki.{split}.raw', 'w') as f:
        f.write(''.join(ds[split]['text']))
