from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

files = [f"./data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "validation"]]
print(files)
tokenizer.train(files, trainer)
tokenizer.save("data/tokenizer-wiki.json")

tokenizer = Tokenizer.from_file("data/tokenizer-wiki.json")

sentence = "Hello, y'all! How are you 😁 ?"
tokenizer.encode(sentence)

unknown_token_index = output.tokens.index('[UNK]')
unknown_token_offsets = output.offsets[unknown_token_index]
sentence[unknown_token_offsets[0]: unknown_token_offsets[1]]