from transformers import MBartForConditionalGeneration
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer

# model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
# tok = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

# model.save_pretrained("facebook/mbart-large-cc25")
# tok.save_pretrained("facebook/mbart-large-cc25")


model = T5ForConditionalGeneration.from_pretrained("t5-base")
tok = AutoTokenizer.from_pretrained("t5-base")

model.save_pretrained("t5-base")
tok.save_pretrained("t5-base")


print("Done. ")