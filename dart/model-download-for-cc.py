from transformers import MBartForConditionalGeneration
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
tok = AutoTokenizer.from_pretrained("facebook/bart-large")

model.save_pretrained("facebook/bart-large")
tok.save_pretrained("facebook/bart-large")


# model = T5ForConditionalGeneration.from_pretrained("t5-base")
# tok = AutoTokenizer.from_pretrained("t5-base")

# model.save_pretrained("t5-base")
# tok.save_pretrained("t5-base")


print("Done. ")