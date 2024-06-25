from transformers import MBartForConditionalGeneration
from transformers import T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

model.save_pretrained("bert-base-uncased")
tok.save_pretrained("bert-base-uncased")


# model = T5ForConditionalGeneration.from_pretrained("t5-base")
# tok = AutoTokenizer.from_pretrained("t5-base")

# model.save_pretrained("t5-base")
# tok.save_pretrained("t5-base")


print("Done. ")