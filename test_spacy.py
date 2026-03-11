import spacy_transformers
import spacy
from pyabsa import AspectTermExtraction as ATEPC
from transformers import pipeline
import os
from spacy import vocab
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

aspect_extractor = ATEPC.AspectExtractor('english', auto_device=True)

spacy.require_gpu()

nlp = spacy.load("en_core_web_trf")

text = "Both the rate and the depth of her compressions are not appropriate. The rate needs to be increased to 100-120 beats per minute and compression depth should be at least 2 inches or 5 cm."

doc = nlp(text)

chunks = list(doc.noun_chunks)
print(chunks)

relevant_phrases = []
for chunk in doc.noun_chunks:
    if chunk.root.dep_ == "nsubj":
        print(chunk.text)
        relevant_phrases.append(chunk)

result_nov = aspect_extractor.predict(
    [text], 
)

absa_pipeline = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")

# Inputs: The sentence + the target word
for aspect in relevant_phrases:
    result = absa_pipeline(text, text_pair=str(aspect))
    print(aspect)
    print(result)

print(result)


