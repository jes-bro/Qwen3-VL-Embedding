import json
import spacy_transformers
import spacy
from pyabsa import AspectTermExtraction as ATEPC
from transformers import pipeline
from ollama import chat
from ollama import ChatResponse
import os
from spacy import vocab
from spacytextblob.spacytextblob import SpacyTextBlob

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

aspect_extractor = ATEPC.AspectExtractor('english', auto_device=True)
absa_pipeline = pipeline("text-classification", model="yangheng/deberta-v3-base-absa-v1.1")
spacy.require_gpu()

nlp = spacy.load("en_core_web_trf")
# nlp.add_pipe('spacytextblob')
sentiment_task = pipeline("sentiment-analysis", model='cardiffnlp/twitter-roberta-base-sentiment-latest', tokenizer='cardiffnlp/twitter-roberta-base-sentiment-latest')
# text = "Both the rate and the depth of her compressions are not appropriate. The rate needs to be increased to 100-120 beats per minute and compression depth should be at least 2 inches or 5 cm."
def get_good_and_bad_lists(text):
    overall_sentiment = sentiment_task(text)
    print(f'overall sentiment: {overall_sentiment}')
    if overall_sentiment[0]['label'] == 'neutral' and overall_sentiment[0]['score'] > 0.8:
        return [], []
    # if abs(doc._.blob.polarity) < 0.2:
    #     print(f'neutral text: {text}')
    #     return [], []
    result = aspect_extractor.predict(
    [text]
    )
    print(result)
    doc = nlp(text)
    chunks = list(doc)
    print(chunks)
    seen_lemmas = []
    # seen_chunks = []
    seen_roots = []
    relevant_phrases = []
    for token in doc:
        if token.pos_ == "VERB":
            if token.dep_ not in ("aux", "auxpass"):
                subj = [child.text for child in token.children if child.dep_ == "nsubj"]
                obj = [child.text for child in token.children if child.dep_ == "dobj"]
                if subj and obj:
                    relevant_phrases.append("" + subj[0] + " " + token.lemma_ + " " + obj[0])
                elif obj: 
                    relevant_phrases.append("" + token.lemma_ + " " + obj[0])
                elif subj:
                    relevant_phrases.append("" + subj[0] + " " + token.lemma_)
                # else:
                #     relevant_phrases.append(token.lemma_)
                # else:
                #     relevant_phrases.append(token.lemma_)

    # for chunk in doc.noun_chunks:
    #     chunk_lemma = " ".join(token.lemma_ for token in chunk)
    #     print(f"checking chunk: {chunk}. The lemma is: {chunk_lemma}")
    #     if chunk.root.dep_ == "nsubj" and chunk_lemma not in seen_lemmas and chunk.root.text not in seen_roots:
    #         print(f' chunk {chunk} made it through!')
    #         print(chunk.text)
    #         relevant_phrases.append(chunk.text)
    #         seen_lemmas.append(chunk_lemma)
    #         seen_roots.append(chunk.root.text)
    #     elif chunk_lemma in seen_lemmas:
    #         print(f"we've seen the lemma for chunk {chunk}!")
    #         chunk_lemma_idx = seen_lemmas.index(chunk_lemma)
    #         print(chunk.text)
    #         print(relevant_phrases[chunk_lemma_idx])
    #         if len(chunk.text) < len(relevant_phrases[chunk_lemma_idx]):
    #             relevant_phrases[chunk_lemma_idx] = chunk.text
    #     elif chunk.root.text in seen_roots:
    #         root_idx = seen_roots.index(chunk.root.text)
    #         print(relevant_phrases[root_idx])
    #         old_word = relevant_phrases[root_idx]
    #         if len(relevant_phrases[root_idx]) > len(chunk.text):
    #             relevant_phrases[root_idx] = chunk.text
    #             print(f'{chunk} shorter than {old_word}!')

    #     else:
    #         print(f'Probably less relevant phrase: {chunk}')
        

    relevant_phrases_lowered = [word.lower() for word in relevant_phrases]

    good_list = []
    bad_list = []
    for aspect in relevant_phrases_lowered:
        result = absa_pipeline(text, text_pair=str(aspect))
        if result[0]['label'] == 'Negative' and result[0]['score'] > 0.8:
            bad_list.append(aspect)
        elif result[0]['label'] == 'Positive'and result[0]['score'] > 0.55: # check and see if this threshold is actually meaningful
            good_list.append(aspect)
        print(aspect)
        print(result)
    return good_list, bad_list

input_json_path = "/home/jess/sm/temp_cpr_sub_result.json"

with open(input_json_path, "r") as f:
    subtasks = json.load(f)

for subtask in subtasks:
    print(subtask)
    for video_name in subtasks[subtask]:
        timestamp_dicts = subtasks[subtask][video_name]["time_stamps_file_paths_poses"]
        print(timestamp_dicts.keys())
        for timestamp in timestamp_dicts:
            if "commentary" in subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp].keys():
                commentary_at_timestamp = subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["commentary"]
                overall_good_list = []
                overall_bad_list = []
                for individual_comment in commentary_at_timestamp:
                    response_summary: ChatResponse = chat(model='gemma3', messages=[
                    {
                        'role': 'user',
                        'content': f'Rephrase the feedback that the person/camera-wearer got. Try to rephrase it like subject verbed noun in adjective way: {individual_comment}. Plain text only. No Markdown or formatting or new lines. Be objective. Include negative stuff too. If there\'s none of good or bad, just leave it out. If the text is neutral, leave that out as well.',
                        'format' : 'json'
                    },
                    ])
                    positive_fb_summary: ChatResponse = chat(model='gemma3', messages=[
                    {
                        'role': 'user',
                        'content': f'Rephrase the positive feedback that the person/camera-wearer got, if any. Try to rephrase it like subject verbed noun in adjective way: {individual_comment}. Plain text only. No Markdown or formatting or new lines. Be objective. Include negative stuff too. If there\'s none of good or bad, just leave it out. If the text is neutral, leave that out as well.',
                        'format' : 'json'
                    },
                    ])
                    individual_comment_summary = response_summary.message.content
                    # doc = nlp(individual_comment)
                    print(f"Individual comment: {individual_comment}") # Maybe just filter out negative sentiment ones?
                    print(f'Comment summary: {individual_comment_summary}')
                    _, bad_list_individual = get_good_and_bad_lists(individual_comment_summary) 
                    good_list_individual, _ = get_good_and_bad_lists(individual_comment) 
                    if good_list_individual is not None:
                        overall_good_list.extend(good_list_individual)
                        print(f"Added {good_list_individual} to overall good list!")
                    if bad_list_individual is not None:
                        overall_bad_list.extend(bad_list_individual)
                        print(f"Added {bad_list_individual} to overall bad list!")
                subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["good_list"] = overall_good_list
                subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["needs_improvement_list"] = overall_bad_list
                print(f'commentary: {individual_comment}')
                print(f'commentary summary: {individual_comment_summary}')
                print(f'good list: {overall_good_list}')
                print(f'needs improvement list: {overall_bad_list}')


output_path = "/home/jess/Qwen3-VL-Embedding/test_spacy.json"
with open (output_path, 'w') as f:
    json.dump(subtasks, f)
