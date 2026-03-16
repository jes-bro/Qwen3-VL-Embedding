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
    # result = aspect_extractor.predict(
    # [text]
    # )
    # print(result)
    doc = nlp(text)
    chunks = list(doc)
    print(chunks)
    seen_lemmas = []
    # seen_chunks = []
    seen_roots = []
    relevant_phrases = []
    STATE_VERBS = {"want", "like", "know", "believe", "think", "have", "be", "get", "seem", "need", "try", "mean"}
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_ not in STATE_VERBS:
            if token.dep_ not in ("aux", "auxpass"):
                print(f' relevant token added! : {token.lemma_}')
                relevant_phrases.append((token, token.lemma_))
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
        

    # relevant_phrases_lowered = [(token, word[1].lower()) for word in relevant_phrases]

    good_list = []
    bad_list = []
    good_list_lower = []
    bad_list_lower = []
    for (token, lemma) in relevant_phrases:
        result = absa_pipeline(text, text_pair=token.text)
        print(result)
        subj = [child.text for child in token.children if child.dep_ == "nsubj"]
        obj = [child.text for child in token.children if child.dep_ == "dobj"]
        # advmod = [child.text for child in token.children if child.dep_ == "advmod"]
        if result[0]['label'] == 'Negative' and result[0]['score'] > 0.85:
            if subj and obj:
                bad_list.append("" + token.lemma_ + " " + obj[0])
            elif obj: 
                bad_list.append("" + token.lemma_ + " " + obj[0])
            elif subj:
                bad_list.append(token.lemma_) # should i move these down? 
            else:
                bad_list.append(token.lemma_)
            print(lemma)
            print(result)
        elif result[0]['label'] == 'Positive'and result[0]['score'] > 0.85: # check and see if this threshold is actually meaningful
            if subj and obj:
                good_list.append("" + token.lemma_ + " " + obj[0])
            elif obj: 
                good_list.append("" + token.lemma_ + " " + obj[0])
            elif subj:
                good_list.append(token.lemma_) # should i move these down? 
            else:
                good_list.append(token.lemma_)
            # good_list.append(aspect)
            print(lemma)
            print(result)
    good_list_lower = list(set([phrase.lower() for phrase in good_list]))
    bad_list_lower = list(set([phrase.lower() for phrase in bad_list]))

    return good_list_lower, bad_list_lower

input_json_path = "/home/jess/sm/temp_compression_sub_result.json" # "/home/jess/sm/temp_cpr_sub_result.json"

with open(input_json_path, "r") as f:
    subtasks = json.load(f)

smol = "llama3.1:8b"

for subtask in subtasks:
    print(subtask)
    for video_name in subtasks[subtask]:
        if "time_stamps_file_paths_poses" in subtasks[subtask][video_name].keys():
            timestamp_dicts = subtasks[subtask][video_name]["time_stamps_file_paths_poses"]
            print(timestamp_dicts.keys())
            for timestamp in timestamp_dicts:
                if "commentary" in subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp].keys():
                    commentary_at_timestamp = subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["commentary"]
                    if commentary_at_timestamp:
                        overall_good_list = []
                        overall_bad_list = []
                        for individual_comment in commentary_at_timestamp:
                            overall_sentiment = sentiment_task(individual_comment)
                            print(f'overall sentiment: {overall_sentiment}')
                            if overall_sentiment[0]['label'] == 'neutral' and overall_sentiment[0]['score'] > 0.8:
                                pass
                            else:

                                response_constructive: ChatResponse = chat(model=smol, messages=[
                                {
                                    'role': 'user',
                                    'content': f'Analyze the following text for instances of negative feedback on skills the subject needs to improve upon. If no negative feedback exists, state \'None found\' and do not attempt to identify any. If negative feedback on skill attributes that need to be performed better exists, create a comma separated list of the lemmas of skill actions (1-3 words) that the subject of the commentary needs to improve upon based on the following commentary from an expert. Do not include anything else. No markdown. No additional text. No adjectives. No information about CPR broadly. Do not include the word CPR. Do not include duplicates. Here is the commentary: {individual_comment}',
                                },
                                ])

                                response_positive: ChatResponse = chat(model=smol, messages=[
                                {
                                    'role': 'user',
                                    'content': f'Analyze the following text for instances of positive feedback. If no positive feedback exists, state \'None found\' and do not attempt to identify any. If positive feedback exists, create a comma separated list of the lemmas of skill actions (1-3 words) that the subject performed well based on the commentary. Do not include anything else. No markdown. No additional text. No adjectives. No information about CPR broadly. Do not include the word CPR. Do not include duplicates. Here is the commentary: {individual_comment}',
                                },
                                ])

                                print(f"Individual comment: {individual_comment}") # Maybe just filter out negative sentiment ones?
                                # print(f'Comment summary: {individual_comment_summary}')
                                # good_list_individual, bead_list_individual = get_good_and_bad_lists(individual_comment) 
                                print(f'constructive feedback skill attributes: {response_constructive.message.content}')
                                print(f'postive feedback skill attributes: {response_positive.message.content}') # can always lemma later if you need to and extract the lemmas from the lists
                                needs_improvement_list = str(response_constructive.message.content).split(", ")
                                good_executions_list = str(response_positive.message.content).split(", ")
                                # good_list_individual, _ = get_good_and_bad_lists(positive_fb_summary.message.content) 
                                # if good_list_individual:
                                overall_good_list.extend(good_executions_list)
                                print(f"Added {good_executions_list} to overall good list!")
                                # if bad_list_individual:
                                overall_bad_list.extend(needs_improvement_list)
                                print(f"Added {needs_improvement_list} to overall bad list!")
                    subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["good_executions_list"] = overall_good_list
                    subtasks[subtask][video_name]["time_stamps_file_paths_poses"][timestamp]["needs_improvement_list"] = overall_bad_list
                    print(f'timestamp: {timestamp}')
                    print(f'video name: {video_name}')
                    print(f'GOOD LIST: {overall_good_list}')
                    print(f'NEEDS IMPROVEMENT LIST: {overall_bad_list}')


output_path = "/home/jess/Qwen3-VL-Embedding/test_spacy.json"
with open (output_path, 'w') as f:
    json.dump(subtasks, f)
