import sys
import numpy as np
import torch
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from natsort import natsorted
# from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient, models

try:
    from transformers.utils.generic import check_model_inputs  # old
except Exception:
    from transformers.utils import generic as _generic
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs

    def check_model_inputs(func=None):
        def _wrap(f):
            return capture_outputs(merge_with_config_defaults(f))
        return _wrap if func is None else _wrap(func)

    # make it importable as: from transformers.utils.generic import check_model_inputs
    _generic.check_model_inputs = check_model_inputs

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
# embed the good and bad lists and store as separate vectors
process = subprocess.Popen(['bash', '/home/jess/Qwen3-VL-Embedding/run_get_embeddings.sh'], stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True)


log = []

for line in process.stdout:
    print(line, end="")
    log.append(line)

process.wait()
goodbadlistfile = '/home/jess/Qwen3-VL-Embedding/goodbadoutputs.json'

# Define a list of query texts
with open(goodbadlistfile, 'r') as file:
    goods_and_bads = json.load(file)

# Define a list of document texts and images
# documents = [
#     {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
#     {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
#     {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
# ]
documents = []
queries = []
exp_video_names = []
query_vid_names = []

for video_name in goods_and_bads.keys():
    if '1' not in video_name:
        documents.append({
        "text": goods_and_bads[video_name]['good']
    })
        exp_video_names.append(video_name)
    else:
        queries.append({"text": goods_and_bads[video_name]['bad']})
        queries.append({"text": goods_and_bads[video_name]['good']})
        query_vid_names.append(video_name)

print(documents)
print(queries)

# Specify the model path
model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"

# Initialize the Qwen3VLEmbedder model
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path,
    max_pixels=112*112,
    total_pixels=4*112*112)
# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

paths = []
records = []
query_paths = []
top_level_path = Path("/home/jess/Downloads/cpr_vids").rglob('*.png')
for file_path in top_level_path:
    if "cam" in str(file_path) and "press" in str(file_path):
        paths.append(str(file_path))
        if "08_2" in str(file_path):
            query_paths.append(str(file_path))
    # print(file_path)
    # print(file_path)
# 32 frames per video rn
# need to get real vids going and query with a video and update on that progress yay if you get that that is good progress
# then you can mention the next steps and the metrics and the testing the last part and training m2iv on more stuff 
# and all of the intricacies of the rag need to be selected and done 
# but it's ok you will figure that out
# but tomorrow query with video with similarity goal for meeting time 
# figure out what to say to group with woody and stuff 
# you should tell the people in the gc about woody before the meeting and email chris back finally
paths = paths[0:128]
paths = natsorted(paths)
query_paths = natsorted(query_paths)
videos = defaultdict(list)
# rewrite later
for path in paths:
    posix_path = Path(path)
    videos[posix_path.parent].append(path)
for video in videos.keys():
    videos[video] = videos[video][0:4]
print(videos.keys())


# we want expert good lists
# for video_dir, frame_paths in videos.items():
#     documents.append({
#         "text": good_list_string,
#         "image": frame_paths   # list of frame paths
#     })
# documents.append({
#         "text": good_list_intexp
#     })

# documents.append({
#         "text": good_list_exp
#     })
# and novice bad list and to find similarity between those and visual component, but mostly the poses not the raw video
# so will need to do poses

print(f'paths 0-4 query frames: {query_paths[0:4]}')
    
# Combine queries and documents into a single input list
inputs = queries + documents

# Process the inputs to get embeddings
embeddings = model.process(inputs)
# make embeddings into good/bad tuples
print(embeddings.shape)

client = QdrantClient(":memory:")

# if not client.collection_exists("video_embs"):
#    client.create_collection(
#       collection_name="video_embs",
#       vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
#    )

# client.upsert(
#    collection_name="video_embs",
#    points=[
#       PointStruct(
#             id=idx,
#             vector=vector,
#             # payload={"color": "red", "rand_number": idx % 10}
#       )
#       for idx, vector in enumerate(embeddings)
#    ]
# )

if not client.collection_exists("expert_data"):
   client.create_collection(
      collection_name="expert_data",
      vectors_config={"good_exp_vector":VectorParams(size=2048, distance=Distance.COSINE),
    #   "bad_exp_vector":VectorParams(size=2048, distance=Distance.COSINE), # might need bad exp later
      }
   )
# what's more important right now is two queries - nov good and nov bad so will need to update json file and processing there
# and then use pre-fetch to do the two comparisons and then weighted fusion

num_queries = 2
bad_novice_vector = embeddings[0].detach().cpu().float().numpy()
good_novice_vector = embeddings[1].detach().cpu().float().numpy()
expert_good_vectors = embeddings[num_queries:].detach().cpu().float().numpy()

client.upsert(
   collection_name="expert_data",
   points=[
      PointStruct(
            id=idx,
            vector={
                "good_exp_vector":vector,
            },
            payload={
                "clip_name": exp_video_names[idx],
                # "annotation": 1.99, clip_path
            },
            # payload={"color": "red", "rand_number": idx % 10}
      )
      for idx, vector in enumerate(expert_good_vectors)
   ]
)

# client.query_points(
#     collection_name="expert_data",
#     prefetch=models.Prefetch(
#         query=bad_novice_vector,  # <------------- small byte vector
#         using="good_exp_vector",
#         limit=3,
#     ),
#     query=good_novice_vector,  # <-- full vector
#     using="good_exp_vector",
#     limit=3,
# ) # difference? 

# this seems to be it though
result = client.query_points(
    collection_name="expert_data",
    prefetch=[
        models.Prefetch(
            query=bad_novice_vector,
            using="good_exp_vector",
            limit=1,
        ),
        models.Prefetch(
            query=good_novice_vector,  # <-- dense vector
            using="good_exp_vector",
            limit=1, # sweep the limit hyperparams too. not bad spot for now. cause it got the right one.. but it should with the ranking multiple anyway. Need to handle the not same number of them case and do the matching thing i think. i hope theres an efficient way to do that.
        ),
    ],
    query=models.RrfQuery(rrf=models.Rrf(weights=[2.0, 1.0])), # try 2 and sweep some hyperparams maybe 
)

print(result)
print(good_novice_vector @ bad_novice_vector)

# Need to fuse results from the different comparisons 
# so the nov bad exp good pairing and the nov good exp good pairing 
# then fuse with the pose similarity / other perspective similarity things 
# how to make all those comparisons happen? figurre that out after we do this first one
# Compute similarity scores between query embeddings and document embeddings
# Realized this is just dot product not cosine similarity
# So will need to change to cosine similarity so we don't care about magnitude
# There will only ever be one query for now
# what is precision of float() cast?
# num_queries = 2
# mag_query = np.linalg.norm(embeddings[:1].detach().cpu().float().numpy())
# mag_docs = np.linalg.norm(embeddings[1:].detach().cpu().float().numpy())
# similarity_scores = (embeddings[:1] @ embeddings[1:].T) / (mag_query * mag_docs)

# Raw similarity does not work
# Next will try top k similarity words (k = number of words in novice bad list or exp good list whicever is shorter)
# See if that works - like per word for top k words that are similar
# Cause it's ok if the expert is good at other things 
# And maybe do cosine similarity try that cause magnitude invariant
# and incorporate positive lists we want similar positive lists as well with extra 
# so as to not give people a poor example of what they are already doing well at so as to only provide a helpful example
# see how that goes
# then do poses 
# and try with real data

# Print out the similarity scores in a list format
# print(similarity_scores.tolist())

# [[0.8157786130905151, 0.7178360223770142, 0.7173429131507874], [0.5195091962814331, 0.3302568793296814, 0.4391537308692932], [0.3884059488773346, 0.285782128572464, 0.33141762018203735], [0.1092604324221611, 0.03871120512485504, 0.06952016055583954]]
