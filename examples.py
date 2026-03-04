import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from natsort import natsorted
from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient
# Define a list of query texts


# Define a list of document texts and images
# documents = [
#     {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
#     {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
#     {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
# ]


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
documents = []

for video_dir, frame_paths in videos.items():
    documents.append({
        "text": "Represent this video for retrieval.",
        "image": frame_paths   # list of frame paths
    })

queries = [
    {"text": "Represent this video for retrieval.","image": query_paths[0:4]}
]

print(f'paths 0-4 query frames: {query_paths[0:4]}')
    
# Combine queries and documents into a single input list
inputs = queries + documents

# Process the inputs to get embeddings
embeddings = model.process(inputs)
print(embeddings.shape)

client = QdrantClient(":memory:")

if not client.collection_exists("video_embs"):
   client.create_collection(
      collection_name="video_embs",
      vectors_config=VectorParams(size=2048, distance=Distance.COSINE),
   )

client.upsert(
   collection_name="video_embs",
   points=[
      PointStruct(
            id=idx,
            vector=vector,
            payload={"color": "red", "rand_number": idx % 10}
      )
      for idx, vector in enumerate(embeddings)
   ]
)

# Compute similarity scores between query embeddings and document embeddings
# Realized this is just dot product not cosine similarity
# So will need to change to cosine similarity so we don't care about magnitude
similarity_scores = (embeddings[:1] @ embeddings[1:].T)

# Print out the similarity scores in a list format
print(similarity_scores.tolist())

# [[0.8157786130905151, 0.7178360223770142, 0.7173429131507874], [0.5195091962814331, 0.3302568793296814, 0.4391537308692932], [0.3884059488773346, 0.285782128572464, 0.33141762018203735], [0.1092604324221611, 0.03871120512485504, 0.06952016055583954]]
