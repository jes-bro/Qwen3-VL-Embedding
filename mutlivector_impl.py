import json
import subprocess
from qdrant_client import QdrantClient, models
from PIL import Image
from ultralytics import YOLO
import random
# 1. Connect to Qdrant server
client = QdrantClient(":memory:")

from fastembed import TextEmbedding, LateInteractionTextEmbedding
import os
# from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
# embed the good and bad lists and store as separate vectors
# process = subprocess.Popen(['bash', '/home/jess/Qwen3-VL-Embedding/run_get_embeddings.sh'], stdout=subprocess.PIPE,
#     stderr=subprocess.STDOUT,
#     text=True)


log = []

# for line in process.stdout:
#     print(line, end="")
#     log.append(line)

# process.wait()
goodbadlistfile = '/home/jess/Qwen3-VL-Embedding/test_spacy.json'

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
pose_model = YOLO("yolov8l-pose.pt")
for subtask in goods_and_bads.keys():
    if subtask == "press hard at a rate of 100 to 120 compressions per minute":
        for vid_name in goods_and_bads[subtask]:
            if 'time_stamps_file_paths_poses' in goods_and_bads[subtask][vid_name].keys():
                for timewindow in goods_and_bads[subtask][vid_name]['time_stamps_file_paths_poses']:
                        # print(goods_and_bads[subtask][vid_name]['time_stamps_file_paths_poses'][timewindow].keys()) # only need to check one because they always co-exist- double check
                        if "good_executions_list" in goods_and_bads[subtask][vid_name]['time_stamps_file_paths_poses'][timewindow].keys():
                            good_executions_list = goods_and_bads[subtask][vid_name]['time_stamps_file_paths_poses'][timewindow]["good_executions_list"]
                            needs_improvement_list = goods_and_bads[subtask][vid_name]['time_stamps_file_paths_poses'][timewindow]["needs_improvement_list"]
                        for camera_angle_clip in goods_and_bads[subtask][vid_name]['time_stamps_file_paths_poses'][timewindow]:
                            if 'commentary' not in camera_angle_clip and 'good_executions_list' not in camera_angle_clip and 'needs_improvement_list' not in camera_angle_clip:
                                for idx, (frame_path, empty_list) in enumerate(goods_and_bads[subtask][vid_name]['time_stamps_file_paths_poses'][timewindow][camera_angle_clip]):
                                    frame = Image.open(frame_path)
                                    # generate pose
                                    frame_pose_result = pose_model(frame) 
                                    # extract pose 
                                    pose = frame_pose_result[0].keypoints.data.cpu().numpy().tolist()
                                    goods_and_bads[subtask][vid_name]['time_stamps_file_paths_poses'][timewindow][camera_angle_clip][idx] = (frame_path, pose)
                                    print(pose)
        # print(goods_and_bads[subtask][vid_name]['time_window'])
        # print(goods_and_bads[subtask][vid_name]['time_window'].keys())
    # print((subtask))
    # print(goods_and_bads[subtask].keys())

test_pose_path = "/home/jess/Qwen3-VL-Embedding/all_info.json"
with open(test_pose_path, 'w') as f:
    json.dump(goods_and_bads, f)
exit()
    # if 'nov' not in video_name:
    #     documents.append(goods_and_bads[video_name]['good'])
    #     exp_video_names.append(video_name)
    #     print("doc added")
    # else:
    #     queries.append(goods_and_bads[video_name]['bad'])
    #     queries.append(goods_and_bads[video_name]['good'])
    #     print("query added")
    #     query_vid_names.append(video_name)

print(f'documents: {documents}')
print(f'queries: {queries}')

# Example documents and query
# documents = [
#     "Apple, banana, orange, grape, blueberry, pineapple, juice, house, car",
#     "Banana, orange",
#     "Grape, apple",
#     # ...,
# ]
# query_text = "Apple, banana"

dense_documents = [
    models.Document(text=doc, model="BAAI/bge-small-en")
    for doc in documents
]
dense_queries = [
    models.Document(text=query, model="BAAI/bge-small-en")
    for query in queries
        
]

colbert_documents = [
    models.Document(text=doc, model="colbert-ir/colbertv2.0")
    for doc in documents
]
colbert_queries = [
    models.Document(text=query, model="colbert-ir/colbertv2.0")
    for query in queries
]

collection_name = "dense_multivector_demo"
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
            # Leave HNSW indexing ON for dense
        ),
        "colbert": models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=models.HnswConfigDiff(m=0)  # Disable HNSW for reranking
        )
    }
)

points = [
    models.PointStruct(
        id=i,
        vector={
            "dense": dense_documents[i],
            "colbert": colbert_documents[i]
        },
        payload={"text": documents[i]}
    ) for i in range(len(documents))
]
client.upload_points(
    collection_name="dense_multivector_demo", 
    points=points, 
    batch_size=8
)
# May or may not need the dense vector part, toggle it on and off
results = client.query_points(
    collection_name="dense_multivector_demo",
    prefetch= [
    #     models.Prefetch(
    #     query=dense_queries[0],
    #     using="dense", # only good expert in there for now
    # ),
    models.Prefetch(
        query=colbert_queries[0],
        using="colbert", # only good expert in there for now
        limit=3
    ),
    models.Prefetch(
        query=colbert_queries[1],
        using="colbert", # only good expert in there for now
        limit=3,
    )],
    query=models.RrfQuery(rrf=models.Rrf(weights=[2.0, 1.0])), # try 2 and sweep some hyperparams maybe 
    with_payload=True
    # query=colbert_query,
    # using="colbert",
    # limit=3,
    # with_payload=True
)

print(results)
# print(colbert_queries[0] @ colbert_queries[1])
