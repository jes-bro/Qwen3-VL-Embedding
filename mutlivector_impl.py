import json
import subprocess
from qdrant_client import QdrantClient, models

# 1. Connect to Qdrant server
client = QdrantClient(":memory:")

from fastembed import TextEmbedding, LateInteractionTextEmbedding

# from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
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
    if 'nov' not in video_name:
        documents.append(goods_and_bads[video_name]['good'])
        exp_video_names.append(video_name)
        print("doc added")
    else:
        queries.append(goods_and_bads[video_name]['bad'])
        queries.append(goods_and_bads[video_name]['good'])
        print("query added")
        query_vid_names.append(video_name)

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
    prefetch= [models.Prefetch(
        query=dense_queries[0],
        using="dense", # only good expert in there for now
    ),
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
    query=models.RrfQuery(rrf=models.Rrf(weights=[1.0, 2.0, 1.0])), # try 2 and sweep some hyperparams maybe 
    with_payload=True
    # query=colbert_query,
    # using="colbert",
    # limit=3,
    # with_payload=True
)

print(results)
# print(colbert_queries[0] @ colbert_queries[1])
