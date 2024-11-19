# Description: This script is used to generate embeddings for a list of placeholder queries and save them to .npy files.
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())

# Initialize the model and specify the device
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# List of placeholder queries
queries = [
    "What is learning rate in gradient descent?",
    "What is Microbial biogeography?",
    "Give me details about The Arch of Cabanes.",
    "Give me details about the history of the Taj Mahal.",
    "Tell me something about the labelling used on aid packages created and sent under the Marshall Plan"
]

# Generate embeddings and save each to a .npy file
for i, query in enumerate(queries, 1):
    embeddings = model.encode([query], device="cuda", batch_size=2)
    filename = f"../queries_data/query{i}.npy"
    np.save(filename, embeddings)
    print(f"Embeddings for Query {i} have been saved to {filename}")
