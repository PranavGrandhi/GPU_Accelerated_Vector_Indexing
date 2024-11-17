from sentence_transformers import SentenceTransformer
import torch
import numpy as np

print("CUDA Available:", torch.cuda.is_available())

# Initialize the model and specify the device
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

# Input texts
texts = ["What is learning rate in gradient descent?"]

# Generate embeddings
embeddings = model.encode(texts, device="cuda", batch_size=2)

# Save embeddings to a .npy file
np.save("./queries_data/query1.npy", embeddings)

print("Embeddings have been saved to embeddings.npy")