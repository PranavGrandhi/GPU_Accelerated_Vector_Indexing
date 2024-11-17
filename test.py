from sentence_transformers import SentenceTransformer
import torch

print("CUDA Available:", torch.cuda.is_available())

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
texts = ["This is a test sentence.", "Another sentence here."]
embeddings = model.encode(texts, device="cuda", batch_size=2)

print("Embedding Shape:", embeddings.shape)