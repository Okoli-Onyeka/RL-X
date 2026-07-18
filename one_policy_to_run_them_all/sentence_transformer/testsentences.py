import pandas as pd
from sentence_transformers import SentenceTransformer
from multilayerperceptron import MultiLayerPerceptron


data = pd.read_csv("commands.csv")

sentences = data["text"].to_list()

print(sentences)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(sentences)

print(embeddings.shape)

y = data[""]

# mlp = MultiLayerPerceptron(layers_size = [384, 300, 200, 50, 10, 4])

# mlp.train(embeddings, )