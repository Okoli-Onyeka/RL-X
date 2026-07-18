from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


class ProcessData():
    def __init__(self, text = "sentence-transformers/all-MiniLM-L6-v2"):
        self.path = "commands.csv"
        self.data = pd.read_csv(self.path)
        self.model = SentenceTransformer(text)

    def loadXData(self):
        text = self.data["text"].to_list()
        return text
        
    def loadYData(self):
        y = self.data[["x", "y", "yaw"]].to_numpy(dtype=float)
        return y
        
    def getEmbeddings(self, text):

        embeddings = self.model.encode(text)
        return embeddings

    def getSimilarities(self, embeddings):
        similarities = self.model.similarity(embeddings, embeddings)
        return similarities



    # sentences = [
    #         "Move to the Left.",
    #         "Don't go right.",
    #         "Turn leftwards.",
    #         "Move to the right.",
    #         "Don't go left.",
    #         "Turn rightwards."
    #     ]