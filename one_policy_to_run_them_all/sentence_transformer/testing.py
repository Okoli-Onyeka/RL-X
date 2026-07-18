from semanticreasoning import getInputData 
import torch

sentences = [
        "Move to the Left.",
        "Don't go right.",
        "Turn leftwards.",
        "Move to the right.",
        "Don't go left.",
        "Turn rightwards."
    ]

layers = [384, 300, 200, 50, 10, 4]
input = getInputData(sentences[0])
X = torch.tensor(input)