import torch
import torch.nn as nn


class HierarchicalEmbodimentAgnosticNetwork():
    def __init__(self):
        self.create_networks()

    def create_networks(self):

        self.scores_net = nn.Sequential(
            nn.Linear(23,100),
            nn.ELU(),
            nn.Linear(100,23),
        )

        self.value_net = nn.Sequential(
            nn.Linear(23, 100),
            nn.ELU(),
            nn.Linear(100, 23)
        )

        self.main_net = nn.Sequential(
            nn.Linear(407, 500),
            nn.Sigmoid(),
            nn.Linear(500, 500),
            nn.Sigmoid(),
            nn.Linear(500, 500),
            nn.Sigmoid(),
            nn.Linear(500, 100),
            nn.Sigmoid(),
            nn.Linear(100, 19)
        )
    
    def load_networks(self, filenames = ["score_nn.pth","value_nn.pth","main_nn.pth"]):
        self.scores_net.load_state_dict(torch.load(f"saved_networks/{filenames[0]}"))
        self.value_net.load_state_dict(torch.load(f"saved_networks/{filenames[1]}"))
        self.main_net.load_state_dict(torch.load(f"saved_networks/{filenames[2]}"))

    def get_commands(self, _desc_vector, _embedding):
        self.scores_net.eval()
        self.value_net.eval()
        self.main_net.eval()

        with torch.no_grad():
            _scores = self.scores_net(_desc_vector)
            _weights = torch.softmax(_scores, dim=-1)

            _values = self.value_net(_desc_vector)

            _latent = torch.multiply(_values, _weights)
            _latent_sum = _latent.sum(dim=0).unsqueeze(0)

            _input = torch.cat([_embedding, _latent_sum], dim=1)
            _output = self.main_net(_input)
            return _output