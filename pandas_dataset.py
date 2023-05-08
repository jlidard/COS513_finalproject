from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, dataframe_list):

        spatial_dim = 7
        # Convert pandas dataframe to tensor
        episode_tensor_list = []
        for df in dataframe_list:
            columns_of_interest = df.columns[2:]
            num_agents = len(columns_of_interest) // 7
            values = df[columns_of_interest].values
            values_as_tensor = torch.Tensor(values).reshape(-1, num_agents, spatial_dim)
            values_as_tensor = values_as_tensor.permute(1, 0, 2)
            episode_tensor_list.append(values_as_tensor)
        self.raw_tensor_list = episode_tensor_list

        # Truncate episodes
        sequence_lengths = [x.shape[1] for x in episode_tensor_list]
        median_episode_length = int(np.min(sequence_lengths))
        episode_tensor_list = [x for x in episode_tensor_list if x.shape[1] > median_episode_length]
        episode_tensor_list = [x[:, :median_episode_length] for x in episode_tensor_list]

        self.data = torch.stack(episode_tensor_list, 0)
        self.data = self.add_random_noise(self.data)
        self.update_stats()

        self.pred_len = self.sequence_length

        self.assignments = None
        self.modes = None

        self.batch_mean = None
        self.batch_std = None
        self.original_data = self.data

    def __getitem__(self, index):
        self.original_data = self.data
        traj = self.data[index]
        traj_pred = self.data[index]
        assignment = self.assignments[index]
        prob = self.hist[assignment]
        raw_traj = self.original_data[index]
        return traj, traj_pred, assignment, prob, index, raw_traj

    def __len__(self):
        return self.size

    def update_stats(self):
        self.size, self.num_agents, self.sequence_length, self.traj_dim = self.data.shape

    def data_augment(self, agent_i, agent_j, traj_anchor, type="reflection"):

        agent_i_traj = traj_anchor[:, agent_i]
        agent_j_traj = traj_anchor[:, agent_j]

        new_traj = traj_anchor.clone()
        new_traj_i = agent_j_traj.clone()
        new_traj_j = agent_i_traj.clone()

        if type == "reflection":
            new_traj_i[..., 1:] *= -1
            new_traj_j[..., 1:] *= -1
        elif type == "rotate_left":
            new_traj_i = self.column_swap_every_other(new_traj_i)
            new_traj_j = self.column_swap_every_other(new_traj_j)
            new_traj_i[..., 1::2] *= -1
            new_traj_j[..., 2::2] *= -1
        elif type == "rotate_right":
            new_traj_i = self.column_swap_every_other(new_traj_i)
            new_traj_j = self.column_swap_every_other(new_traj_j)
            new_traj_i[..., 1::2] *= -1
            new_traj_j[..., 2::2] *= -1

        new_traj[:, agent_i] = new_traj_i
        new_traj[:, agent_j] = new_traj_j
        new_traj = self.add_random_noise(new_traj)

        self.data = torch.cat((self.data, new_traj))
        self.update_stats()

        return new_traj

    def data_reduce(self, assignments, hist, num_iter = 1):

        batches = []

        for iter in range(num_iter):
            for i_item, item in enumerate(hist):
                matches = (torch.Tensor(assignments) == i_item).nonzero().squeeze()

                # 0 elements
                if matches.nelement() == 0:
                    continue

                # 1 element
                if len(matches.shape) == 0:
                    matches = matches.unsqueeze(-1)
                indices = np.random.choice(matches)
                batches.append(self.data[indices])

        return batches

    def data_normalize(self, traj):

        batch_mean = self.data.mean(dim=0).unsqueeze(0)
        batch_std = self.data.std(dim=0).unsqueeze(0)

        self.original_data = self.data.clone()
        new_traj = (traj - batch_mean) / (batch_std + 1e-3)


        self.batch_mean = batch_mean
        self.batch_std = batch_std

        return new_traj


    def data_unormalize(self, traj):
        return traj * (self.batch_std[..., 1:3] + 1e-3) + self.batch_mean[..., 1:3]


    def add_random_noise(self, in_tensor):
        noise = 0.001*torch.randn(in_tensor.shape)
        noise[..., 0] = 0
        noise[..., 3:] = 0
        noise = noise.cumsum(2)
        return in_tensor + noise

    def column_swap_every_other(self, traj):
        traj = traj.clone()
        old_traj = traj.clone()
        traj[..., 1::2] = old_traj[..., 2::2]
        traj[..., 2::2] = old_traj[..., 1::2]
        return traj

    def reduce_indices(self, indices):
        indices = torch.Tensor(indices).int()
        self.data = self.data[indices]
        self.update_stats()

    def assign_modes(self, assignments, hist):
        self.pred_len = self.sequence_length
        self.assignments = assignments
        self.hist = hist




