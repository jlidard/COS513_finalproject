
from torch.optim import Adam
from torch import nn
import torch
from torch.utils.data import DataLoader
from attention_vae import AttentionVAE
from pandas_dataset import CustomDataset
from torch.distributions import Categorical
from torch.distributions import RelaxedOneHotCategorical, OneHotCategorical

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm
import pandas as pd
import os

from utils import *

from tqdm import tqdm


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD  = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def vae_loss(x, x_hat, mean, log_var):
    reproduction_loss = l2_loss(x, x_hat)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD




def train(eval=False, load=False, from_scratch=False):

    # Model Hyperparameters

    dataset_path = '~/datasets'

    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    #todo: overwrite dims
    batch_size = 32
    x_dim = 784
    hidden_dim = 540
    latent_dim = 24
    split_percentage = 0.8

    num_csvs = 2


    if from_scratch:
        df_list = []
        path = os.getcwd()
        csv_path = os.path.join(path, "datasets", "csv")
        for filename in os.listdir(csv_path):
            df = pd.read_csv(os.path.join(csv_path, filename))
            df_list.append(df)
        N = len(df_list)
        print(f"found {N} instances")
        indices = np.arange(N)
        np.random.shuffle(indices)
        split_index = int(split_percentage*N)
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        print(test_indices)
        train_dataset = CustomDataset(df_list)
        test_dataset = CustomDataset(df_list)
        print(f"train dataset size: {train_dataset.data.shape[0]}")
        print(f"train dataset size: {test_dataset.data.shape[0]}")
        torch.save(train_dataset, 'datasets/train_dataset.pt')
        torch.save(test_dataset, 'datasets/test_dataset.pt')

        traj = train_dataset.data
        assignments, hist, permutations = filter_traj(traj)

        print(hist)
        print(hist.sum())

        initial_hist = hist/hist.sum()

        # Augments 13
        agent_i = 1
        agent_j = 2
        matches = (torch.Tensor(assignments) == 7).nonzero().squeeze()
        test_traj = traj[matches]
        new_data = train_dataset.data_augment(agent_i, agent_j, test_traj)

        # Augments 12
        agent_i = 2
        agent_j = 3
        matches = (torch.Tensor(assignments) == 18).nonzero().squeeze()
        test_traj = traj[matches]
        new_data = train_dataset.data_augment(agent_i, agent_j, test_traj, "rotate_left")

        # Augments 14
        agent_i = 2
        agent_j = 3
        matches = (torch.Tensor(assignments) == 20).nonzero().squeeze()
        test_traj = traj[matches]
        new_data = train_dataset.data_augment(agent_i, agent_j, test_traj, "rotate_left")

        # Augments 15
        agent_i = 0
        agent_j = 2
        matches = (torch.Tensor(assignments) == 1).nonzero().squeeze()
        test_traj = traj[matches]
        new_data = train_dataset.data_augment(agent_i, agent_j, test_traj, "rotate_right")

        # Augments 17
        agent_i = 2
        agent_j = 3
        matches = (torch.Tensor(assignments) == 4).nonzero().squeeze()
        test_traj = traj[matches]
        new_data = train_dataset.data_augment(agent_i, agent_j, test_traj, "rotate_left")

        # Augments 16
        agent_i = 1
        agent_j = 2
        matches = (torch.Tensor(assignments) == 10).nonzero().squeeze()
        test_traj = traj[matches]
        new_data = train_dataset.data_augment(agent_i, agent_j, test_traj)

        # Augments 2
        agent_i = 2
        agent_j = 3
        matches = (torch.Tensor(assignments) == 23).nonzero().squeeze()
        test_traj = traj[matches]
        new_data = train_dataset.data_augment(agent_i, agent_j, test_traj, "rotate_left")


        traj = train_dataset.data
        assignments, hist, permutations = filter_traj(traj)

        print(hist)
        print(hist.sum())

        data_hist = hist/hist.sum()

        test_dataset.reduce_indices(test_indices)

        new_train_indices = list(range(train_dataset.data.shape[0]))
        t1 = torch.Tensor(new_train_indices)
        t2 = torch.Tensor(test_indices)
        combined = torch.cat((t1, t2))
        uniques, counts = combined.unique(return_counts=True)
        difference = uniques[counts == 1]
        train_dataset.reduce_indices(difference)

        print(f"train dataset size: {train_dataset.data.shape[0]}")
        print(f"test dataset size: {test_dataset.data.shape[0]}")

        torch.save(train_dataset, 'datasets/train_dataset.pt')
        torch.save(test_dataset, 'datasets/test_dataset.pt')

        plot_traj_heatmap(traj, assignments, hist)

        torch.save(initial_hist, 'initial_hist.pt')
        torch.save(data_hist, 'data_hist.pt')


        return

    else:
        train_dataset = torch.load('datasets/train_dataset.pt')
        test_dataset = torch.load('datasets/test_dataset.pt')


    traj = train_dataset.data
    assignments, hist, permutations = filter_traj(traj)

    log_empirical_mode_hist = torch.Tensor(hist) + 1e-15
    log_empirical_mode_hist /= log_empirical_mode_hist.sum()
    log_empirical_mode_hist = log_empirical_mode_hist.log()


    lr = 3e-4
    top_train_epochs = 125

    if eval:
        epochs = 1
    else:
        epochs = 501 #top_train_epochs

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # TODO


    # TRAJECTORY format should be [batch_size, num_agents, timesteps, spatial_dim (=2)]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    BCE_loss = nn.BCELoss()



    kld = torch.nn.KLDivLoss(log_target=True, reduction="batchmean")

    train_dataset.assign_modes(assignments, hist/hist.sum())

    pred_horizon = train_dataset.sequence_length - train_dataset.pred_len
    model = AttentionVAE(sequence_length=train_dataset.sequence_length,
                         num_agents=train_dataset.num_agents,
                         latent_dim=latent_dim,
                         embedding_dim=hidden_dim,
                         num_samples=1,
                         discrete=True)
    if load:
        model.load_state_dict(torch.load(f'checkpoints/epoch25.pt'))
        #model.eval()
    optimizer = Adam(model.parameters(), lr=lr)

    # if eval:
    #     model.training = False
    print("Start training VAE...")
    model.train()

    pred_data = []
    z_data = []
    truth_data = []
    temp = 100

    num_iter = 50
    data = train_dataset.data_reduce(assignments, hist, num_iter=num_iter)
    data = torch.stack(data, 0)
    train_dataset.data = data
    train_dataset.update_stats()
    traj = train_dataset.data
    assignments, hist, permutations = filter_traj(traj)
    plot_traj_heatmap(traj, assignments, hist, "new_data")

    one_hot = torch.eye(latent_dim)

    #train_dataset.data_normalize()
    losses = []

    for epoch in range(epochs):
        overall_loss = 0
        overall_kl = 0
        overall_mse = 0
        count = 0
        expon = 1 + epoch
        # if epoch > 2:
        #     temp = 0.05
        # elif epoch > 3:
        #     temp = 0.01
        # temp /= 2.5
        model.set_temp(temp)
        for (x, x_truth, m, p, ind, raw_traj) in tqdm(train_loader):

            optimizer.zero_grad()


            one_hots = [one_hot[x] for x in m]
            one_hots = torch.stack(one_hots, 0)

            x = train_dataset.data_normalize(x)
            x_truth = train_dataset.data_normalize(x_truth)

            x_hat, mean, log_var, z = model(x)
            # z = one_hots
            # x_hat = model.decode(one_hots)
            # mean = torch.ones_like(mean)
            # log_var = None

            loss = model.loss_function(x_truth, x_hat, mean, log_var) #vae_loss(x[..., :2], x_hat, mean, log_var).mean()
            mu = loss["mu"]
            mse = loss["Reconstruction_Loss"].mean()
            KL = loss["KLD"].mean()
            obj = loss['loss'].mean()
            #obj += 100*((p - mu[:, m])**2).mean()

            class_error = ((z.squeeze(1)-one_hots)**2).sum(-1)
            #obj += 1*class_error.mean()

            log_p = (mu+1e-15).log()

            batch_size = x_hat.shape[0]

            kldiv_metric = kld(log_p, log_empirical_mode_hist.unsqueeze(0).repeat(batch_size, 1)).detach()

            overall_loss += obj
            overall_mse += mse
            overall_kl += KL

            obj.backward()
            optimizer.step()

            x_hat = train_dataset.data_unormalize(x_hat.squeeze(-1))
            x_truth = train_dataset.data_unormalize(x_truth[..., 1:3])
            x_hat = x_hat.squeeze(-1) # train_dataset.unormalize(x_hat.squeeze(-1))
            test = x_truth[..., 1:3] #train_dataset.unormalize()
            pred_data.append(x_hat)
            z_data.append(z)
            truth_data.append(x_truth)

            #TAKE OFF WHEN TRAINING
            # if count > 5:
            #     break

            count += 1

        dist = Categorical(probs=mu.mean(0).detach())
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss/(count *  batch_size))
        print("\t KL Loss: ", overall_kl / (count * batch_size))
        print("\t MSE Loss: ", overall_mse / (count * batch_size))
        #print(f"Mode entropy: {dist.entropy()}, KL Divergence of modes: {kldiv_metric}")
        print(z.sum(0))
        losses.append(overall_loss.detach()/count)

        if not eval and epoch%25 == 0:
            torch.save(model.state_dict(), f'checkpoints/epoch{epoch}.pt')

    if not eval:
        plt.figure()
        plt.plot(losses)
        plt.savefig('plots/epoch_losses.png')
        return

    pred_data = torch.cat(pred_data, 0).squeeze(-1).detach()
    z_data = torch.cat(z_data, 0).squeeze(-1).detach()
    truth_data = torch.cat(truth_data, 0).detach()

    print(f"z_count: {z_data.sum(0)}")

    z_hist = z_data.sum(0) / z_data.sum()
    z_ent = torch.sum(z_hist * z_hist.log())

    print(f"z_entropy: {z_ent}")

    print(f"Modes covered: {torch.count_nonzero(z_hist)}")

    assignments, hist_data, permutations = filter_traj(pred_data)
    plot_traj_heatmap(pred_data, assignments, hist_data, header="reconstructions")

    assignments, hist_truth, permutations = filter_traj(truth_data)
    plot_traj_heatmap(truth_data, assignments, hist_truth, header="truth_samples")

    hist_data /= hist_data.sum()

    log_correction = 0
    hist_data += log_correction
    initial_hist = torch.load('initial_hist.pt')
    data_hist = hist_truth/hist_truth.sum()  # torch.load('data_hist.pt')
    initial_hist += log_correction
    data_hist += log_correction


    plt.figure()
    ax = plt.gca()
    histograms = {"Initial": initial_hist,
                  "Augmented": data_hist,
                  "Reconstructions": hist_data}

    bar_plot(ax, histograms)


    torch.save(pred_data, 'pred_data.pt')
    torch.save(z_data, 'z_data.pt')


    print("Finish!!")




if __name__ == "__main__":
    train(eval=False, load=False, from_scratch=True)
    train(eval=True, load=True)