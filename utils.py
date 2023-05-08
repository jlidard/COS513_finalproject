import torch
import itertools
import matplotlib.pyplot as plt
import numpy as np

def filter_traj(traj):
    '''
    Partition the trajectories based on order of crossing of the intersection.
    Assume N! (factorial) modes for each agent.

    :param
    :return:

    assignments
    mode_histogram

    '''
    num_modes = 4 * 3 * 2 * 1
    batch_size, num_agents, timesteps, spatial_dim = traj.shape
    if traj.shape[-1] > 2:
        spatial_dim_x = 1
        spatial_dim_y = 2
    else:
        spatial_dim_x = 0
        spatial_dim_y = 1
    endpoints = traj[:, :, :, spatial_dim_x:spatial_dim_y+1]
    beginning_points = traj[:, :, 0:1, spatial_dim_x:spatial_dim_y+1]

    mode_histogram = torch.zeros(num_modes)
    assignments = []

    permutations = list(itertools.permutations([0, 1, 2, 3]))

    sq_disp = ((endpoints - beginning_points)**2).sum(-1)
    arc_length = sq_disp.cumsum(-1)
    arc_length = arc_length[..., -1]
    order = torch.argsort(arc_length, -1)

    for batch_order in order:
        key = 0
        for perm in permutations:
            if tuple([int(x) for x in batch_order]) == perm:
                break
            key += 1
        mode_histogram[key] += 1
        assignments.append(key)

    return assignments, mode_histogram, permutations

def plot_traj_heatmap(traj, assignments, hist, header='collage'):

    cm = plt.cm.get_cmap('RdYlBu')
    batch_size, num_agents, timesteps, spatial_dim = traj.shape

    max_figs = 1
    max_items_in_plot = 250

    figs = 0
    items_in_plot = 0

    plt.clf()
    plt.cla()

    batches = []

    num_iter = max_items_in_plot

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
            batches.append(indices)

    if traj.shape[-1] > 2:
        spatial_dim_x = 1
        spatial_dim_y = 2
    else:
        spatial_dim_x = 0
        spatial_dim_y = 1

    for b in batches:
        batch_traj = traj[b]
        for agent_traj in batch_traj:
            t = np.arange(timesteps)
            sc = plt.scatter(-1*agent_traj[:, spatial_dim_x], agent_traj[:, spatial_dim_y],
                             c=t, vmin=0, vmax=timesteps, cmap=cm)

        if figs > max_figs:
            return

        if items_in_plot > max_items_in_plot:
            cbar = plt.colorbar(sc)
            plt.xlabel('East Position')
            plt.ylabel('North Position')
            cbar.ax.set_title(f't={timesteps}')
            cbar.ax.set_xlabel(f't={0}')
            plt.savefig(f'./plots/{header}{figs}.png')
            plt.clf()
            plt.cla()
            figs += 1
            items_in_plot = 0
        else:
            items_in_plot += 1

def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

    plt.yscale('log')

    plt.savefig('plots/hist.png')

def normalize(traj):

    return traj / traj.sum(-1)

def l2_loss(x, x_hat):
    sq_error = (x-x_hat)**2
    sq_error = sq_error.sum(-1).sum(-1).sum(-1)
    return sq_error

def mon_loss(x, x_hat):

    num_samples = x_hat.shape[-1]

    l2_loss_list = []
    min_l2_loss = 0
    for sample in range(num_samples):
        loss = l2_loss(x, x_hat[..., sample])
        l2_loss_list.append(loss)
    min_losses = torch.stack(l2_loss_list, -1).min(-1).values
    ind = torch.stack(l2_loss_list, -1).min(-1).indices
    return min_losses, ind, l2_loss_list






if __name__ == "__main__":
    a = torch.randn(10, 4, 30, 7)
    filter_traj(a)

    plot_traj_heatmap(a)

