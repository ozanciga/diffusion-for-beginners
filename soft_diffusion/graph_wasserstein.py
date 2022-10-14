'''
graph based selection of gaussian blur levels
plug-in the resulting sigmas to the training script
'''

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from soft_diffusion.wasserstein import calculate_2_wasserstein_dist
import torch
import numpy as np
import torchvision.transforms as tt
from torch.utils.data import DataLoader
import torchvision


def get_shortest_path(M, start_node, end_node):
    # https://stackoverflow.com/a/53078901/1696420
    path = [end_node]
    k = end_node
    while M[start_node, k] != -9999:
        path.append(Pr[start_node, k])
        k = M[start_node, k]
    return path[::-1]


if __name__ == '__main__':

    gbs_min, gbs_max = 0.01, 23  # gaussian blur sigma - appendix D, blur
    T = 256
    sigma_blurs = np.linspace(gbs_min, gbs_max, T)

    distributions = dict()
    for sigma in sigma_blurs:
        distributions[sigma] = list()

        data_root = ''
        data_transform = tt.Compose([
            tt.Resize((64, 64)),
            tt.ToTensor(),
            tt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        dataset = torchvision.datasets.CelebA(data_root, split='train', target_type='attr', transform=data_transform, download=True)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        for batch in loader:
            distributions[sigma].append(batch[0])


    # compute wasserstein distances between each edge
    # ``we start with 256 different blur levels and we tune \eps such that
    # the shortest path contains 32 distributions.
    # we then use linear interpolation to extend to the continuous case."
    eps_dist_max = 1E-3  # dataset specific, eq 19, must set! this is arbitrary

    wasserstein_dist = torch.empty((T, T))
    for index, sigma in enumerate(sigma_blurs):
        for next_index, next_sigma in enumerate(sigma_blurs):
            if index == next_index:
                wasserstein_dist[index, next_index] = torch.inf
            else:
                wdist = calculate_2_wasserstein_dist(distributions[sigma], distributions[next_sigma])
                if wdist > eps_dist_max:
                    wdist = torch.inf
                wasserstein_dist[index, next_index] = wdist

    wasserstein_dist = csr_matrix(wasserstein_dist.cpu().tolist())
    D, Pr = shortest_path(wasserstein_dist, directed=False, method='FW', return_predecessors=True)
    # get the list of sigmas giving the shortest path
    sigma_index = get_shortest_path(wasserstein_dist, start_node=0, end_node=T-1)
    sigma_blurs = [sigma for j, sigma in enumerate(sigma_blurs) if j in sigma_index]
    assert len(sigma_blurs) == 32, 'adjust eps_dist_max in eqn 19 until you get 32 nodes/sigmas'

    print(sigma_blurs)