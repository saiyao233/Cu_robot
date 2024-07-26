import numpy as np
import torch
seed=torch.load("succ_seed.pth")
# mid=seed[:,1:29]
seed=seed[0].cpu().numpy()

# print(mid)
# end=seed[:,31]
# mid[:,-1]=end
# print(seed[0])
# print(mid)
# seed = np.load("succ_seed.npy", allow_pickle=True)
# # print(seed.type
# tensor = torch.from_numpy(seed)
# print(tensor )

np.savetxt("succ_seed.csv", seed, delimiter=",", fmt="%.7f")