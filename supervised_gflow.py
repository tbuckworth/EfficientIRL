import torch as T
import numpy as np
import tqdm
import pickle
import os
import matplotlib.pyplot as plt
import einops

def main():
    device = 'cuda' if T.cuda.is_available() else 'cpu'

    device = T.device(device)

    horizon = 16

    ndim = 2

    n_hid = 256
    n_layers = 2
    # n_hid = 256
    # n_layers = 2
    reward_temp = 1
    generation_temp = 6.0
    generate_drop_in = True
    eval_drop_in = False
    r_0 = np.exp(-20.0)
    r_mode = 5.0

    low_reward_structure = False  # False means use the normal hypergrid reward (but with possibly changed r_mode), True means use my modified reward.
    low_reward_structure_random = False
    bs = 128  # batch size

    uniform_pb = False


    def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
        return T.nn.Sequential(*(sum(
            [[T.nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
             for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))


    # def log_reward(x):  #the original version
    #     ax = abs(x / (horizon-1) * 2 - 1)
    #     return ((ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + r_0).log()

    def my_log_reward(z):  # my changed version
        reward = T.zeros(z.shape[:-1], device=z.device)

        # # Condition 3: low reward structure along edge
        if low_reward_structure:
            edges = (((z == horizon - 1) | (z == 0)).sum(-1) == ndim - 1)
            distance_from_corner = T.minimum(T.abs(horizon - 1 - z[edges]), z[edges]).float().max(dim=-1)[0]
            # for i in range(10):
            #     print(f"\ncoordinate = {z[edges][i]}")
            #     print(f"max dist from a high edge {distances_from_corner[0][i]}")
            #     print(f"max dist from a low edge {distances_from_corner[1][i]}")
            #     print(f"distance from corner = {distance_from_corner[i]}")

            reward[edges] = np.exp(-distance_from_corner)
        elif low_reward_structure_random:
            edges = (((z == horizon - 1) | (z == 0)).sum(-1) == ndim - 1)
            exponents = T.randint(5, 20, (len(reward[edges]),))
            reward[edges] = np.exp(-exponents).float()

        # Condition 2: fairly high reward around modes
        plateaus = ((z == horizon - 1) | (z == horizon - 2) | (z == 0) | (z == 1)).all(dim=-1)
        reward[plateaus] = 2.0

        # Condition 1: modes in corners
        modes = ((z == horizon - 1) | (z == 0)).all(dim=-1)
        reward[modes] = r_mode

        reward = T.maximum(reward, T.tensor(r_0))

        log_reward = reward.log()
        return log_reward / reward_temp


    def plot_modes_vs_iterations(it, n_modes_off_p_log, n_modes_on_p_log, bs):
        plt.figure(figsize=(10, 6))
        plt.plot(it, n_modes_off_p_log, label="off-policy")
        plt.plot(it, n_modes_on_p_log, label="on-policy")

        # Add the red dashed line at y = number of dimensions (assuming it's the length of it)
        plt.plot(it, [2 ** ndim] * len(it), 'r--', label='max')

        plt.title(f"Number of unique modes in {bs} samples")
        plt.xlabel("Iterations")
        plt.ylabel("Modes")

        # Set y-axis to show only integer values
        y_max = 2 ** ndim
        plt.ylim(0, y_max)
        plt.yticks(range(0, int(y_max) + 2))

        plt.grid(True)
        plt.legend()

        # Create the 'results' folder if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')

        # Save the plot
        plt.savefig('results/modes_vs_iterations.png')
        plt.close()


    def plot_reward(truelr):
        # Create the 'results' folder if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')

        plt.figure(figsize=(8, 8))
        plt.imshow(truelr, cmap='Reds', origin='lower')
        plt.colorbar(label='Log Reward')
        plt.title('Reward Grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.legend()
        plt.savefig(f'results/reward_distribution.png')
        plt.close()


    def plot_reward_with_scatter(truelr, z, it, label):
        plt.figure(figsize=(8, 8))
        plt.imshow(truelr, cmap='Reds', origin='lower')
        plt.colorbar(label='Log Reward')
        plt.title(f'Distribution after {it} iterations ({label})')
        plt.xlabel('X')
        plt.ylabel('Y')

        x = z[:, 0].cpu().numpy()
        y = z[:, 1].cpu().numpy()
        plt.scatter(x, y, color='g')

        plt.savefig(f'results/{label}_reward_distribution_with_samples.png')
        plt.close()


    j = T.zeros((horizon,) * ndim + (ndim,))
    for i in range(ndim):
        jj = T.linspace(0, horizon - 1, horizon)
        for _ in range(i): jj = jj.unsqueeze(1)
        j[..., i] = jj

    truelr = my_log_reward(j)
    plot_reward(truelr)

    total_reward = truelr.view(-1).logsumexp(0).exp()
    p_mode = r_mode / total_reward
    n_modes = 2 ** ndim
    print('total reward= {total_reward}')
    print(f'Target prob for each mode is {p_mode}')
    print(f"target Z = {total_reward.log()}")
    print(f"Expected number of unique modes in {bs} draws is at least {min(n_modes, bs * p_mode)}")
    # print(f"and less than {bs*(1-(1-p_mode)**bs)}")
    true_dist = truelr.flatten().softmax(0).cpu().numpy()


    def toin(z):
        return T.nn.functional.one_hot(z, horizon).view(z.shape[0], -1).float()


    Z = T.zeros((1,)).to(device)

    model = make_mlp([ndim * horizon] + [n_hid] * n_layers + [2 * ndim + 1]).to(device)
    opt = T.optim.Adam([{'params': model.parameters(), 'lr': 0.001}, {'params': [Z], 'lr': 0.1}])
    Z.requires_grad_()

    losses = []
    Zs = []
    # zs = []
    all_visited = []
    first_visit = -1 * np.ones_like(true_dist)
    l1log = []
    n_modes_off_p_log = []
    n_modes_on_p_log = []
    it_log = []


    def generate_batch(generation_temp=1.0, drop_in=False):
        if drop_in:
            z = T.randint(horizon, (bs, ndim), dtype=T.long).to(device)
        else:
            z = T.zeros((bs, ndim), dtype=T.long).to(device)  # the current position for each trajectory in the batch

        done = T.full((bs,), False, dtype=T.bool).to(device)

        action = None

        ll_diff = T.zeros((bs,)).to(device)
        ll_diff += Z
        i = 0
        while T.any(~done):

            pred = model(toin(z[~done]))

            edge_mask = T.cat([(z[~done] == horizon - 1).float(), T.zeros(((~done).sum(), 1), device=device)], 1)
            logits = (pred[..., :ndim + 1] - 1000000000 * edge_mask).log_softmax(1)

            init_edge_mask = (z[~done] == 0).float()
            back_logits = ((0 if uniform_pb else 1) * pred[...,
                                                      ndim + 1:2 * ndim + 1] - 1000000000 * init_edge_mask).log_softmax(1)
            # What's going on with this backward policy? I'd rather learn a backward policy just like the forward one. Not sure what this is doing.

            # This is learning a backward policy mate. model predicts both forwards and backwards, the first ndim logits
            # are the forward policy, the last ndim logits are the backward policy.
            if action is not None:
                ll_diff[~done] -= back_logits.gather(1, action[action != ndim].unsqueeze(1)).squeeze(1)

            exp_weight = 0.
            sample_ins_probs = (1 - exp_weight) * (logits / generation_temp).softmax(1) + exp_weight * (1 - edge_mask) / (
                        1 - edge_mask + 0.0000001).sum(1).unsqueeze(1)

            action = sample_ins_probs.multinomial(1)
            ll_diff[~done] += logits.gather(1, action).squeeze(1)

            terminate = (action == ndim).squeeze(1)
            for x in z[~done][terminate]:
                state = (x.cpu() * (horizon ** T.arange(ndim))).sum().item()
                if first_visit[state] < 0: first_visit[state] = it
                all_visited.append(state)

            done[~done] |= terminate

            with T.no_grad():
                z[~done] = z[~done].scatter_add(1, action[~terminate],
                                                T.ones(action[~terminate].shape, dtype=T.long, device=device))

            i += 1

        lr = my_log_reward(z.float())
        ll_diff -= lr
        loss = (ll_diff ** 2).sum() / (bs)

        return z, lr, loss


    for it in tqdm.trange(200001):
        opt.zero_grad()

        z, lr, loss = generate_batch(generation_temp=generation_temp, drop_in=generate_drop_in)
        loss.backward()
        opt.step()
        losses.append(loss.item())

        Zs.append(Z.item())

        if it % 100 == 0:
            it_log.append(it)
            print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())
            emp_dist = np.bincount(all_visited[-200000:], minlength=len(true_dist)).astype(float)
            emp_dist /= emp_dist.sum()
            l1 = np.abs(true_dist - emp_dist).mean()
            print('L1 =', l1)
            l1log.append((len(all_visited), l1))

            # off policy
            lr_mask = np.isclose(lr.cpu(), np.log(r_mode))
            z_values_at_lr = z[lr_mask]
            n_unique_modes = len(z_values_at_lr.unique(dim=0))
            print(f"off policy: {n_unique_modes} unique modes in {bs} samples")
            n_modes_off_p_log.append(n_unique_modes)
            # zs.append(z)
            plot_reward_with_scatter(truelr, z, it, "off-policy")

            # on policy
            z, lr, loss = generate_batch(generation_temp=1.0, drop_in=eval_drop_in)
            lr_mask = np.isclose(lr.cpu(), np.log(r_mode))
            z_values_at_lr = z[lr_mask]
            n_unique_modes = len(z_values_at_lr.unique(dim=0))
            print(f"on policy: {n_unique_modes} unique modes in {bs} samples")
            n_modes_on_p_log.append(n_unique_modes)
            plot_modes_vs_iterations(it_log, n_modes_off_p_log, n_modes_on_p_log, bs)
            # zs.append(z)
            plot_reward_with_scatter(truelr, z, it, "on-policy")

    pickle.dump([losses, Zs, all_visited, first_visit, l1log], open(f'results/out.pkl', 'wb'))

if __name__ == "__main__":
    main()
