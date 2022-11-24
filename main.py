import matplotlib.pyplot as plt
import numpy as np


def multimodal_gaussian(size: int, return_hist: bool = False):
    ## number of Gaussian distributions to be drawn later
    mixNum = 17

    ## Generate means and stds of those Gaussians
    mus = np.random.randint(1, 100, mixNum)

    sigmas = np.random.random(mixNum) * 10 + 1

    pos = np.arange(len(mus)) / len(mus)

    ## 0-based indexing
    idx = np.digitize(np.random.random(size), pos) - 1

    a = map(lambda x: mus[x], idx)
    b = map(lambda x: sigmas[x], idx)

    sample = [mu + sigma * np.random.randn() for mu, sigma in zip(a, b)]

    if return_hist:
        hist, _ = np.histogram(sample, bins=250)
        return hist

    return sample


def simulated_annealing(arr: np.ndarray):

    length = len(arr)

    ## Initial temperature
    temperature = 100

    ## Current state
    s_curr = np.random.randint(length)

    trajectory = []

    while temperature >= 1:

        ## Next state
        s_next = np.random.randint(
            low=max(0, s_curr - int(0.3 * length)),
            high=min(length, s_curr + int(0.3 * length)),
        )

        ## Current energy
        E_curr = -arr[s_curr]

        ## Next energy
        E_next = -arr[s_next]

        ## Difference of energy
        loss = E_next - E_curr

        if loss < 0:
            s_curr = s_next
        else:
            prob = np.exp(-loss / (temperature * 0.2))

            if prob >= np.random.random():
                s_curr = s_next

        ## Record the state, energy, and temperature
        ## E_curr may have been changed, use -arr[s_curr] instead
        trajectory.append([s_curr, -arr[s_curr], temperature])
        temperature -= 1

    return s_curr, trajectory


if __name__ == "__main__":
    figsize = (24, 10)

    # sample = multiModalGaussian(size=5000)
    # plt.hist(sample, bins=120);
    plt.figure(figsize=figsize)
    hist = multimodal_gaussian(size=10000, return_hist=True)
    plt.plot(hist, linewidth=6)

    #
    s_final, trajectory = simulated_annealing(arr=hist)
    plt.figure(figsize=figsize)
    plt.plot(hist, linewidth=6)

    ## Returns only the position of the final state
    plt.plot(
        s_final,
        hist[s_final],
        marker="o",
        markersize=24,
        linewidth=12,
    )
