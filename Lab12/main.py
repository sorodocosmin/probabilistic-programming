import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    # prior = np.repeat(1/grid_points, grid_points)  # uniform prior
    prior = abs(grid - 0.5)
    # prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def ex_1(nr_heads=10, nr_tails=3):

    data = np.repeat([0, 1], (nr_tails, nr_heads))
    # print(data)
    points = nr_heads + nr_tails
    h = data.sum()
    t = len(data) - h
    grid_d, posterior_d = posterior_grid(points, h, t)
    plt.plot(grid_d, posterior_d, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ - prob of heads')
    plt.show()


def ex_2_estimate_pi(N):
    """
    :return: the error of the estimate
    """
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / N
    error = abs((pi - np.pi) / pi) * 100
    return error


def ex_2_avg_std_err(N, nr_simulations=100):
    """
    :return: a tuple containing the average error and the standard error of the estimate pi
    """
    errors = []
    for _ in range(nr_simulations):
        errors.append(ex_2_estimate_pi(N))

    avg = np.mean(errors)
    std_err = np.std(errors)

    return avg, std_err


def ex_2_plot_avg_std_err(possible_N):
    avg_errors = []
    std_errors = []

    for N in possible_N:
        avg, std_err = ex_2_avg_std_err(N)
        avg_errors.append(avg)
        std_errors.append(std_err)

    # plotting the results
    plt.errorbar(possible_N, avg_errors, yerr=std_errors, fmt='o-', capsize=5)
    plt.xscale('log')
    plt.xlabel('Number of Points (N)')
    plt.ylabel('Average Error (%)')
    plt.title('Estimation of pi: Relationship between N and Error')
    plt.show()

    # we can see that the error decreases as N increases, which is expected
    # as the number of points increases, the estimate of pi should get closer to the real value of pi


def ex_3_metropolis(alpha, beta, draws=10000):

    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = stats.beta.pdf(old_x, alpha, beta)
    delta = np.random.normal(0, 0.5, draws)

    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = stats.beta.pdf(new_x, alpha, beta)

        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x

    return trace

def ex_3():
    # alpha_prior, beta_prior = 2, 5  # Beta(2, 5) as prior
    # trace = ex_3_metropolis(alpha_prior, beta_prior)
    #
    # # Plot the results
    # func = stats.beta(alpha_prior, beta_prior)
    # x = np.linspace(0.01, 0.99, 100)
    # y = func.pdf(x)
    #
    # plt.figure(figsize=(8, 6))
    # plt.xlim(0, 1)
    # plt.plot(x, y, 'C1-', lw=3, label='True distribution')
    # plt.hist(trace[trace > 0], bins=25, density=True, label='Estimated distribution')
    # plt.xlabel('Theta')
    # plt.ylabel('pdf(Theta)')
    # plt.yticks([])
    # plt.legend()
    # plt.show()

    alpha_prior, beta_prior = 2, 5  # Same alpha and beta for both methods
    draws_metropolis = ex_3_metropolis(alpha_prior, beta_prior)

    grid, posterior_grid_result = posterior_grid(heads=alpha_prior, tails=beta_prior)

    # Plot the results side by side
    plt.figure(figsize=(12, 5))

    # Metropolis
    plt.subplot(1, 2, 1)
    plt.hist(draws_metropolis[draws_metropolis > 0], bins=25, density=True, alpha=0.5, label='Metropolis')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.title('Metropolis with Beta(2, 5) Prior')
    plt.legend()

    # Posterior Grid
    plt.subplot(1, 2, 2)
    plt.plot(grid, posterior_grid_result, label='Posterior Grid')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.title(f'Posterior Grid with heads={alpha_prior}, tails={beta_prior}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # ex_1()
    # ex_1(12, 6)
    # ex_1(47, 68)
    # ex_2_plot_avg_std_err([100, 500, 1_000, 5_000, 10_000, 100_000])
    ex_3()

if __name__ == '__main__':
    main()
