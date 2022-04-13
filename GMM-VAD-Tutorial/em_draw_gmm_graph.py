import numpy as np
import matplotlib.pyplot as plt
import math


def make_samples():
    # Group A
    muA_ans = [10, 30]
    covA_ans = [[6 ** 2, -30], [-30, 8 ** 2]]
    samplesA = np.random.multivariate_normal(muA_ans, covA_ans, 200).T

    # Group B
    muB_ans = [36, 27]
    covB_ans = [[5. ** 2, 20], [20, 6 ** 2]]
    samplesB = np.random.multivariate_normal(muB_ans, covB_ans, 100).T

    # Group C
    muC_ans = [25, 57]
    covC_ans = [[7 ** 2, -10], [-10, 6 ** 2]]
    samplesC = np.random.multivariate_normal(muC_ans, covC_ans, 130).T
    return np.column_stack((samplesA, samplesB, samplesC))


# Make data
samples = make_samples()

# Draw data
plt.scatter(samples[0], samples[1], color='g', marker="+")
# Parameter settings
K = 3  # Number of clusters
N = len(samples[0])  # Number of samples
plt.show()

from scipy.stats import multivariate_normal

# Three Gaussian Distribution
distributions = []
distributions.append(multivariate_normal(mean=[10, 30], cov=[[100, 0], [0, 100]]))
distributions.append(multivariate_normal(mean=[33, 22], cov=[[100, 0], [0, 100]]))
distributions.append(multivariate_normal(mean=[22, 51], cov=[[100, 0], [0, 100]]))

mixing_coefs = [1.0 / K for k in range(K)]


def draw(ds, X):
    x, y = np.mgrid[(min(X[0])):(max(X[0])):1, (min(X[1])):(max(X[1])):1]
    for d in ds:
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        plt.contour(x, y, d.pdf(pos), alpha=0.2)

    plt.scatter(X[0], X[1], color='g', marker="+")
    plt.show()


draw(distributions, samples)


def expectation_step(ds, X, pis):
    ans = []
    for n in range(N):
        ws = [pis[k] * ds[k].pdf([X[0][n], X[1][n]]) for k in range(K)]
        ans.append([ws[k] / sum(ws) for k in range(K)])

    return ans

    N_k = sum([gammas[n][k] for n in range(N)])

    tmp_x = sum([gammas[n][k] * X[0][n] for n in range(N)]) / N_k
    tmp_y = sum([gammas[n][k] * X[1][n] for n in range(N)]) / N_k
    mu = [tmp_x, tmp_y]

    ds = [np.array([[X[0][n], X[1][n]]]) - np.array([mu]) for n in range(N)]
    sigma = sum([gammas[n][k] * ds[n].T.dot(ds[n]) for n in range(N)]) / N_k

    return multivariate_normal(mean=mu, cov=sigma), N_k / N


def maximization_step(k, X, gammas):
    N_k = sum([gammas[n][k] for n in range(N)])

    tmp_x = sum([gammas[n][k] * X[0][n] for n in range(N)]) / N_k
    tmp_y = sum([gammas[n][k] * X[1][n] for n in range(N)]) / N_k
    mu = [tmp_x, tmp_y]

    ds = [np.array([[X[0][n], X[1][n]]]) - np.array([mu]) for n in range(N)]
    sigma = sum([gammas[n][k] * ds[n].T.dot(ds[n]) for n in range(N)]) / N_k

    return multivariate_normal(mean=mu, cov=sigma), N_k / N


def log_likelihood(ds, X, pis):
    ans = 0.0
    for n in range(N):
        ws = [pis[k] * ds[k].pdf([X[0][n], X[1][n]]) for k in range(K)]
        ans += math.log1p(sum(ws))

    return ans


def one_step():
    # E step
    gammas = expectation_step(distributions, samples, mixing_coefs)
    # M step
    for k in range(K):
        distributions[k], mixing_coefs[k] = maximization_step(k, samples, gammas)

    return log_likelihood(distributions, samples, mixing_coefs)


one_step()
draw(distributions, samples)

prev_log_likelihood = 0.0
for i in range(99):
    after_log_likelihood = one_step()

    if prev_log_likelihood / after_log_likelihood > 0.999:
        break
    else:
        prev_log_likelihood = after_log_likelihood

    if i % 3 == 0:
        plt.figure()
        draw(distributions, samples)

plt.figure()
draw(distributions, samples)

print("---------------------------------------------")
print("repeat: ", i + 1)
for k in range(K):
    print("Gauss", k, ": ")
    print("  share: ", mixing_coefs[k])
    print("  mean: ", distributions[k].mean)
    print("  cov: ", distributions[k].cov)

