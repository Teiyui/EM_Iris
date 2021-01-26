import numpy as np
import math
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split

# Step 1 : Get the train dataset and test dataset
iris = pd.read_csv('../dataset/iris.csv')
x_df = iris.drop('species', axis=1)
y_df = iris['species']
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, train_size=0.2, random_state=10)

# Step 2 : Get the mean and variance of every variable
types = y_train.tolist()
X_train['species'] = types
type1 = X_train[X_train['species'] == 'Iris-versicolor']
type2 = X_train[X_train['species'] == 'Iris-virginica']
type3 = X_train[X_train['species'] == 'Iris-setosa']
types = []
types.append(type1)
types.append(type2)
types.append(type3)


def get_each_type_conditions(types, name):
    conditions = []
    for type in types:
        mean = type.drop(name, axis=1).mean()
        var = type.drop(name, axis=1).var()
        conditions.append([mean, var])
    return conditions


conds = get_each_type_conditions(types, 'species')

# Step 3 : Get the initial distribution of every type
def get_each_type_initial_distribution(conditions):
    distributions = []
    for condition in conditions:
        means = condition[0].tolist()
        variances = condition[1].tolist()
        distributions.append(multivariate_normal(mean=[means[0], means[1], means[2], means[3]],
                                                 cov=np.diag([variances[0], variances[1], variances[2],
                                                              variances[3]])))
    return distributions


distributions = get_each_type_initial_distribution(conds)


# Step 4 : E step -- find the weight list of every data based on the distributions
def expectation_step(distributions, X, former_weights_list):
    # weight list
    ans = []
    for n in range(N):
        # calculate percentage of every type according to pdf (need to normalize)
        ws = [former_weights_list[k] * distributions[k].pdf([X.iloc[n][0], X.iloc[n][1], X.iloc[n][2],
                                                             X.iloc[n][3]]) for k in range(K)]
        ans.append([ws[k] / sum(ws) for k in range(K)])

    return ans  # K * N numbers weight list


# Initialization process of the first weight list
K = len(distributions)
N = len(X_test)
former_weights_list = [1.0 / K for k in range(K)]
# ans = expectation_step(distributions, X_test, former_weights_list)


# Step 5 : M step -- update mean and variance by using weight list which got form E step
def maximization_step(k, X, E_step_weights_list):  # distributions number、train sample、weight list of each types
    N_k = sum([E_step_weights_list[n][k] for n in range(N)])

    # update mean
    tmp_x = sum([E_step_weights_list[n][k] * X.iloc[n][0] for n in range(N)]) / N_k
    tmp_y = sum([E_step_weights_list[n][k] * X.iloc[n][1] for n in range(N)]) / N_k
    tmp_m = sum([E_step_weights_list[n][k] * X.iloc[n][2] for n in range(N)]) / N_k
    tmp_n = sum([E_step_weights_list[n][k] * X.iloc[n][3] for n in range(N)]) / N_k
    mu = [tmp_x, tmp_y, tmp_m, tmp_n]

    # update variance
    ds = [np.array([[X.iloc[n][0], X.iloc[n][1], X.iloc[n][2], X.iloc[n][3]]]) - np.array([mu]) for n in range(N)]
    sigma = sum([E_step_weights_list[n][k] * ds[n].T.dot(ds[n]) for n in range(N)]) / N_k

    return multivariate_normal(mean=mu, cov=sigma), N_k / N


# Step 6 : Find the log_likelihood
def log_likelihood(distributions, X, M_Step_weights_list):  # Find the divergence of each EM step
    ans = 0.0
    for n in range(N):
        ws = [M_Step_weights_list[k] * distributions[k].pdf([X.iloc[n][0], X.iloc[n][1], X.iloc[n][2], X.iloc[n][3]])
              for k in range(K)]
        ans += math.log1p(sum(ws))

    return ans


# Step 7 : One step for testing
def one_step():
    # E Step
    E_step_weights_list = expectation_step(distributions, X_test, former_weights_list)
    # M Step
    for k in range(K):
        distributions[k], former_weights_list[k] = maximization_step(k, X_test, E_step_weights_list)

    return log_likelihood(distributions, X_test, former_weights_list)


# test
# log_likelihood_test = one_step()
# print(log_likelihood_test)


# Step 8 : Model for this dataset
# Step8
prev_log_likelihood = 0.0
for i in range(99):
    after_log_likelihood = one_step()

    if prev_log_likelihood / after_log_likelihood > 0.999:  
        break
    else:
        prev_log_likelihood = after_log_likelihood

print("repeat: ", i + 1)
for k in range(K):
    print("Gauss", k, ": ")
    print("  share: ", former_weights_list[k])
    print("  mean: ", distributions[k].mean)
    print("  cov: ", distributions[k].cov)


# Step 9 : Comparison between Original distribution and EM distribution
print(iris[iris['species'] == 'Iris-versicolor'].describe())
print(iris[iris['species'] == 'Iris-virginica'].describe())
print(iris[iris['species'] == 'Iris-setosa'].describe())
