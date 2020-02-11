import numpy as np
import pandas as pd
import math
import random
from functools import partial

def mean(x):
    return sum(x) / len(x)

# degree of (obs - mean)
def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

def sum_of_squares(x):
    return sum(x_i ** 2 for x_i in x)

def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def shape(M):
    num_rows = len(M)
    num_cols = len(M[0]) if M else 0
    return num_rows, num_cols

# jth value of each ith in i x j matrix
def get_column(M, j):
    return [M_i[j] for M_i in M]

# x_1 * b_1 + ... + x_n * b_n
def dot(x, b):
    return sum(x_i * b_i
               for x_i, b_i in zip(x, b))

# get means and standard deviations of each column
def scale(matrix):
    num_rows, num_cols = shape(matrix)
    means = [mean(get_column(matrix, j))
             for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(matrix, j))
              for j in range(num_cols)]
    return means, stdevs

# get matrix
def make_matrix(num_rows, num_cols, entry):
    return [[entry(i, j)
             for j in range(num_cols)]
            for i in range(num_rows)]

# each column has mean 0 and standard deviation 1
def rescale(matrix):
    means, stdevs = scale(matrix)
    def rescaled(i, j):
        if stdevs[j] > 0:
            return (matrix[i][j] - means[j]) / stdevs[j]
        else:
            return matrix[i][j]
    num_rows, num_cols = shape(matrix)
    return make_matrix(num_rows, num_cols, rescaled)

def split_data(data, prob):
    results = [], []
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results
 
def train_test_split(x, y, test_pct):
    data = zip(x, y)
    train, test = split_data(data, 1 - test_pct)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test



# subtracts corresponding values
def vector_subtract(x, b):
    return [x_i - b_i for x_i, b_i in zip(x, b)]

# scalar * vector
def scalar_multiply(s, v):
    return [s * v_i for v_i in v]

# minimize stochastic
def minimize_estimate(target, gradient, x, y, theta_0, alpha_0=0.01, safe=False):
    data = zip(x, y)
    
    # initial value
    theta = theta_0 
    alpha = alpha_0
    
    # minimum so far
    min_theta, min_value = None, float("inf")
    
    iterations_with_no_improvement = 0
    cnt_for_inf_loop = 0
    
    # limit 100
    while ((iterations_with_no_improvement < 100) & (cnt_for_inf_loop < 1e10)):
        cnt_for_inf_loop += 1
        if safe:
            if cnt_for_inf_loop > 1e5:
                print('too much iteration')
                break
        value = sum(target(x_i, y_i, theta) for x_i, y_i in zip(x, y))
 
        # if find new minimum, remeber it and go back to original step
        if value < min_value:
            min_theta, min_value = theta, value
            if cnt_for_inf_loop % 10 == 1:
                print('min_theta updates', min_theta)
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # not improving, so reduce step size
            iterations_with_no_improvement += 1
            if (iterations_with_no_improvement % 10 == 5):
                print("iterations_with_no_improvement is growing...", iterations_with_no_improvement)
            alpha *= 0.9

        # take a gradient step for each of the data points
        indexes = [i for i in range(len(x))];
        random.shuffle(indexes)
        for rand_i in indexes:
            gradient_i = gradient(x[rand_i], y[rand_i], theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
    return min_theta

# get function that for any input x returns -f(x)
def negate(f):
    return lambda *args, **kwargs: -f(*args, **kwargs)
 
# same when returns list of numbers
def negate_all(f):
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

def maximize_estimate(target, gradient, x, y, theta_0, alpha_0=0.01):
    return minimize_estimate(negate(target), negate_all(gradient), x, y, theta_0, alpha_0)



# logistic
# if x is big value, error occurs. create except syntax
def logistic(x):
    try:
        return 1.0 / (1 + math.exp(-x))
    except OverflowError:
        return 1e-7


# likehood
# get each log likelihood
def logistic_log_likelihood_i(x_i, y_i, beta):
    if y_i == 1:
        return math.log(logistic(dot(x_i, beta)))
    else:
        return math.log(1 - logistic(dot(x_i, beta)) + 1e-7)


# cumsum each log likelihood
def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
               for x_i, y_i in zip(x, y))


# get coefficient of 1~j
def logistic_log_partial_ij(x_i, y_i, beta, j):
    return (y_i - logistic(dot(x_i, beta))) * x_i[j]


# vector of 1~j partial derivative
def logistic_log_gradient_i(x_i, y_i, beta):
    return [logistic_log_partial_ij(x_i, y_i, beta, j) for j, _ in enumerate(beta)]


# vector sum about all data i
def logistic_log_gradient(x, y, beta):
    return np.sum(np.array([logistic_log_gradient_i(x_i, y_i, beta) for x_i, y_i in zip(x, y)]), axis=0)






random.seed(1030)

data_frame = pd.read_csv('data.csv', engine='python')

# split x,y in dataframe
x_frame = data_frame.loc[:, 'hour_max':'sewer_Near']
y_frame = data_frame['flooding']

# x, y to list
# 1 value is for intercept
x = [[1] + list(row[:8]) for row in x_frame.values.tolist()]
y = y_frame.values.tolist()

# rescale
x_rescale = rescale(x)

x_train, x_test, y_train, y_test = train_test_split(x_rescale, y, 0.3)

# max log likelihood of train data
pt = partial(logistic_log_likelihood, x_train, y_train)
gradient_pt = partial(logistic_log_gradient, x_train, y_train)

# set starting point
beta_0 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# gradient descent
beta_hat = maximize_estimate(logistic_log_likelihood_i,
                           logistic_log_gradient_i,
                           x_train, y_train, beta_0)

print('beta', beta_hat)
# flooding = -0.793 + 3.790hour_max - 1.719day_rain - 0.368slope - 0.325elevation - 0.216River_Near + 0.141rainT_Near - 0.366pump_Near + 0.081sewer_Near





# confusion matrix and accuracy
true_positives = false_positives = true_negatives = false_negatives = 0

for x_i, y_i in zip(x_test, y_test):
    predict = logistic(dot(beta_hat, x_i))

    if y_i == 1 and predict >= 0.5:  # true positives
        true_positives += 1
    elif y_i == 1:                   # false negatives
        false_negatives += 1
    elif predict >= 0.5:             # false positives
        false_positives += 1
    else:                            # true negatives
        true_negatives += 1

accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

print("accuracy", accuracy)

