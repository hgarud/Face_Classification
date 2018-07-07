import numpy as np
import cv2
import os
from random import randint
from scipy.special import digamma
from scipy.special import gamma
from scipy.optimize import fmin
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc

def tDist_pdf(x, mean, covar, nu, D):
    diff = x-mean
    dell1 = np.matmul(diff[None,:],np.linalg.inv(covar))
    dell = (np.matmul(dell1,diff[:,None]))
    numer = (gamma(0.5*(nu+D)))*((1 + (dell/nu)))**((-0.5)*(nu+D))
    denom = gamma(nu*0.5)#*((nu*3.1416)**(0.5*D))*((np.linalg.det(covar))**0.5)
    return numer/denom

# Load training samples
fddb = FDDB_Data()
pos_vector_space = fddb.load(train=True)[0]
neg_vector_space = fddb.load(train=True)[1]

# Mean init for positive images
pos_mean_vector = np.mean(pos_vector_space, axis = 0)

# Positive Covariance matrix init
pos_co_var = np.cov((pos_vector_space), rowvar = False)
pos_co_var = np.diag(np.diag(pos_co_var))

# Mean init for negative images
neg_mean_vector = np.mean(neg_vector_space, axis = 0)

# Negative Covariance matrix init
neg_co_var = np.cov((neg_vector_space), rowvar = False)
neg_co_var = np.diag(np.diag(neg_co_var)+1)

num_iterations = 10
pos_nu = 10
neg_nu = 10

for iter in range(num_iterations):
    print("Iteration #: ", iter+1)

    # E-step
    print("**Expectation**")
    pos_expect_hidden = []
    pos_expect_log_hidden = []
    numer = pos_nu + (input_dimension**2)
    for datum in pos_vector_space:
        diff = datum - pos_mean_vector
        dell1 = (np.matmul(diff[None,:],np.linalg.inv(pos_co_var)))
        dell = (np.matmul(dell1,diff[:,None]))
        demon = pos_nu + dell
        pos_expect_hidden = np.append(pos_expect_hidden, numer/demon)
        pos_expect_log_hidden = np.append(pos_expect_log_hidden, (digamma((0.5)*(numer)) - np.log((0.5)*(demon))))
    neg_expect_hidden = []
    neg_expect_log_hidden = []
    numer = neg_nu + (input_dimension)**2
    for datum in neg_vector_space:
        diff = datum - neg_mean_vector
        dell1 = np.matmul(diff[None,:],np.linalg.inv(neg_co_var))
        dell = (np.matmul(dell1,diff[:,None]))
        demon = neg_nu + dell
        neg_expect_hidden = np.append(neg_expect_hidden, numer/demon)
        neg_expect_log_hidden = np.append(neg_expect_log_hidden, (digamma((0.5)*(numer)) - np.log((0.5)*(demon))))

    # M-step
    print("**Maximization**")
    num = 0
    for i in range(pos_vector_space.shape[0]):
        num = num + np.dot(pos_expect_hidden[i], pos_vector_space[i,:])
    pos_mean_vector = num/sum(pos_expect_hidden)

    numerator = 0
    for i in range(pos_vector_space.shape[0]):
        diff = pos_vector_space[i] - pos_mean_vector
        mult = np.matmul(diff[:,None], diff[None,:])
        numerator = numerator + pos_expect_hidden[i]*mult
    pos_co_var = numerator/np.sum(pos_expect_hidden)
    cv2.imwrite("Pos_Covar_TDist.png", pos_co_var)
    pos_co_var = np.diag(np.diag(pos_co_var))

    def pos_nu_Cost_func(nu):
        return ((pos_vector_space.shape[0]*np.log(gamma(0.5*nu)))+(pos_vector_space.shape[0]*(0.5)*nu*np.log(0.5*nu))-(((0.5*nu)-1)*np.sum(pos_expect_log_hidden))+((0.5*nu)*np.sum(pos_expect_hidden)))
    pos_nu = fmin(pos_nu_Cost_func, pos_nu)[0]
    print("Positive Nu: ", pos_nu)
    print("Positive Mean: ", pos_mean_vector)

    num = 0
    for i in range(neg_vector_space.shape[0]):
        num = num + np.dot(neg_expect_hidden[i], neg_vector_space[i,:])
    neg_mean_vector = num/sum(neg_expect_hidden)

    numerator = 0
    for i in range(neg_vector_space.shape[0]):
        diff = neg_vector_space[i] - neg_mean_vector
        mult = np.matmul(diff[:,None], diff[None,:])
    numerator = numerator + neg_expect_hidden[i]*mult
    neg_co_var = numerator/np.sum(neg_expect_hidden)
    cv2.imwrite("Neg_Covar_TDist.png", neg_co_var)
    neg_co_var = np.diag(np.diag(neg_co_var)*1000+1)

    def neg_nu_Cost_func(nu):
        return ((neg_vector_space.shape[0]*np.log(gamma(0.5*nu)))+(neg_vector_space.shape[0]*(0.5*nu)*np.log(0.5*nu))-(((0.5*nu)-1)*np.sum(neg_expect_log_hidden))+((0.5*nu)*np.sum(neg_expect_hidden)))
    neg_nu = fmin(neg_nu_Cost_func, neg_nu)[0]
    print("Negative Nu: ", neg_nu)
    print("Negative Mean: ", neg_mean_vector)

# cv2.imshow('Mean Face', pos_mean_vector.reshape((10,10)).astype('uint8'))
cv2.imwrite("Mean_Face_TDist.png", pos_mean_vector.reshape((10,10)).astype('uint8'))
# cv2.imshow('Mean Non-Face', neg_mean_vector.reshape((10,10)).astype('uint8'))
cv2.imwrite("Mean_NonFace_TDist.png", neg_mean_vector.reshape((10,10)).astype('uint8'))

# Load testing samples
pos_test_vector_space = fddb.load(train=False)[0]
neg_test_vector_space = fddb.load(train=False)[1]

pos_likelihood_p = 0
pos_likelihood_n = 0
neg_likelihood_p = 0
neg_likelihood_n = 0

pos_posterior_p = np.zeros(pos_test_vector_space.shape[0])
pos_posterior_n = np.zeros(pos_test_vector_space.shape[0])
neg_posterior_p = np.zeros(neg_test_vector_space.shape[0])
neg_posterior_n = np.zeros(neg_test_vector_space.shape[0])
for i in range(pos_test_vector_space.shape[0]):
    pos_likelihood_p = (tDist_pdf(pos_test_vector_space[i], pos_mean_vector, pos_co_var, pos_nu, input_dimension**2))
    pos_likelihood_n = (tDist_pdf(neg_test_vector_space[i], pos_mean_vector, pos_co_var, pos_nu, input_dimension**2))
    neg_likelihood_p = (tDist_pdf(pos_test_vector_space[i], neg_mean_vector, neg_co_var, neg_nu, input_dimension**2))
    neg_likelihood_n = (tDist_pdf(neg_test_vector_space[i], neg_mean_vector, neg_co_var, neg_nu, input_dimension**2))

    pos_posterior_p[i] = pos_likelihood_p/(pos_likelihood_p+neg_likelihood_p)
    pos_posterior_n[i] = pos_likelihood_n/(pos_likelihood_n+neg_likelihood_n)
    neg_posterior_p[i] = neg_likelihood_p/(pos_likelihood_p+neg_likelihood_p)
    neg_posterior_n[i] = neg_likelihood_n/(pos_likelihood_n+neg_likelihood_n)

Posterior = np.append(pos_posterior_p, pos_posterior_n)
labels = np.append(np.ones(len(pos_posterior_p)), np.zeros(len(pos_posterior_p))   )

fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
plot.plot(fpr, tpr, color='darkorange')
print("False Positive Rate: {}, False Negative Rate: {}".format(fpr[int(fpr.shape[0]/2)], 1-tpr[int(fpr.shape[0]/2)]))
print("Misclassification Rate: {}".format(fpr[int(fpr.shape[0]/2)] + (1-tpr[int(fpr.shape[0]/2)])))
plot.xlim([-0.1,1.1])
plot.ylim([-0.1,1.1])
plot.title("ROC for T-Distribution Classifier")
plot.ylabel("True Positives")
plot.xlabel("False Positives")
plot.show()
