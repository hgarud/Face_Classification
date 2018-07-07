import numpy as np
import cv2
import os
from random import randint
from scipy.special import digamma
from scipy.special import gamma
from scipy.optimize import fmin

def tDist_pdf(x, mean, covar, nu, D):
    diff = x-mean
    dell1 = np.matmul(diff[None,:],np.linalg.inv(covar))
    dell = (np.matmul(dell1,diff[:,None])).flatten()
    # print(dell)
    numer = (gamma(0.5*(nu+D)))*((1 + (dell/nu)))**((-0.5)*(nu+D))
    denom = gamma(nu*0.5)#*((nu*3.1416)**(0.5*D))*((np.linalg.det(covar))**0.5)
    return numer/denom

input_dimension = 10
num_mixtures = 5                                # Num of mixtures for both classes
num_iterations = 10                              # Num iterations for E-M

# Load training samples
fddb = FDDB_Data()
pos_vector_space = fddb.load(train=True)[0]
neg_vector_space = fddb.load(train=True)[1]

# Mean init for positive images
pos_mean_vector_space = np.random.randint(0, 255, (num_mixtures, pos_vector_space.shape[1]))

# Mean init for negative images
neg_mean_vector_space = np.random.randint(0, 255, (num_mixtures, neg_vector_space.shape[1]))

# Positive Covariance matrix init
pos_co_var_space = np.empty((100,100,num_mixtures))
for i in range(num_mixtures):
    pos_co_var_space[:,:,i] = (i+1)*(np.eye(100))
pos_co_var_space = 5000*pos_co_var_space

# Negative Covariance matrix init
neg_co_var_space = np.empty((100,100,num_mixtures))
for i in range(num_mixtures):
    neg_co_var_space[:,:,i] = (i+1)*(np.eye(100))
neg_co_var_space = 5000*neg_co_var_space

# Init positive mixture weights
pos_mixt_weights = np.zeros(num_mixtures)
for i in range(num_mixtures):
    pos_mixt_weights[i] = 1/num_mixtures

# Init negative mixture weights
neg_mixt_weights = np.zeros(num_mixtures)
for i in range(num_mixtures):
    neg_mixt_weights[i] = 1/float(num_mixtures)

# Init positive Nu's
pos_nu_space = np.array([10,10,10,10,10])

# Init negative Nu's
neg_nu_space = np.array([10,10,10,10,10])

pos_responsibility = np.zeros((len(pos_vector_space),num_mixtures))
neg_responsibility = np.zeros((len(neg_vector_space),num_mixtures))

for iter in range(num_iterations):
    print("Iteration #: ", iter+1)
    for k in range(num_mixtures):
        pos_co_var_space[:,:,k] = (np.diag(np.diag(pos_co_var_space[:,:,k])+1))
        neg_co_var_space[:,:,k] = (np.diag(np.diag(neg_co_var_space[:,:,k])+1))

    pos_expect_hidden = np.zeros((pos_vector_space.shape[0], num_mixtures))
    pos_expect_log_hidden = np.zeros((pos_vector_space.shape[0], num_mixtures))
    neg_expect_hidden = np.zeros((neg_vector_space.shape[0], num_mixtures))
    neg_expect_log_hidden = np.zeros((neg_vector_space.shape[0], num_mixtures))

    # E-step
    print("**Expectation**")
    for k in range(num_mixtures):
        print("Cluster #: ",k+1)
        pos_numer = pos_nu_space[k] + (input_dimension**2)
        neg_numer = neg_nu_space[k] + (input_dimension**2)
        for i in range(pos_vector_space.shape[0]):

            pos_likelihood = tDist_pdf(pos_vector_space[i,:], pos_mean_vector_space[k,:], pos_co_var_space[:,:,k], pos_nu_space[k], input_dimension**2)
            neg_likelihood = tDist_pdf(neg_vector_space[i,:], neg_mean_vector_space[k,:], neg_co_var_space[:,:,k], neg_nu_space[k], input_dimension**2)

            pos_evidence = 0
            neg_evidence = 0
            for j in range(num_mixtures):
                pos_evidence = pos_evidence + pos_mixt_weights[j]*tDist_pdf(pos_vector_space[i,:], pos_mean_vector_space[j,:], pos_co_var_space[:,:,j], pos_nu_space[j], input_dimension**2)
                neg_evidence = neg_evidence + neg_mixt_weights[j]*tDist_pdf(neg_vector_space[i,:], neg_mean_vector_space[j,:], neg_co_var_space[:,:,j], neg_nu_space[j], input_dimension**2)

            pos_responsibility[i,k] = (pos_mixt_weights[k]*pos_likelihood)/pos_evidence
            neg_responsibility[i,k] = (neg_mixt_weights[k]*neg_likelihood)/neg_evidence

            diff = pos_vector_space[i,:] - pos_mean_vector_space[k,:]
            dell1 = (np.matmul(diff[None,:],np.linalg.inv(pos_co_var_space[:,:,k])))
            dell = (np.matmul(dell1,diff[:,None]))
            demon = pos_nu_space[k] + dell
            pos_expect_hidden[i,k] = pos_numer/demon
            pos_expect_log_hidden[i,k] = (digamma((0.5)*(pos_numer)) - np.log((0.5)*(demon)))

            diff = neg_vector_space[i,:] - neg_mean_vector_space[k,:]
            dell1 = np.matmul(diff[None,:],np.linalg.inv(neg_co_var_space[:,:,k]))
            dell = (np.matmul(dell1,diff[:,None]))
            demon = neg_nu_space[k] + dell
            neg_expect_hidden[i,k] = neg_numer/demon
            neg_expect_log_hidden[i,k] = (digamma((0.5)*(neg_numer)) - np.log((0.5)*(demon)))

    # M-step:
    print("**Maximization**")
    pos_mixt_weights = np.sum(pos_responsibility, axis = 0)/np.sum(np.sum(pos_responsibility, axis = 0))
    neg_mixt_weights = np.sum(neg_responsibility, axis = 0)/np.sum(np.sum(neg_responsibility, axis = 0))
    print("Updated Positive Weights after iteration {}: ".format(iter+1), pos_mixt_weights)
    print("Updated Negative Weights after iteration {}: ".format(iter+1), neg_mixt_weights)

    for k in range(num_mixtures):
        print("Positive Cluster #: ", k+1)
        num = np.zeros(pos_vector_space.shape[1])
        for i in range(len(pos_vector_space)):
            num = num + pos_responsibility[i,k]*pos_expect_hidden[i,k]*(pos_vector_space[i,:])
        pos_mean_vector_space[k,:] = num/np.sum(np.multiply(pos_responsibility[:,k], pos_expect_hidden[:,k]))

    for k in range(num_mixtures):
        numerator = 0
        for i in range(pos_vector_space.shape[0]):
            diff = pos_vector_space[i] - pos_mean_vector_space[k,:]
            mult = np.matmul(diff[:,None], diff[None,:])
            numerator = numerator + pos_responsibility[i,k]*pos_expect_hidden[i,k]*mult
        pos_co_var_space[:,:,k] = numerator/np.sum(np.multiply(pos_responsibility[:,k], pos_expect_hidden[:,k]))
        # cv2.imwrite("Pos_Covar_TDist.png", pos_co_var)
        pos_co_var_space[:,:,k] = np.diag(np.diag(pos_co_var_space[:,:,k]))

    for k in range(num_mixtures):
        def pos_nu_Cost_func(nu):
            return ((pos_vector_space.shape[0]*np.log(gamma(0.5*nu)))+(pos_vector_space.shape[0]*(0.5)*nu*np.log(0.5*nu))-(((0.5*nu)-1)*np.sum(pos_expect_log_hidden[:,k]))+((0.5*nu)*np.sum(pos_expect_hidden[:,k])))
        pos_nu_space[k] = fmin(pos_nu_Cost_func, pos_nu_space[k])[0]

    for k in range(num_mixtures):
        print("Negative Cluster #: ", k+1)
        num = np.zeros(neg_vector_space.shape[1])
        for i in range(len(neg_vector_space)):
            num = num + neg_responsibility[i,k]*neg_expect_hidden[i,k]*(neg_vector_space[i,:])
        neg_mean_vector_space[k,:] = num/np.sum(np.multiply(neg_responsibility[:,k], neg_expect_hidden[:,k]))

    for k in range(num_mixtures):
        numerator = 0
        for i in range(neg_vector_space.shape[0]):
            diff = neg_vector_space[i] - neg_mean_vector_space[k,:]
            mult = np.matmul(diff[:,None], diff[None,:])
            numerator = numerator + neg_responsibility[i,k]*neg_expect_hidden[i,k]*mult
        neg_co_var_space[:,:,k] = numerator/np.sum(np.multiply(neg_responsibility[:,k], neg_expect_hidden[:,k]))
        # cv2.imwrite("neg_Covar_TDist.png", neg_co_var)
        neg_co_var_space[:,:,k] = np.diag(np.diag(neg_co_var_space[:,:,k]))

    for k in range(num_mixtures):
        def neg_nu_Cost_func(nu):
            return ((neg_vector_space.shape[0]*np.log(gamma(0.5*nu)))+(neg_vector_space.shape[0]*(0.5)*nu*np.log(0.5*nu))-(((0.5*nu)-1)*np.sum(neg_expect_log_hidden[:,k]))+((0.5*nu)*np.sum(neg_expect_hidden[:,k])))
        neg_nu_space[k] = fmin(neg_nu_Cost_func, neg_nu_space[k])[0]

for i in range(num_mixtures)
    cv2.imwrite("Mean_Face_TDistMix_"+str(i+1)+".png", pos_mean_vector_space[0].reshape((10,10)).astype('uint8'))
    cv2.imwrite("Mean_NonFace_TDistMix_"+str(i+1)+".png", neg_mean_vector_space[0].reshape((10,10)).astype('uint8'))
    cv2.imwrite("Pos_Covar_TDistMix_"+str(i+1)+".png", pos_co_var_space[:,:,0])
    cv2.imwrite("NonPos_Covar_TDistMix_"+str(i+1)+".png", neg_co_var_space[:,:,0])

# Load testing samples
pos_test_vector_space = fddb.load(train=False)[0]
neg_test_vector_space = fddb.load(train=False)[1]

pos_posterior_p = np.zeros(pos_test_vector_space.shape[0])
pos_posterior_n = np.zeros(pos_test_vector_space.shape[0])
neg_posterior_p = np.zeros(neg_test_vector_space.shape[0])
neg_posterior_n = np.zeros(neg_test_vector_space.shape[0])
for i in range(pos_test_vector_space.shape[0]):
    pos_likelihood_p = 0
    pos_likelihood_n = 0
    neg_likelihood_p = 0
    neg_likelihood_n = 0
    for k in range(num_mixtures):
        pos_likelihood_p = pos_likelihood_p + pos_mixt_weights[k]*tDist_pdf(pos_test_vector_space[i], pos_mean_vector_space[k,:], pos_co_var_space[:,:,k], pos_nu_space[k], input_dimension**2)
        pos_likelihood_n = pos_likelihood_n + pos_mixt_weights[k]*tDist_pdf(neg_test_vector_space[i], pos_mean_vector_space[k,:], pos_co_var_space[:,:,k], pos_nu_space[k], input_dimension**2)
        neg_likelihood_p = neg_likelihood_p + neg_mixt_weights[k]*tDist_pdf(pos_test_vector_space[i], neg_mean_vector_space[k,:], neg_co_var_space[:,:,k], neg_nu_space[k], input_dimension**2)
        neg_likelihood_n = neg_likelihood_n + neg_mixt_weights[k]*tDist_pdf(neg_test_vector_space[i], neg_mean_vector_space[k,:], neg_co_var_space[:,:,k], neg_nu_space[k], input_dimension**2)
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
plot.title("ROC for Mixture of T-Distribution Classifier")
plot.ylabel("True Positives")
plot.xlabel("False Positives")
plot.show()
