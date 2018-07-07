import numpy as np
import cv2
import os
from random import randint
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc

def normpdf(x, mean, covar):
    diff = x - mean
    dell1 = np.matmul(diff[None,:],np.linalg.inv(covar))
    dell = (np.matmul(dell1,diff[:,None]))
    num = np.exp((-0.5)*dell)
    return num

# Load training samples
fddb = FDDB_Data()
pos_vector_space = fddb.load(train=True)[0]
neg_vector_space = fddb.load(train=True)[1]

num_mixtures = 5                                # Num of mixtures for both classes
num_iterations = 10                              # Num iterations for E-M

# Init positive mixture weights
pos_mixt_weights = np.zeros(num_mixtures)
for i in range(num_mixtures):
    pos_mixt_weights[i] = 1/num_mixtures

# Init negative mixture weights
neg_mixt_weights = np.zeros(num_mixtures)
for i in range(num_mixtures):
    neg_mixt_weights[i] = 1/float(num_mixtures)

# Init Mean for positive images
pos_mean_vector_space = np.random.randint(0, 255, (num_mixtures, pos_vector_space.shape[1]))

# Init Mean for negative images
neg_mean_vector_space = np.random.randint(0, 255, (num_mixtures, neg_vector_space.shape[1]))

# Init Positive Covariance matrix
pos_co_var_space = np.empty((100,100,num_mixtures))
for i in range(num_mixtures):
    pos_co_var_space[:,:,i] = (i+1)*(np.eye(100))
pos_co_var_space = 5000*pos_co_var_space

# Init Negative Covariance matrix
neg_co_var_space = np.empty((100,100,num_mixtures))
for i in range(num_mixtures):
    neg_co_var_space[:,:,i] = (i+1)*(np.eye(100))
neg_co_var_space = 5000*neg_co_var_space

pos_responsibility = np.zeros((len(pos_vector_space),num_mixtures))
neg_responsibility = np.zeros((len(neg_vector_space),num_mixtures))

for iter in range(num_iterations):
    print("Iteration #: ", iter+1)
    for k in range(num_mixtures):
        pos_co_var_space[:,:,k] = (np.diag(np.diag(pos_co_var_space[:,:,k])+1))
        neg_co_var_space[:,:,k] = (np.diag(np.diag(neg_co_var_space[:,:,k])+1))

    # E-step
    print("**Expectation**")
    for k in range(num_mixtures):
        print("Cluster #: ",k+1)
        for i in range(pos_vector_space.shape[0]):

            pos_likelihood = normpdf(pos_vector_space[i,:], pos_mean_vector_space[k,:], pos_co_var_space[:,:,k])
            neg_likelihood = normpdf(neg_vector_space[i,:], neg_mean_vector_space[k,:], neg_co_var_space[:,:,k])

            pos_evidence = 0
            neg_evidence = 0
            for j in range(num_mixtures):
                pos_evidence = pos_evidence + pos_mixt_weights[j]*normpdf(pos_vector_space[i,:], pos_mean_vector_space[j,:], pos_co_var_space[:,:,j])
                neg_evidence = neg_evidence + neg_mixt_weights[j]*normpdf(neg_vector_space[i,:], neg_mean_vector_space[j,:], neg_co_var_space[:,:,j])

            pos_responsibility[i,k] = (pos_mixt_weights[k]*pos_likelihood)/pos_evidence
            neg_responsibility[i,k] = (neg_mixt_weights[k]*neg_likelihood)/neg_evidence

    # M-step:
    print("**Maximization**")
    pos_mixt_weights = np.sum(pos_responsibility, axis = 0)/np.sum(np.sum(pos_responsibility, axis = 0))
    neg_mixt_weights = np.sum(neg_responsibility, axis = 0)/np.sum(np.sum(neg_responsibility, axis = 0))
    print("Updated Positive Weights after iteration {}: ".format(iter+1), pos_mixt_weights)
    print("Updated Negative Weights after iteration {}: ".format(iter+1), neg_mixt_weights)
    for k in range(num_mixtures):
        print("Cluster #: ", k+1)
        num = np.zeros(pos_vector_space.shape[1])
        for i in range(len(pos_vector_space)):
            num = num + pos_responsibility[i,k]*(pos_vector_space[i,:])
        pos_mean_vector_space[k,:] = num/np.sum(pos_responsibility[:,k])

        num = np.zeros(neg_vector_space.shape[1])
        for i in range(len(neg_vector_space)):
            num = num + neg_responsibility[i,k]*(neg_vector_space[i,:])
        neg_mean_vector_space[k,:] = num/np.sum(neg_responsibility[:,k])

        numer = np.zeros((pos_vector_space.shape[1], pos_vector_space.shape[1]))
        for i in range(len(pos_vector_space)):
            diff = (pos_vector_space[i,:] - pos_mean_vector_space[k,:])
            dell = np.matmul(diff[:,None],diff[None,:])
            numer = numer + pos_responsibility[i,k]*dell
        pos_co_var_space[:,:,k] = numer/np.sum(pos_responsibility[:,k])

        numer = np.zeros((neg_vector_space.shape[1], neg_vector_space.shape[1]))
        for i in range(len(neg_vector_space)):
            diff = (neg_vector_space[i,:] - neg_mean_vector_space[k,:])
            dell = np.matmul(diff[None,:],diff[:,None])
            numer = numer + neg_responsibility[i,k]*dell
        neg_co_var_space[:,:,k] = numer/np.sum(neg_responsibility[:,k])

# cv2.imshow("Mean Face", pos_mean_vector_space.reshape((10,10)).astype('uint8'))
for i in range(num_mixtures):
    cv2.imwrite("Mean_Face_GMM_"+str(i+1)+".png", pos_mean_vector_space[i].reshape((10,10)).astype('uint8'))
    cv2.imwrite("Mean_NonFace_GMM_"+str(i+1)+".png", neg_mean_vector_space[i].reshape((10,10)).astype('uint8'))
    cv2.imwrite("Pos_Covar_GMM_"+str(i+1)+".png", pos_co_var_space[:,:,i])
    cv2.imwrite("NonPos_Covar_GMM_"+str(i+1)+".png", neg_co_var_space[:,:,i])

for k in range(num_mixtures):
    pos_co_var_space[:,:,k] = (np.diag(np.diag(pos_co_var_space[:,:,k])+1))
    neg_co_var_space[:,:,k] = (np.diag(np.diag(neg_co_var_space[:,:,k])+1))

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
        pos_likelihood_p = pos_likelihood_p + pos_mixt_weights[k]*normpdf(pos_test_vector_space[i], pos_mean_vector_space[k,:], pos_co_var_space[:,:,k])
        pos_likelihood_n = pos_likelihood_n + pos_mixt_weights[k]*normpdf(neg_test_vector_space[i], pos_mean_vector_space[k,:], pos_co_var_space[:,:,k])
        neg_likelihood_p = neg_likelihood_p + neg_mixt_weights[k]*normpdf(pos_test_vector_space[i], neg_mean_vector_space[k,:], neg_co_var_space[:,:,k])
        neg_likelihood_n = neg_likelihood_n + neg_mixt_weights[k]*normpdf(neg_test_vector_space[i], neg_mean_vector_space[k,:], neg_co_var_space[:,:,k])
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
plot.title("ROC for Gaussian Mixture Classifier")
plot.ylabel("True Positives")
plot.xlabel("False Positives")
plot.show()
