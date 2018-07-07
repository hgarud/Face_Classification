import numpy as np
import cv2
import os
from random import randint
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc

def FA_pdf(x, mean, covar, phi):
    diff = x - mean
    mul1 = np.matmul(phi, np.ndarray.transpose(phi))
    dell1 = np.matmul(diff[None,:],np.linalg.inv(covar + mul1))
    dell = (np.matmul(dell1,diff[:,None]))
    num = np.exp((-0.5)*dell)
    return num

# Load training samples
fddb = FDDB_Data()
pos_vector_space = fddb.load(train=True)[0]
neg_vector_space = fddb.load(train=True)[1]

num_factors = 5                                 # Num of factors for both classes
num_iterations = 10                              # Num iterations for E-M

# Basis init for positive images
pos_basis_vectors = np.random.random_sample((pos_vector_space.shape[1], num_factors)) + 1

# Basis init for negative images
neg_basis_vectors = np.random.random_sample((pos_vector_space.shape[1], num_factors)) + 1

# Mean init for positive images
pos_mean_vector = np.mean(pos_vector_space, axis = 0)

# Mean init for negative images
neg_mean_vector = np.mean(neg_vector_space, axis = 0)

# Positive Covariance matrix init
pos_co_var = np.cov((pos_vector_space), rowvar = False)
# pos_co_var = np.diag(np.diag(pos_co_var))
print("Old Co Var: ", pos_co_var)
# Negative Covariance matrix init
neg_co_var = np.cov((neg_vector_space), rowvar = False)
# neg_co_var = np.diag(np.diag(neg_co_var))

pos_E_h = np.zeros((pos_vector_space.shape[0],num_factors))
pos_E_h_h = np.zeros((num_factors,num_factors,pos_vector_space.shape[0]))
neg_E_h = np.zeros((neg_vector_space.shape[0],num_factors))
neg_E_h_h = np.zeros((num_factors,num_factors,neg_vector_space.shape[0]))

for iter in range(num_iterations):
    print("Iteration #: ", iter+1)
    # pos_co_var = np.diag(np.diag(pos_co_var))
    # neg_co_var = np.diag(np.diag(neg_co_var))

    # E-step
    print("**Expectation**")
    for i in range(pos_vector_space.shape[0]):
        mul1 = np.matmul(pos_basis_vectors.transpose(), np.linalg.inv(pos_co_var))
        mul2 = np.matmul(mul1, pos_basis_vectors) + np.eye(num_factors)
        term1 = np.linalg.inv(mul2 + np.diag(np.diag(np.ones(num_factors))))
        mul3 = np.matmul(term1, pos_basis_vectors.transpose())
        mul4 = np.matmul(mul3, np.linalg.inv(pos_co_var))
        diff = pos_vector_space[i] - pos_mean_vector
        pos_E_h[i,:] = np.matmul(mul4, diff)
        pos_E_h_h[:,:,i] = term1 + np.matmul(pos_E_h[i,:], pos_E_h[i,:].transpose())

    for i in range(neg_vector_space.shape[0]):
        mul1 = np.matmul(neg_basis_vectors.transpose(), np.linalg.inv(neg_co_var))
        mul2 = np.matmul(mul1, neg_basis_vectors) + np.eye(num_factors)
        term1 = np.linalg.inv(mul2 + np.diag(np.diag(np.ones(num_factors))))
        mul3 = np.matmul(term1, neg_basis_vectors.transpose())
        mul4 = np.matmul(mul3, np.linalg.inv(neg_co_var))
        diff = neg_vector_space[i] - neg_mean_vector
        neg_E_h[i,:] = np.matmul(mul4, diff)
        neg_E_h_h[:,:,i] = term1 + np.matmul(neg_E_h[i,:], neg_E_h[i,:].transpose())

    # M-step
    print("**Maximization**")
    pos_mean_vector = np.sum(pos_vector_space, axis=0)/pos_vector_space.shape[0]
    neg_mean_vector = np.sum(neg_vector_space, axis=0)/neg_vector_space.shape[0]

    for i in range(pos_vector_space.shape[0]):
        diff = pos_vector_space[i] - pos_mean_vector
        term1 = np.matmul(diff[:,None], pos_E_h[i,:,None].transpose())

    pos_basis_vectors = np.matmul(term1, np.linalg.inv(np.sum(pos_E_h_h, axis=2)))

    temp = np.zeros((pos_vector_space.shape[1], pos_vector_space.shape[1]))
    for i in range(pos_vector_space.shape[0]):
        diff = pos_vector_space[i] - pos_mean_vector
        mul1 = np.matmul(diff[:,None], diff[None,:])
        mul2 = np.matmul(pos_basis_vectors, pos_E_h[i,:,None])
        mul3 = np.matmul(mul2,diff[None,:])
        temp = temp + np.diag(np.diag(mul1 - mul3))
    pos_co_var = temp/pos_vector_space.shape[0]

    for i in range(neg_vector_space.shape[0]):
        diff = neg_vector_space[i] - neg_mean_vector
        term1 = np.matmul(diff[:,None], neg_E_h[i,:,None].transpose())

    neg_basis_vectors = np.matmul(term1, np.linalg.inv(np.sum(neg_E_h_h, axis=2)))

    temp = np.zeros((neg_vector_space.shape[1], neg_vector_space.shape[1]))
    for i in range(neg_vector_space.shape[0]):
        diff = neg_vector_space[i] - neg_mean_vector
        mul1 = np.matmul(diff[:,None], diff[None,:])
        mul2 = np.matmul(neg_basis_vectors, neg_E_h[i,:,None])
        mul3 = np.matmul(mul2,diff[None,:])
        temp = temp + np.diag(np.diag(mul1 - mul3))
    neg_co_var = temp/neg_vector_space.shape[0]

# Load testing samples
pos_test_vector_space = fddb.load(train=False)[0]
neg_test_vector_space = fddb.load(train=False)[1]

pos_posterior_p = np.zeros(pos_test_vector_space.shape[0])
pos_posterior_n = np.zeros(pos_test_vector_space.shape[0])
neg_posterior_p = np.zeros(neg_test_vector_space.shape[0])
neg_posterior_n = np.zeros(neg_test_vector_space.shape[0])
for i in range(pos_test_vector_space.shape[0]):
    pos_likelihood_p = FA_pdf(pos_test_vector_space[i], pos_mean_vector, pos_co_var, pos_basis_vectors)
    pos_likelihood_n = FA_pdf(neg_test_vector_space[i], pos_mean_vector, pos_co_var, pos_basis_vectors)
    neg_likelihood_p = FA_pdf(pos_test_vector_space[i], neg_mean_vector, neg_co_var, neg_basis_vectors)
    neg_likelihood_n = FA_pdf(neg_test_vector_space[i], neg_mean_vector, neg_co_var, neg_basis_vectors)

    pos_posterior_p[i] = pos_likelihood_p/(pos_likelihood_p+neg_likelihood_p)
    pos_posterior_n[i] = pos_likelihood_n/(pos_likelihood_n+neg_likelihood_n)
    neg_posterior_p[i] = neg_likelihood_p/(pos_likelihood_p+neg_likelihood_p)
    neg_posterior_n[i] = neg_likelihood_n/(pos_likelihood_n+neg_likelihood_n)

Posterior = np.append(pos_posterior_p, pos_posterior_n)
labels = np.append(np.ones(len(pos_posterior_p)), np.zeros(len(pos_posterior_p))   )

cv2.imwrite("Mean_Face_FA.png", pos_mean_vector.reshape((10,10)).astype('uint8'))
cv2.imwrite("Mean_NonFaceFA.png", neg_mean_vector.reshape((10,10)).astype('uint8'))
cv2.imwrite("Pos_Covar_FA.png", pos_co_var)
cv2.imwrite("NonPos_Covar_FA.png", neg_co_var)


fpr, tpr, _ = roc_curve(labels, Posterior, pos_label=1)
plot.plot(fpr, tpr, color='darkorange')
print("False Positive Rate: {}, False Negative Rate: {}".format(fpr[int(fpr.shape[0]/2)], 1-tpr[int(fpr.shape[0]/2)]))
print("Misclassification Rate: {}".format(fpr[int(fpr.shape[0]/2)] + (1-tpr[int(fpr.shape[0]/2)])))
plot.xlim([-0.1,1.1])
plot.ylim([-0.1,1.1])
plot.title("ROC for Factor Analyzer Classifier")
plot.ylabel("True Positives")
plot.xlabel("False Positives")
plot.show()
