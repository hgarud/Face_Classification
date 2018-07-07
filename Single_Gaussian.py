import numpy as np
import cv2
import os
import matplotlib.pyplot as plot
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from FDDB_data import FDDB_Data

def normpdf(x, mean, covar):
    diff = x - mean
    dell1 = np.matmul(diff[None,:],np.linalg.inv(covar)).flatten()
    dell = (np.matmul(dell1,diff[:,None])).flatten()
    num = np.exp((-0.5)*dell)
    return num

# Load training samples
fddb = FDDB_Data()
pos_vector_space = fddb.load(train=True)[0]
neg_vector_space = fddb.load(train=True)[1]

pos_mean_vector = np.mean(pos_vector_space, axis = 0)
cv2.imshow("Mean Face", pos_mean_vector.reshape((10,10)).astype('uint8'))
cv2.imwrite("Mean_Face_SGaussian.png", pos_mean_vector.reshape((10,10)).astype('uint8'))

pos_co_var = np.cov((pos_vector_space), rowvar = False)
cv2.imshow("Positive Covariance Matrix", pos_co_var)
cv2.imwrite("Pos_Covar_SGaussian.png", pos_co_var)
pos_co_var = np.diag(np.diag(pos_co_var))
# cv2.imshow("Positive Covariance Matrix", pos_co_var)

neg_mean_vector = np.mean(neg_vector_space, axis = 0)
cv2.imshow("Mean Non-Face", neg_mean_vector.reshape((10,10)).astype('uint8'))
cv2.imwrite("Mean_NonFace_SGaussian.png", neg_mean_vector.reshape((10,10)).astype('uint8'))

neg_co_var = np.cov((neg_vector_space), rowvar = False)
cv2.imshow("Negative Covariance Matrix", neg_co_var)
cv2.imwrite("Neg_Covar_SGaussian.png", neg_co_var)
neg_co_var = np.diag((np.diag(neg_co_var)+1)*5000)
# cv2.imshow("Negative Covariance Matrix", neg_co_var)

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
    pos_likelihood_p = normpdf(pos_test_vector_space[i], pos_mean_vector, pos_co_var)
    pos_likelihood_n = normpdf(neg_test_vector_space[i], pos_mean_vector, pos_co_var)
    neg_likelihood_p = normpdf(pos_test_vector_space[i], neg_mean_vector, neg_co_var)
    neg_likelihood_n = normpdf(neg_test_vector_space[i], neg_mean_vector, neg_co_var)

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
plot.title("ROC for single Gaussian Classifier")
plot.ylabel("True Positives")
plot.xlabel("False Positives")
plot.show()
cv2.waitKey(0)
