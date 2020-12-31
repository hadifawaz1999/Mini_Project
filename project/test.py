from model import MODEL
import numpy as np
from tslearn.metrics import cdist_dtw
from sklearn.metrics import accuracy_score as score
from sklearn.metrics.pairwise import euclidean_distances as ED

data_path = "/media/hadi/laban/data_sets/keras/mnist/"

xtrain = np.load(data_path+"x_train.npy")
# xtrain = np.asarray(xtrain, dtype=np.float64)
xtrain = xtrain.astype('float32')

xtrain = xtrain/255

ytrain = np.load(data_path+"y_train.npy")
ytrain = ytrain.astype('int')

xtest = np.load(data_path+"x_test.npy")
# xtest = np.asarray(xtest, dtype=np.float64)
xtest = xtest.astype('float32')

xtest = xtest/255

ytest = np.load(data_path+"y_test.npy")
ytest = ytest.astype('int')

# my_model = MODEL(xtrain=xtrain,
#                  ytrain=ytrain,
#                  xtest=xtest,
#                  ytest=ytest,
#                  load_model=False,
#                  draw_model=True,
#                  show_summary=True,
#                  show_details=True,
#                  save_loss=True,
#                  save_accuracy=True,
#                  learning_rate=0.01,
#                  batch_size=512,
#                  num_epochs=10)
# my_score = my_model.evaluate_score()

# print(my_score)

xtrain_copy = xtrain[0:200]
ytrain_copy = ytrain[0:200]
xtest_copy = xtest[0:100]
ytest_copy = ytest[0:100]

xtrain_copy.shape = (-1,28*28)
xtest_copy.shape = (-1,28*28)

ed_distances = ED(X=xtest_copy,Y=xtrain_copy)
print(ed_distances.shape)
indexes = np.argmin(ed_distances,axis=1)
ypred = ytrain_copy[indexes]
print(score(ypred,ytest_copy))