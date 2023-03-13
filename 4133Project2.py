import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

X = np.genfromtxt('Handrawn_data.csv', delimiter='   ')
y = np.genfromtxt('Handrawn_label.csv', delimiter = '   ')
y = np.where(y == 6, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X.T,y.T,test_size=0.2,shuffle=True)
print(X_test.shape)
print(X_train.shape)

_, axes = plt.subplots(1, 10)
images = list(zip(X_train,y_train))
for ax, (image, label) in zip(axes[:], images[:10]):
    ax.set_axis_off()
    ax.imshow(image.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label, fontsize=4)


# SUPPORT VECTOR MACHINE
svmclf = svm.SVC(gamma = 0.001)
svmclf.fit(X_train, y_train)
svm_y_pred = svmclf.predict(X_test)

svmwrongidx = np.where(svm_y_pred != y_test)[0]


_, svmaxes = plt.subplots(1, len(svmwrongidx))
svmerrorimage = list(zip(X_test, svm_y_pred[svmwrongidx]))
for ax, (image, prediction) in zip(svmaxes[:], svmerrorimage[:]):
    ax.set_axis_off()
    ax.imshow(image.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('label(SVM): %i' % prediction, fontsize=4)


svm_disp = metrics.plot_confusion_matrix(svmclf, X_test, y_test)

print("SVM Accuracy:",metrics.accuracy_score(y_test, svm_y_pred))
print(metrics.classification_report(y_test, svm_y_pred))
print("SVM Confusion matrix:\n%s" % svm_disp.confusion_matrix)



#LOGISTIC REGREESSION
lrclf = LogisticRegression(max_iter=1000)
lrclf.fit(X_train, y_train)
lr_y_pred = lrclf.predict(X_test)

lrwrongidx = np.where(lr_y_pred != y_test)[0]

_, lraxes = plt.subplots(1, len(lrwrongidx))
lrerrorimage = list(zip(X_test, lr_y_pred[lrwrongidx]))
for ax, (image, prediction) in zip(lraxes[:], lrerrorimage[:]):
    ax.set_axis_off()
    ax.imshow(image.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Label(LR): %i' % prediction, fontsize=4)

lr_disp = metrics.plot_confusion_matrix(lrclf, X_test, y_test)

print("LR Accuracy:",metrics.accuracy_score(y_test, lr_y_pred))
print(metrics.classification_report(y_test, lr_y_pred))
print("LR Confusion matrix:\n%s" % lr_disp.confusion_matrix)


# NEURAL NETWORK
scaler = StandardScaler()
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

nnclf = MLPClassifier(hidden_layer_sizes=(15,15),max_iter=1000, alpha=0.0001, solver='lbfgs', activation='relu')

nnclf.fit(X_train,y_train)
nn_y_pred = nnclf.predict(X_test)

nnwrongidx = np.where(nn_y_pred != y_test)[0]
_, nnaxes = plt.subplots(1, len(nnwrongidx))

nnerrorimage = list(zip(X_test, nn_y_pred[nnwrongidx]))
for ax, (image, prediction) in zip(nnaxes[:], nnerrorimage[:]):
    ax.set_axis_off()
    ax.imshow(image.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Label(NN): %i' % prediction, fontsize=4)

nn_disp = metrics.plot_confusion_matrix(nnclf, X_test, y_test)

print("NN Accuracy:",metrics.accuracy_score(y_test, nn_y_pred))
print(metrics.classification_report(y_test, nn_y_pred))
print("NN Confusion matrix:\n%s" % nn_disp.confusion_matrix)

plt.show()



