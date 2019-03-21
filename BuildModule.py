from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn import neural_network

# 导入样本数据
wine = datasets.load_wine()
X = wine.data
Y = wine.target
print(X, "\n\n", Y)

print(np.shape(X), np.shape(Y))

# 把数据分成训练数据和测试数据
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(np.shape(X_train), np.shape(X_test))

# 训练模型1：利用逻辑回归模型
print("===================利用逻辑回归模型===================")
model = LogisticRegression().fit(X_train, Y_train)
print("训练数据上的准确率率为：%f" % (model.score(X_train, Y_train)))
print("测试数据上的准确率率为：%f" % (model.score(X_test, Y_test)))

# 训练模型2：利用支持向量机模型
print("===================利用支持向量机模型===================")
model = svm.SVC().fit(X_train, Y_train)
print("训练数据上的准确率率为：%f" % (model.score(X_train, Y_train)))
print("测试数据上的准确率率为：%f" % (model.score(X_test, Y_test)))

# 训练模型3：利用决策树模型
print("===================利用决策树模型===================")
model = tree.DecisionTreeClassifier().fit(X_train, Y_train)
print("训练数据上的准确率率为：%f" % (model.score(X_train, Y_train)))
print("测试数据上的准确率率为：%f" % (model.score(X_test, Y_test)))

# 训练模型4：利用神经网络模型
print("===================利用神经网络模型===================")
model = neural_network.MLPClassifier(alpha=1e-3).fit(X_train, Y_train)
print("训练数据上的准确率率为：%f" % (model.score(X_train, Y_train)))
print("测试数据上的准确率率为：%f" % (model.score(X_test, Y_test)))
