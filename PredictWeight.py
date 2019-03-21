import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.array([[150, 50], [152, 52], [160, 55], [164, 57], [165, 58], [168, 59], [170, 60],
                 [171, 61], [173, 61], [173, 61], [173, 63], [177, 64], [180, 67], [183, 70],
                 [184, 71]], np.int32)

# 提取出特征和标签
x = data[:, 0:-1]
y = data[:, 1]
print(x, "\n", y)
print(np.shape(x), np.shape(y))
plt.scatter(x, y)

# 通过线性回归来拟合给定的数据
model = LinearRegression().fit(x, y)
print(model.predict([[169]]))
print(model.score(x, y))
plt.plot(x, model.predict(x))
plt.show()
