from sklearn.datasets import load_iris # 导入鸢尾花数据集
iris = load_iris() # 载入数据集
print('iris数据集特征')
print(iris.data[:10])
print('iris数据集标签')
print(iris.target[:10])
from sklearn import svm # 导入 SVM 包
clf = svm.SVC()
clf.fit(iris.data[:120], iris.target[:120]) # 模型训练，取前五分之四作训练集
predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
predictions[:5]
from sklearn.metrics import accuracy_score # 导入准确率评价指标
print('Accuracy:%s'% accuracy_score(iris.target[120:], predictions))
from sklearn.datasets import load_iris # 导入鸢尾花数据集
from sklearn import svm # 导入 SVM 包
from sklearn.metrics import accuracy_score # 导入准确率评价指标
iris = load_iris() # 载入数据集
clf = svm.SVC(kernel='poly')
clf.fit(iris.data[:120], iris.target[:120]) # 模型训练，取前五分之四作训练集
predictions = clf.predict(iris.data[120:]) # 模型测试，取后五分之一作测试集
print('Accuracy:%s' % accuracy_score(iris.target[120:], predictions))