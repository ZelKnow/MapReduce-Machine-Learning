# 机器学习算法的MapReduce实现

## Requirements
- mrjob 0.6.9 
- numpy 1.12.1
- scipy 0.19.1 

## 测试环境
MAacOS Mojave 10.14.5
Python 3.6.4
hadoop 2.8.0

## 项目简介

本项目使用python语言，基于hadoop平台，用MapReduce算法实现了三种机器学习算法，分别为：Naive Bayes、K Nearest Neighbor、K Means算法。

## 使用说明

code文件夹包含了代码文件，test_dataset包含了测试用的数据集，others文件夹包含了一个串行的Naive Bayes实现，用于比较MapReduce的性能。

### Naive Bayes

NaiveBayes.py: 朴素贝叶斯算法训练代码。选项continuous_features，代表了数据集中连续特征的编号。接收训练集作为输入，输出model。

NBPredictor.py: 朴素贝叶斯算法预测代码。选项continuous_features，代表了数据集中连续特征的编号；model，代表model的路径。接收测试集作为输入，输出准确率。

数据集格式参考./test_dataset/adult.data.csv，每行代表一个样例，每个样例的各个特征用“,”隔开，最后一列为该样例的类别。

```
python ./code/NaiveBayes.py --continuous_features 0,2,4,10,11,12 ./test_dataset/adult.data.csv > model.json
```

可以得到一个训练好的model：model.json。

```
python ./code/NBPredictor.py --continuous_features 0,2,4,10,11,12 --model model.json ./test_dataset/adult.test.csv
```

可以用来测试模型的性能。得到输出：

Accuary:83.14599840304649%

### K Nearest Neighbor

KNN.py: K近邻算法训练代码。接收训练集作为输入，输出model。

KNNPredictor.py: K近邻算法预测代码。选项model，代表model的路径。接收测试集作为输入，输出准确率。

数据集格式参考./test_dataset/haberman.data.csv，每行代表一个样例，每个样例的各个特征用“,”隔开。最后一列为该样例的类别。

```
python ./code/KNN.py ./test_dataset/haberman.data.csv > model.json
```

可以得到一个训练好的model：model.json

```
python ./code/KNNPredictor.py --model model.json -k 3 ./test_dataset/haberman.test.csv
```

可以用来测试模型的性能。得到输出：

Accuary:73.49397590361446%

### K Means

KMeans.py: K Means算法学习代码。选项centroids_input，代表输入中心点文件的路径。格式参考./test_dataset/Centroids.txt；centroids_output代表输出中心点文件的路径，默认为与输入路径相同；iterations，代表迭代次数，默认为10。接收训练集作为输入，输出训练集的分类结果和新的中心点坐标。

训练集的格式参考./test_dataset/e.txt。点的坐标，用“,”隔开。（可选：在每个样例最后加“\|”，后面跟该样例所属的类别编号）

```
python ./code/Kmeans.py --centroids_input ./test_dataset/Centroid.txt --centroids_output ./test_dataset/Centroid.txt  --iterations 10 ./test_dataset/e.txt
```


若提示无写入文件权限，请用sudo执行上述命令。

### 在hadoop平台进行测试

上述命令均为本机测试命令。若想在hadoop平台进行测试，只需要添加选项-r hadoop即可。例如：

```
python ./code/NaiveBayes.py -r hadoop --continuous_features 0,2,4,10,11,12 ./test_dataset/adult.data.csv > model.json
```

## License

This project is MIT licensed, as found in the LICENSE file.