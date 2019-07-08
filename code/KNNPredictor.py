# -*- coding: UTF-8 -*-
import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONValueProtocol
import KNN
import heapq
import os

current = os.getcwd()
true = 0
false = 0
class KNNTest(MRJob):

    '''
    KNN预测类。从文件接收测试集，并根据测试集的特征预测它们的类，并与真实的类进行比较，看是否预测成功
    '''

    def configure_args(self):
        '''
        输入args。包括model的地址（KNNTrain的输出），以及K的值。
        '''
        super(KNNTest,self).configure_args()

        #model的地址
        self.add_passthru_arg("--model",
                                type = str,
                                help = "the model path")
        #K的值
        self.add_passthru_arg("-k",
                                type = str,
                                help = "the value of K",
                                default = 3)
                                     
    def load_args(self,args):
        '''
        根据输入的args读取相应数据。
        '''
        super(KNNTest,self).load_args(args)

        #读取model
        if self.options.model is None:
            #没有输入model，报错
            self.option_parser.error("please type the path to the model.")
        else:
            #读取model
            self.model = {}
            job = KNN.KNNTrain()
            with open(current+'/'+self.options.model,encoding='utf-8') as src:
                for line in src:
                    #对model文件的每一行，读取相应的label和features，并存储到字典中。
                    label, features = job.parse_output_line(line.encode())
                    self.model[label] = features

        #读取K值。
        try:
            self.k = int(self.options.k)
        except:
            self.option_parser.error("K value must be integer.")

    def __init__(self, *args, **kwargs):
        super(KNNTest, self).__init__(*args, **kwargs)

    def steps(self):
        return ([MRStep(mapper=self.mapper,reducer=self.reducer)])

    def mapper(self,_,line):
        '''
        Mapper函数。接收测试集每一行，抽取出其的特征集，计算与其最接近的训练集中的K个点。
        判断K个点中最多的类别，并预测该测试样例对应的类为该类。然后与真实的类别比较，若
        预测正确，则输出(true, 1)，否则输出(false, 1)
        '''
        #抽取特征集和类别
        data = line.split(',')
        label = data[-1]
        features = [float(x) for x in data[:-1]]
        nearest = [] #距离最近的K个点
        count = {} #nearest中每个类别对应的数量

        #对训练集中的每个点，计算欧氏距离
        for cat in self.model:
            for point in self.model[cat]:
                #距离，乘以-1是因为之后要用到堆排序，需要从大到小排，但python的实现是最小堆，所以*(-1)
                dis = -1*np.linalg.norm(np.array(point)-np.array(features)) 
                #将距离、点、所属类别作一个元组，方便比较
                item = tuple([dis, point, cat])
                if(len(nearest)<self.k):
                    #若nearest长度小于k，直接append
                    nearest.append(item)
                    continue
                elif(len(nearest)==self.k):
                    #若nearest长度等于k，将nearest转化成堆
                    heapq.heapify(nearest)
                if(dis > nearest[0][0]):
                    #若新的点的距离小于nearest中距离最长的点，则将最长点弹出，新的点进入nearest
                    heapq.heapreplace(nearest,item)
        #计算nearest中各个点所属类别
        for i in range(len(nearest)):
            temp = heapq.heappop(nearest)
            if(temp[2] not in count):
                count[temp[2]] = 1
            else:
                count[temp[2]] += 1
        #计算最多的类别
        res = max(count, key=count.get)
        #若预测成功，输出true，否则false
        if(res==label):
            yield 'true', 1
        else:
            yield 'false', 1

    def reducer(self, label, num):
        '''
        Reducer函数，统计预测成功和预测失败的数量。
        '''
        if False: yield
        if(label=='true'):
            global true
            true = sum(num)
        else:
            global false
            false = sum(num)

if __name__ == '__main__':
    KNNTest.run()
    print("Accuary:"+str(true/(true+false)*100)+"%")