# -*- coding: UTF-8 -*-
import numpy as np
import collections
from scipy import stats
from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.protocol import JSONProtocol

class KNNTrain(MRJob):
    '''
    KNN训练类。接收训练集输入，输出(类型, 特征集的列表)。
    '''
    OUTPUT_PROTOCOL = JSONProtocol

    def __init__(self, *args, **kwargs):
        super(KNNTrain, self).__init__(*args, **kwargs)
        
    def steps(self):
        return ([MRStep(mapper=self.mapper,reducer=self.reducer)])

    def mapper(self,_,line):
        '''
        Mapper函数，接收训练集的行，将行中的类型与特征集区分开，输出(类型，特征集)
        '''
        data = line.split(',')
        yield data[-1], data[:-1]

    def reducer(self, label, features):
        '''
        Reducer函数，接收Mapper输出，并将相同的类型的特征集连接起来，输出(类型，特征集的列表)
        '''
        features_list = []
        for feature in features:
            feature = [float(x) for x in feature]
            features_list.append(feature)
        yield label, features_list

if __name__ == '__main__':
    KNNTrain.run()
