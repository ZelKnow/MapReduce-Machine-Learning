from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np
import mrjob

class MRKmeans(MRJob):
    '''
    Kmeans学习类。接收训练集、初始中心点输入，输出预测集，训练后的中心点。
    '''
    OUTPUT_PROTOCOL = mrjob.protocol.RawProtocol

    def __init__(self, *args, **kwargs):
        super(MRKmeans, self).__init__(*args, **kwargs)

    def configure_args(self):
        '''
        输入args。包括中心点输入文件、输出路径、迭代次数。
        '''
        super(MRKmeans, self).configure_args()
        self.add_file_arg('--centroids_input')
        self.add_file_arg('--centroids_output')    
        self.add_passthru_arg('--iterations', help='iterations', default=10, type=int)        

    def load_args(self,args):
        '''
        根据args读取数据。
        '''
        super(MRKmeans,self).load_args(args)
        #读取中心点输入文件
        if self.options.centroids_input is None:
            self.option_parser.error("please type the centroids input file.")
        else:
            self.infile = self.options.centroids_input
        #读取中心点输出路径。若无，则默认与输入路径相同，覆盖原文件。
        if self.options.centroids_output is None:
            self.outfile = self.infile
        else:
            self.outfile = self.options.centroids_output
        self.iterations = self.options.iterations

    def get_centroids(self):
        '''
        读取中心点
        '''
        Centroid = np.loadtxt(self.infile, delimiter = ',')
        return Centroid

    def write_centroids(self, Centroid):
        '''
        写中心点到文件中
        '''
        np.savetxt(self.outfile, Centroid[None], fmt = '%.5f',delimiter = ',')
 
    def relabel_data(self, _, line):
        '''
        Mapper函数。接收训练集的每一行，并计算该点与各个中心点的距离，计算出最近的中心点，
        以其作为其类型。输出(类型, 坐标)
        '''
        try:
            Coord, Cluster_ID = line.split('|')
        except:
            Coord = line
        Coord_arr = np.array(Coord.split(','), dtype = float)
        global Centroid
        Centroid = self.get_centroids()
        Centroid_arr = np.reshape(Centroid, (-1, len(Coord_arr)))
        global nclass
        global ndim
        nclass = Centroid_arr.shape[0]
        ndim = Centroid_arr.shape[1]
        Distance = ((Centroid_arr - Coord_arr)**2).sum(axis = 1)
        Cluster_ID = str(Distance.argmin() + 1)
        Coord_arr = Coord_arr.tolist()
        yield Cluster_ID, Coord_arr
    
    def node_combine(self, Cluster_ID, values):
        '''
        Combiner函数。接收Mapper的输出，输出(类型, (坐标点之和, 坐标点集))
        '''
        Coord_set = []
        Coord_sum = np.zeros(ndim)
        for Coord_arr in values:
            Coord_set.append(','.join(str(e) for e in Coord_arr))
            Coord_arr = np.array(Coord_arr, dtype = float)
            Coord_sum += Coord_arr
            Coord_sum = Coord_sum.tolist()
        yield Cluster_ID, (Coord_sum, Coord_set)
    
    def update_centroid(self, Cluster_ID, values):
        '''
        Reducer函数。接收Combiner输出，计算最终的坐标点的中心，并且将中心点输出到文件，
        然后再把坐标点加上预测的类型输出。
        '''
        final_Coord_set = []
        n = 0
        final_Coord_sum = np.zeros(ndim)
        for Coord_sum, Coord_set in values:
            final_Coord_set += Coord_set
            Coord_sum = np.array(Coord_sum, dtype = float)
            final_Coord_sum += Coord_sum
            n += 1
        
        new_Centroid = final_Coord_sum / n
        Centroid[ndim * (int(Cluster_ID) - 1) : ndim * int(Cluster_ID)] = new_Centroid
        if int(Cluster_ID) == nclass:
	        self.write_centroids(Centroid)

        for final_Coord in final_Coord_set:
            yield None, (final_Coord + '|' + Cluster_ID)

         
    def steps(self):   
        return [MRStep(mapper=self.relabel_data,
                       combiner=self.node_combine,
                       reducer=self.update_centroid)] * self.iterations #乘以迭代次数，就有多少个step
        
if __name__ == '__main__':
    MRKmeans.run() 