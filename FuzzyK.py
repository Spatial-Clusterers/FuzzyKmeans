                                                                     
                                                                     
                                                                     
                                             
################################################################################
import numpy
from numpy import dot, array, sum, zeros, outer, any
from numpy import *
import os, sys
from numpy.random import random
import random
import os.path
import matplotlib.pyplot as plt
import time
sys.path.append('.')
################################################################################


################################################################################
# Fuzzy C-Means class
################################################################################
class FuzzyCMeans(object):
    '''
    Fuzzy C-Means convergence.

    Use this class to instantiate a fuzzy c-means object. The object must be
    given a training set and initial conditions. The training set is a list or
    an array of N-dimensional vectors; the initial conditions are a list of the
    initial membership values for every vector in the training set -- thus, the
    length of both lists must be the same. The number of columns in the initial
    conditions must be the same number of classes. That is, if you are, for
    example, classifying in ``C`` classes, then the initial conditions must have
    ``C`` columns.

    There are restrictions in the initial conditions: first, no column can be
    all zeros or all ones -- if that happened, then the class described by this
    column is unnecessary; second, the sum of the memberships of every example
    must be one -- that is, the sum of the membership in every column in each
    line must be one. This means that the initial condition is a perfect
    partition of ``C`` subsets.

    Notice, however, that *no checking* is done. If your algorithm seems to be
    behaving strangely, try to check these conditions.
    '''
    def __init__(self, training_set, initial_conditions, m=2.):
        '''
        Initializes the algorithm.

        :Parameters:
          training_set
            A list or array of vectors containing the data to be classified.
            Each of the vectors in this list *must* have the same dimension, or
            the algorithm won't behave correctly. Notice that each vector can be
            given as a tuple -- internally, everything is converted to arrays.
          initial_conditions
            A list or array of vectors containing the initial membership values
            associated to each example in the training set. Each column of this
            array contains the membership assigned to the corresponding class
            for that vector. Notice that each vector can be given as a tuple --
            internally, everything is converted to arrays.
          m
            This is the aggregation value. The bigger it is, the smoother will
            be the classification. Please, consult the bibliography about the
            subject. ``m`` must be bigger than 1. Its default value is 2
        '''
        self.__x = array(training_set)
        self.__mu = array(initial_conditions)
        self.m = m
        '''The fuzzyness coefficient. Must be bigger than 1, the closest it is
        to 1, the smoother the membership curves will be.'''
        self.__c = self.centers()

    def __getc(self):
        return self.__c
    def __setc(self, c):
        self.__c = array(c).reshape(self.__c.shape)
    c = property(__getc, __setc)
    '''A ``numpy`` array containing the centers of the classes in the algorithm.
    Each line represents a center, and the number of lines is the number of
    classes. This property is read and write, but care must be taken when
    setting new centers: if the dimensions are not exactly the same as given in
    the instantiation of the class (*ie*, *C* centers of dimension *N*, an
    exception will be raised.'''

    def __getmu(self):
        return self.__mu
    mu = property(__getmu, None)
    '''The membership values for every vector in the training set. This property
    is modified at each step of the execution of the algorithm. This property is
    not writable.'''

    def __getx(self):
        return self.__x
    x = property(__getx, None)
    '''The vectors in which the algorithm bases its convergence. This property
    is not writable.'''

    def centers(self):
        '''
        Given the present state of the algorithm, recalculates the centers, that
        is, the position of the vectors representing each of the classes. Notice
        that this method modifies the state of the algorithm if any change was
        made to any parameter. This method receives no arguments and will seldom
        be used externally. It can be useful if you want to step over the
        algorithm. *This method has a colateral effect!* If you use it, the
        ``c`` property (see above) will be modified.

        :Returns:
          A vector containing, in each line, the position of the centers of the
          algorithm.
        '''
        mm = self.__mu ** self.m
        c = dot(self.__x.T, mm) / sum(mm, axis=0)
        self.__c = c.T
        return self.__c

    def membership(self):
        '''
        Given the present state of the algorithm, recalculates the membership of
        each example on each class. That is, it modifies the initial conditions
        to represent an evolved state of the algorithm. Notice that this method
        modifies the state of the algorithm if any change was made to any
        parameter.

        :Returns:
          A vector containing, in each line, the membership of the corresponding
          example in each class.
        '''
        x = self.__x
        c = self.__c
        M, _ = x.shape
        C, _ = c.shape
        r = zeros((M, C))
        m1 = 1./(self.m-1.)
        for k in range(M):
            den = sum((x[k] - c)**2., axis=1)
            if any(den == 0):
                return self.__mu
            frac = outer(den, 1./den)**m1
            r[k, :] = 1. / sum(frac, axis=1)
        self.__mu = r
        return self.__mu

    def step(self):
        '''
        This method runs one step of the algorithm. It might be useful to track
        the changes in the parameters.

        :Returns:
          The norm of the change in the membership values of the examples. It
          can be used to track convergence and as an estimate of the error.
        '''
        old = self.__mu
        self.membership()
        self.centers()
        return sum(self.__mu - old)**2.

    def __call__(self, emax=1.e-10, imax=20):
        '''
        The ``__call__`` interface is used to run the algorithm until
        convergence is found.

        :Parameters:
          emax
            Specifies the maximum error admitted in the execution of the
            algorithm. It defaults to 1.e-10. The error is tracked according to
            the norm returned by the ``step()`` method.
          imax
            Specifies the maximum number of iterations admitted in the execution
            of the algorithm. It defaults to 20.

        :Returns:
          An array containing, at each line, the vectors representing the
          centers of the clustered regions.
        '''
        error = 1.
        i = 0
        while error > emax and i < imax:
            error = self.step()
            i = i + 1
        return self.c


"""Class to read dataset and initial conditions and converge the data using fuzzy c-means
------------DATA CONVERGING USING THE FUZZY C-MEANS ALGORITHM-------------
"""
class ReadDataConverge:

    def __init__(self, folder_name, file_name, cluster_num, samples):
        self.t0 = time.time()
        self.cluster_num = int(cluster_num)
        self.folder_name = folder_name
        self.file_name = file_name
        self.samples = int(samples)
        self.overall_data = []
        self.read()
        
        

    """Function to read the data from CSV and get the converged output with graph and CSV o/p"""    
    def read(self):
        data_set = []
        file_path = os.path.join(self.folder_name, self.file_name)
        f = open(file_path,'r')
        lines = f.readlines()
        for line in lines[:self.samples]:
            line_sp = line.split(',')
            try:
                data_set.append([float(line_sp[16]), float(line_sp[17])])
                self.overall_data.append([line_sp[0],float(line_sp[11]),float(line_sp[16]), float(line_sp[17])])
            except:
                continue
        self.converge(data_set, self.file_name)
        self.write_to_excel(self.file_name)
        return
                
    """Function to converge the data set by calling Fuzzy c-means object and write the converged data to the 
        CSV and plot the graph"""    
    def converge(self, data_set,  file1):
        #Membership val randomized
        #mu = random((len(data_set), 1))
        #mu = hstack((mu, 1.-mu))
        cluster_num = self.cluster_num
        if cluster_num==1:
            mu = [0.3,0.7]
            mu = [mu]*len(data_set)
        else:    
            mu = []
            for i in range(len(data_set)):
                nums = [random.uniform(0,1) for x in range(0,cluster_num)]
                sum = reduce(lambda x,y: x+y, nums)
                norm = [x/sum for x in nums]
                mu.append(norm)
        #mu = mu*(len(data_set))
        #print mu
        m = 2.0
        fcm = FuzzyCMeans(data_set, mu, m)
        converged = fcm(emax=0)
        mem_vals  = fcm.mu
        for i in range(len(self.overall_data)):
             self.overall_data[i].append(mem_vals[i][0])   
             self.overall_data[i].append(mem_vals[i][1])   
        for i in range(len(mu)):
            self.overall_data[i].append(mu[i])
        self.plot_graph(data_set,converged, file1)
        
        
    """Function to rearrange the converged numpy array object to linear array"""    
    def rearrange_x_y(self,converged):
        print ("........converged points......")
        print converged
        print "------------TIME TAKEN TO EXECUTE THE SCRIPT-----------",time.time() - self.t0, "seconds"
        x = []
        y = []
        for i in converged:
            x.append(i[0])
            y.append(i[1])
        return x,y

    def rearrange_set(self,data):
        x = []
        y = []
        for i in data:
            x.append(i[0])
            y.append(i[1])
        return x,y


    """Plot the converged data using matplotlib"""    
    def plot_graph(self, data_set, converged_vals, file1):
        x,y = self.rearrange_x_y(converged_vals)
        u,v = self.rearrange_set(data_set)
        fig = plt.figure()
        plt.plot(u, v, 'bo')    
        plt.plot(x, y, 'rp')
        
        line_map = []
        from math import hypot
        for i in range(len(u)):
            dist = []
            dist_dict = {}
            for j in range(len(x)):
                dist.append(hypot(u[i]-x[j], v[i]-y[j]))
                dist_dict[hypot(u[i]-x[j], v[i]-y[j])] = [x[j],y[j], u[i],v[i]]
            dist.sort()
            line_map.append(dist_dict[dist[0]])
            self.overall_data[i].append(dist_dict[dist[0]][0])
            self.overall_data[i].append(dist_dict[dist[0]][1])
        for each in line_map:
            plt.plot([each[0],each[2]],[each[1],each[3]])
        plt.xlabel('x-axis Lattitude')
        plt.ylabel('y axis Longitude')
        #plt.show()
        o_p = os.path.join(self.folder_name,'PNG_output')
        ou_file = os.path.join(o_p, file1[:-4]+'_output')
        fig.savefig(ou_file+'.png', dpi=fig.dpi)

        
    """Function to write the dataset to the CSV"""    
    def write_to_excel(self, filename):
        import csv
        ou_path = os.path.join(self.folder_name,'output')
        ou_file = os.path.join(ou_path, filename[:-4]+'_output'+'.csv')
        with open(ou_file, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                             quoting=csv.QUOTE_MINIMAL)
        
            spamwriter.writerow(['ID','HAIL-SIZE','LAT','LONG','MEM_VAL_X','MEM_VAL_Y','MEMBERSHIP VALUES OF DATA POINT','CONVERGED_TO_PT_X','CONVERGED_TO_PT_Y','FUZZY'])
            for i in range(len(self.overall_data)):
                a = self.overall_data[i][2]
                b = self.overall_data[i][3]
                c = self.overall_data[i][6]
                d = self.overall_data[i][7]
                if (str(a)[:4]==str(c)[:4]) and (str(b)[:4]==str(d)[:4]):
                    self.overall_data[i].append('YES')
                    spamwriter.writerow(self.overall_data[i])
                else:
                    self.overall_data[i].append('NO')
                    spamwriter.writerow(self.overall_data[i])
                    
    
        
    
################################################################################

if __name__=='__main__':
    folder_name = '../dataset/'
    file_name = 'noaa-hail-cleaned-index.csv'
    cluster_num = raw_input('enter the number of clusters:')
    samples = raw_input('enter the number of samples:')
    obj = ReadDataConverge(folder_name, file_name, cluster_num, samples)
    

