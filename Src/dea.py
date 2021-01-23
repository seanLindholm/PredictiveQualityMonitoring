import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.optimize import Bounds
from scipy.optimize import linprog
import pulp as p 






class DEA():

    def __init__(self,in_,out_):
        
        #Number of individual samples
        self.sample_num = np.arange(in_.shape[0])
        
        #input and output
        self.input = in_
        self.output = out_
        
        #Initilized input weitghs and output weigts
        self.input_w = np.ones(in_.shape[1])
        self.output_w = np.ones(out_.shape[1])
 
        #Weigthed out, input and efficency
        self.weigth_out = np.ones_like(self.sample_num).reshape(-1,1)
        self.weigth_in = np.ones_like(self.sample_num).reshape(-1,1)
        self.effeciency = np.zeros_like(self.sample_num).reshape(-1,1)
        
        #Constrained relationship
        self.working = np.zeros_like(self.sample_num).reshape(-1,1)


        
 
    def __calc_weigthed(self):
        #Update input weigth
        self.weigth_in = np.dot(self.input,self.input_w).reshape(-1,1)
        
        #Update output weigth
        self.weigth_out = np.dot(self.output,self.output_w).reshape(-1,1)
        
        #update constrained relationship
        self.working = self.weigth_out-self.weigth_in



    def fit(self):
        self.__calc_weigthed()
        self.effeciency = self.weigth_out/self.weigth_in
        


        

if __name__ == "__main__":
    # X = np.array([
    #     [18],
    #     [16],
    #     [17],
    #     [11]
    # ])
    # y = np.array([
    #     [125,50],
    #     [44,20],
    #     [80,55],
    #     [23,12]
    # ])
    # dea = DEA(X,y)
    # dea.fit()
    # print(dea.weigth_in)
    # print()
    
    # print(dea.weigth_out)
    # print()
    
    # print(dea.effeciency)
    # print()
    
    # print(dea.working)

     c = np.array([1,0,0])
     A_ineq = np.array([[44,20,16]])
     A_eq = np.array([[0,0,16]])
     B_ineq = np.array([[0]])
     B_eq = np.array([[1]])
     res_no_bounds = linprog(c, A_ub=A_ineq, b_ub=B_ineq, A_eq=A_eq, b_eq=B_eq, method='interior-point')
     print(res_no_bounds)

