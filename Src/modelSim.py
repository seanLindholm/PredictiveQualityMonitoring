from constants_ import *

# Run -> 8 boards -> 16 arrays
array_cost = 65.83
num_arrays = 16
num_boards = 8


#We want a likelihood taht any given array will fail - from sampled data - tag 1/0




#Sample a number of good and bad (eg. over a cause of a week) taken from 2021-05-05 to 2021-04-28
#Here both good and bad has been set in distinct runs, to make the calculations easier in terms of what
#happens when different parts are discarded. Here 39 % are failed runs (83 good vs 33 bad) - sampled from functionstest
def likelihood_bad():
    good = [ 1 for _ in range(getData(model_approved).shape[0])]
    bad = [ 0 for _ in range(getData(model_failed).shape[0])]
    return len(bad)/(len(good)+len(bad))

# from the data
# 500 good
# 50 bad
# likelihood -> bad/(bad+good)


# Simulation
# Generate on array at a time - generate real tag from likelihood 
# Put array thourgh confusion matrix - Model tag (tag2) - need tag 1 
# Generating RUN (8 boards - 16 arrays) If tag2 is approved(1) then from (1) form the run - so Run will only consist of (1) on tag2 - populate all 16 arrays per board
# Now end of tag2 - only look at tag1
# Sample four arrays from the Runs (do it as they do in real life)
# "Send" it to functiontest -> check tag1 
# Simulate runs untill it reaches steady state -> performance indicator (cost of production/Number of all the sensors) 
 
# Set simulation 1 week -> then to 52 - for 1 configuration / confusion matrix21

class productionCostSimulation:
    def __init__(self,likelihood,confussion,seed = None):
        '''
        Assumbtions
        - When an array fail the whole run will be discarded (W)
        - When discarding from this sample, it is assumed that other arrays from that series would be good / you don't get 
          to test a bad array
        - All tags are distinct runs, so a good run vs bad runs.
        - A run is perfectly made, which for W would be 8 boards of 16 arrays that would be discarded if a bad array is tested doing functiontest
        - functiontest is 100 % accurate, meaning that if it produces an error, then it is the arrays fault and nothing else
        '''
        super().__init__()
        self.likelihood_bad = likelihood
        if (confussion is None):
            self.Baseline = True
            self.acc_a = 1
            self.acc_f = 0
            self.maxRuns = np.inf
        else:
            self.Baseline = False
            self.acc_a = confussion[0]
            self.acc_f = confussion[1]
            self.maxRuns = np.inf
        self.cost = 0
        self.generatedArrays = 0
        self.array_cost = 65.83
        self.num_arrays = 16
        self.num_boards = 8
        self.NumRunsInWeek = 26
        if seed is not None:
            np.random.seed(seed)

        #Analytical variables
        self.discardedRuns = 0
        self.unit_cost = 0
        self.faultyEndProducts = 0
        self.goodEndProducts = 0
        self.accpetedRuns = 0



        
    
    def generateArray(self):
        return 1 if np.random.rand() > self.likelihood_bad else 0

    def generateBoard(self):
        #This function needs to generate 16 arrays where the model tags them as good, meaning that 
        #Any 0 taged arrays will be discarded, and the cost of this, will be accumelated in the
        board = []
        while (len(board) < self.num_arrays):
            array = self.generateArray()
            self.generatedArrays += 1
            if(self.model(array)):
                board.append(array)
            else:
                self.cost += self.array_cost
        return board
    
    def generateRun(self):
        self.Run = []
        for _ in range(self.num_boards):
            self.Run.append(self.generateBoard())
    
    def __removeFirstOccurence(self,indx,val):
        self.Run[indx].remove(val)

    def SampleForFunctionTest(self):
        cartridge = []
        randomChoiceIndx = np.random.choice(np.arange(1,self.num_boards-1),2,replace=False)
        self.generateRun()
        tag1 = np.random.choice(self.Run[0],1)[0]; self.__removeFirstOccurence(0,tag1)
        cartridge.append(tag1)
        tag1 = np.random.choice(self.Run[self.num_boards-1],1)[0]; self.__removeFirstOccurence(self.num_boards-1,tag1)
        cartridge.append(tag1)
        tag1 = np.random.choice(self.Run[randomChoiceIndx[0]],1)[0]; self.__removeFirstOccurence(randomChoiceIndx[0],tag1)
        cartridge.append(tag1)
        tag1 = np.random.choice(self.Run[randomChoiceIndx[1]],1)[0]; self.__removeFirstOccurence(randomChoiceIndx[1],tag1)
        cartridge.append(tag1)
        return cartridge 

    def extractAnalyticsFromRun(self):
        for r in self.Run:
            max_score = len(r)
            self.faultyEndProducts += max_score - sum(r)
            self.goodEndProducts += sum(r)

    def plotAnalytics(self):
      

        fig, ax = plt.subplots(3,1)
        if (self.Baseline):
            fig.suptitle(f"The Baseline without any model to sort after.\nWith the likelihood that any produced array will fail in functiontest being {self.likelihood_bad*100:.2f}%")
        else:
            fig.suptitle(f"predictive accuracy good arrays {self.acc_a*100:.0f}% with predictive accuracy bad arrays {self.acc_f*100:.0f}%\nWith the likelihood that any produced array will fail in functiontest being {self.likelihood_bad*100:.2f}%")
        ax[0].plot(self.unit_cost_hist)
        ax[1].plot(self.accpetedRuns_hist,color='blue',label="Accepted Runs")
        ax[1].plot(self.discardedRuns_hist,color='red',label="Discarded Runs")

        ax[2].plot(self.goodEndProducts_hist,color='blue',label="Good products")
        ax[2].plot(self.faultyEndProducts_hist,color='red',label="Faulty products")

        ax[0].set_xlabel("simulation step")
        ax[0].set_ylabel("The unit cost of an array in production")
        
        ax[1].set_xlabel("simulation step")
        ax[1].set_ylabel("Number of Runs")
        ax[1].legend(loc="best")


        ax[2].set_xlabel("simulation step")
        ax[2].set_ylabel("Number of cartridges")
        ax[2].legend(loc="best")


      
        plt.show()

    def run(self,min_weeks_sim,plot=True):
        self.unit_cost_hist = []
        self.faultyEndProducts_hist = []
        self.goodEndProducts_hist = []
        self.discardedRuns_hist = []
        self.accpetedRuns_hist = []
        self.generatedArrays_hist = []
        counter=0
        unit_cost_old = -100
        while ((abs(unit_cost_old-self.unit_cost) > 10e-4 or counter <= min_weeks_sim) and self.maxRuns > counter):
            counter+=1
        #for _ in range(weeks):
            #if not all are improved throw out the run
            if(sum(self.SampleForFunctionTest()) < 4):
                self.cost += self.num_arrays*self.num_boards*self.array_cost
                self.discardedRuns += 1
            else:
                self.extractAnalyticsFromRun()
                self.accpetedRuns += 1
            unit_cost_old = self.unit_cost
            self.unit_cost = self.cost/self.generatedArrays
            
            self.unit_cost_hist.append(self.unit_cost)
            self.faultyEndProducts_hist.append(self.faultyEndProducts)
            self.goodEndProducts_hist.append(self.goodEndProducts)
            self.discardedRuns_hist.append(self.discardedRuns)
            self.accpetedRuns_hist.append(self.accpetedRuns)
            self.generatedArrays_hist.append(self.generatedArrays)
        if (plot):
            self.plotAnalytics()

        return self.unit_cost_hist,self.faultyEndProducts_hist,self.goodEndProducts_hist,self.generatedArrays_hist,self.accpetedRuns_hist, self.discardedRuns_hist
    

    def model(self,tag1):
        if(tag1):
            return 1 if np.random.rand() < self.acc_a else 0
        else:
            return 0 if np.random.rand() < self.acc_f else 1



def plot_(unit_,faulty_,good_,arrays_,accept_,discard_,names):
    plt.figure()
    base = [unit_[0] for _ in range(len(unit_[1:]))]
    plt.plot(names[1:],unit_[1:],label="Model unit cost")
    plt.plot(names[1:],base,'g--',label="The baseline")
    plt.title("The unit cost as a function of the model accuracy with baseline") 
    plt.xlabel("Model accuracy")
    plt.legend(loc="center right")
    plt.ylabel("The unit cost of an array in production")
    plt.xticks(rotation=90,fontsize=6)

    plt.figure()
    base = [accept_[0] for _ in range(len(unit_[1:]))]
    plt.plot(names[1:],base,'g--',label="The baseline - accepted")
    base = [discard_[0] for _ in range(len(unit_[1:]))]
    plt.plot(names[1:],base,'y--',label="The baseline - discarded")
    plt.plot(names[1:],accept_[1:],color='blue',label="Accepted Runs")
    plt.plot(names[1:],discard_[1:],color='red',label="Discarded Runs")
    plt.title(f"Percentage of accepted vs discarded runs after function test of the total number of runs produced.") 
    plt.xlabel("Model accuracy")
    plt.ylabel("Percentage of total number of runs")
    plt.legend(loc="center right")
    plt.xticks(rotation=90,fontsize=6)

    plt.figure()
    base = [good_[0] for _ in range(len(unit_[1:]))]
    plt.plot(names[1:],base,'g--',label="The baseline - good")
    base = [faulty_[0] for _ in range(len(unit_[1:]))]
    plt.plot(names[1:],base,'y--',label="The baseline - faulty")
    plt.plot(names[1:],good_[1:],color='blue',label="Good products")
    plt.plot(names[1:],faulty_[1:],color='red',label="Faulty products")
    plt.title(f"Percentage of good vs faulty products from the total number of produced products at steady state.") 
    plt.xlabel("Model accuracy")
    plt.ylabel("Percentage of total number of cartridges")
    plt.legend(loc="center right")
    plt.xticks(rotation=90,fontsize=6)

    # plt.figure()
    # base = [arrays_[0] for _ in range(len(unit_[1:]))]
    # plt.plot(names[1:],base,'g--',label="The baseline")
    # plt.plot(names[1:],arrays_[1:])
    # plt.title(f"Number of arrays generated before steady state") 
    # plt.xlabel("Model accuracy")
    # plt.ylabel("number of arrays")
    # plt.legend(loc="lower right")
    # plt.xticks(rotation=90,fontsize=6)


def main():
    #Get the actualt likelihood from real data (function test runs for like 1 week or 2)
    likelihood = likelihood_bad()
    sim = productionCostSimulation(likelihood,None,None)
    unit_ = []
    faulty_ = []
    good_ = []
    arrays_ = []
    accept_ = []
    discard_ = []
    names = []

    unit,fault,good,genereated,accept,discard = sim.run(min_weeks_sim=52,plot=False)

    names.append("Baseline")
    unit_.append(unit[-1])
    
    sum_product = good[-1]+fault[-1]
    sum_runs = accept[-1]+discard[-1]

    faulty_.append(fault[-1]/sum_product)
    good_.append(good[-1]/sum_product)
    arrays_.append(genereated[-1])
    accept_.append(accept[-1]/sum_runs)
    discard_.append(discard[-1]/sum_runs)



    accs = np.arange(0.1,1.1,0.1)
    for acc_good in accs:
        # sim = productionCostSimulation(likelihood,None,None)
        # unit_ = []; faulty_ = []; good_ = []; arrays_ = []; accept_ = [];  discard_ = [];  names = []
        # unit,fault,good,genereated,accept,discard = sim.run(min_weeks_sim=52,plot=False)

        # names.append("Baseline")
        # unit_.append(unit[-1])
        
        # sum_product = good[-1]+fault[-1]
        # sum_runs = accept[-1]+discard[-1]

        # faulty_.append(fault[-1]/sum_product); good_.append(good[-1]/sum_product)
        # arrays_.append(genereated[-1]);accept_.append(accept[-1]/sum_runs)
        # discard_.append(discard[-1]/sum_runs)

        for acc_bad in accs:
            sim = productionCostSimulation(likelihood,[acc_good,acc_bad],None)
            unit,fault,good,genereated,accept,discard = sim.run(min_weeks_sim=52,plot=False)

            names.append(f"A: {acc_good*100:.0f}%, F: {acc_bad*100:.0f}%")
            unit_.append(unit[-1])
            sum_product = good[-1]+fault[-1]
            sum_runs = accept[-1]+discard[-1]
            faulty_.append(fault[-1]/sum_product)
            good_.append(good[-1]/sum_product); arrays_.append(genereated[-1])
            accept_.append(accept[-1]/sum_runs); discard_.append(discard[-1]/sum_runs)
    plot_(unit_,faulty_,good_,arrays_,accept_,discard_,names)
      
    plt.show()
    



if __name__ == "__main__":
    # sim = productionCostSimulation(likelihood_bad(),[.90,.70],None)
    # sim.run(min_weeks_sim=0,plot=True)
    # sim = productionCostSimulation(likelihood_bad(),None,None)
    # sim.run(min_weeks_sim=100,plot=True)
    main()