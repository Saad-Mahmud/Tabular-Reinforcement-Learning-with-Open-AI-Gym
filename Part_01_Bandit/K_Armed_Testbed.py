import numpy as np 

class K_Armed(object):

    '''
        - K is the number of arm.
        - sampDist is the reward distrobution type of each arm. 
          Only implimented for Normal, Uniform and Bernoulli distribution.
    '''
    def __init__(self, K=10, sampDist = "Normal"):

        self._sampDist = ["Normal", "Bernoulli","Uniform"]
        assert(K>0)
        assert(sampDist in self._sampDist)
        self.sampDist = sampDist
        self.K = K
        self.bestArms = []
        self.reset()        
    
    def step(self, action):
        reward = 0
        if self.sampDist == "Normal":
            reward = np.random.normal(self.mu[action],self.std[action])
            
        elif self.sampDist == "Bernoulli":
            reward = np.random.binomial(1,self.theta[action])

        else:
            reward = np.random.uniform(self.low[action],self.high[action])
        return (reward,action in self.bestArms)
 


    def reset(self):
        if self.sampDist == "Normal":
            self.mu = np.random.normal(20,4,self.K)
            self.std = np.random.lognormal(1,0.5,self.K)
            MAX = np.max(self.mu)
            self.bestArms = [i for i in range(self.K) if self.mu[i]==MAX]

        elif self.sampDist == "Bernoulli":
            self.theta = np.random.beta(4,5,self.K)
            MAX = np.max(self.theta)
            self.bestArms = [i for i in range(self.K) if self.theta[i]==MAX]

        else:
            self.low = np.random.uniform(-5,5,self.K)
            self.high = np.random.uniform(20,30,self.K)
            mean = (self.low + self.high)/2
            MAX = np.max(mean)
            self.bestArms = [i for i in range(self.K) if mean[i]==MAX]

        

if __name__ == "__main__" :
    for i in range(100000):
        Bandit = K_Armed(10,"Normal")
        Bandit.step(0)

    
        
