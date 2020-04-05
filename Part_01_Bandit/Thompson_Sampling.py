import numpy as np
from tensorboardX import SummaryWriter
from K_Armed_Testbed import K_Armed

 # Example only for bernoulli sampling distribution with beta prior. 
 # For non-conjugate prior use gibbs sampling.   
class TS_Agent(object):
    
    def __init__(self, env, maxItr = 10000):
        self.env = env 
        self.maxItr = maxItr
        self.avgReward = []
        self.bestAction = []
    
    def learn(self):
        Alpha = [np.random.randint(1,5) for i in range(self.env.K)]
        Beta = [np.random.randint(1,5) for i in range(self.env.K)]
        self.avgReward = []
        self.rightAction = []
        avgReward = 0
        bestAction = 0
        writer = SummaryWriter(comment="KABTS")
        for epi in range(1,self.maxItr+1):
            theta = np.array([np.random.beta(Alpha[i],Beta[i])for i in range(self.env.K)])
            arm = np.argmax(theta)
            reward,isBest = self.env.step(arm)
            avgReward =  avgReward + 1.0/epi*(reward-avgReward)
            if isBest:
                bestAction = bestAction+1 
            if reward == 0:
                Beta[arm]+=1
            else:
                Alpha[arm]+=1
        
            self.avgReward.append(avgReward)
            self.bestAction.append(bestAction/epi*100)
            if epi%int(self.maxItr/10.0) == 0:
                print("%d/%d: average_reward=%.3f, best_action=%.1f" % (\
                    epi,self.maxItr, self.avgReward[-1], self.bestAction[-1]))
            writer.add_scalar("average_reward", self.avgReward[-1],epi)
            writer.add_scalar("best_action", self.bestAction[-1],epi)
        print("Done: average_reward=%.3f, best_action=%.1f" % (\
             self.avgReward[-1], self.bestAction[-1]))
            
if __name__ == "__main__":
    env = K_Armed(10,"Bernoulli")
    agent = TS_Agent(env)
    agent.learn()
    