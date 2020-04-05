import numpy as np
from tensorboardX import SummaryWriter
from K_Armed_Testbed import K_Armed

    
class UCB_Agent(object):
    
    def __init__(self, env, maxItr = 10000, c = 2):
        self.env = env 
        self.maxItr = maxItr
        self.c = c
        self.avgReward = []
        self.bestAction = []
    
    def learn(self):
        Q = [0 for i in range(self.env.K)]
        N = [0 for i in range(self.env.K)]
        self.avgReward = []
        self.rightAction = []
        avgReward = 0
        bestAction = 0
        writer = SummaryWriter(comment="KABUCB_{}".format(self.c))
        for epi in range(1,self.maxItr+1):
            con = (self.c**2)*np.log(epi)
            arm = np.argmax(Q+np.sqrt(np.array([con])/N))
            reward,isBest = self.env.step(arm)
            N[arm] = N[arm]+1
            Q[arm] = Q[arm]+1.0/N[arm]*(reward-Q[arm])
            avgReward =  avgReward + 1.0/epi*(reward-avgReward)
            if isBest:
                bestAction = bestAction+1 
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
    env = K_Armed(10)
    es = [0.5,1,2]
    for i in es:
        agent = UCB_Agent(env,c=i)
        agent.learn()
    