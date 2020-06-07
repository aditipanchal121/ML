import gym
import numpy as np

# 1. Load Environment
env = gym.make('FrozenLake8x8-v0')

#initialize q-table with 0s
Q = np.zeros([env.observation_space.n,env.action_space.n])

# env.obeservation.n, env.action_space.n gives number of states and action
#in env loaded

# 2. Parameters of Q-learning
class QLearning():
    def __init__(self, Q, eta, gma, epis):
        self.table = Q
        self.eta = eta
        self.gma = gma
        self.epis = epis

    def updateQ(self, s, a, r, s1):
            self.table[s,a] = self.table[s,a] + eta*(r + gma*np.max(self.table[s1,:]) - self.table[s,a])
        
    def choice(self, s, i):
        return np.argmax(self.table[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
    
    def run(self):
        rev_list = [] # rewards per episode calculate
        
        for i in range(epis):
        # Reset environment
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            while j < 99:
                env.render()
                j+=1
                # Choose action from Q table
                a = self.choice(s, i)
                #Get new state & reward from environment
                s1,r,d,_ = env.step(a)
                #Update Q-Table with new knowledge
                self.updateQ(s, a, r, s1)
                rAll += r
                s = s1
                if d == True:
                    break
            rev_list.append(rAll)
            env.render()
        
        
   
    
    
eta = .628
gma = .9
epis = 100
    
QLearningAlgo = QLearning(Q, eta, gma, epis)
QLearningAlgo.run()
#print("Reward Sum on all episodes " + str(sum(rev_list)/epis))
#print("Final Values Q-Table")
#print(QLearningAlgo.table)
