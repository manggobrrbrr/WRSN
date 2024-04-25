import numpy as np
import math
from scipy.spatial.distance import euclidean

def init_qtable(n_actions):
    return np.zeros((n_actions+1, n_actions+1), dtype=float)

def init_list_action(n_actions):
    list_action = []
    for i in range(int(math.sqrt(n_actions))):
        for j in range(int(math.sqrt(n_actions))):
            list_action.append([100 * (i + 1), 100 * (j + 1)])
    return list_action
    

class Q_learning:
    def __init__(self, mc, n_actions=81, epsilon=0.05, alpha=3e-4, gamma=0.5):
        self.n_actions = n_actions
        self.action_list = init_list_action(n_actions)
        self.q_table = init_qtable(n_actions)
        self.state = 0
        self.charging_time = [0.0 for _ in self.action_list]
        self.reward = [0.0 for _ in self.action_list]
        self.reward_max = 0.0
        self.epsilon = epsilon
        self.mc = mc
        self.alpha = alpha
        self.gamma = gamma

    def update(self):
        self.caculate_reward()
        
        # update q row i in q_table
        self.q_table[self.state] = (1 - self.alpha) * self.q_table[self.state] \
                                   + self.alpha * (self.reward + self.gamma * self.q_max(self.state))

        q_max_value = max(value for value in self.q_table[self.state])
        # choose next state of MC
        self.state = np.argmax(self.q_table[self.state])
        charging_time = self.get_charging_time(self.mc.net.listNodes[self.state])

        return self.action_list[self.state], charging_time, q_max_value


    def q_max(self, state):
        q_next_state = [max(row) for index, row in enumerate(self.q_table)]
        return np.asarray(q_next_state)
    
    def caculate_reward(self):
        energy_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        priority_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        target_monitoring_factor = np.array([0.0 for _ in self.action_list], dtype=float)
        for id, _ in enumerate(self.q_table):
            reward = self.reward_function(id)
            energy_factor[id] = reward[0]
            priority_factor[id] = reward[1]
            target_monitoring_factor[id] = reward[2]

        energy_factor = energy_factor / np.sum(energy_factor)
        priority_factor = priority_factor / np.sum(priority_factor)
        target_monitoring_factor = target_monitoring_factor / np.sum(target_monitoring_factor)

        self.reward = energy_factor + energy_factor + target_monitoring_factor

    def reward_function(self, state):
        connected_nodes = np.array([node for node in self.mc.net.listNodes 
                                    if euclidean(node.location, self.action_list[state]) <= self.mc.chargingRange])
        
        p = np.array([self.mc.alpha / (euclidean(self.action_list[state], node.location)+ self.mc.beta)**2
                         for node in connected_nodes])
        e = np.array([node for node in connected_nodes])
        E = np.array([node.energy for node in connected_nodes])
        w = np.array([len(np.array([candidate for candidate in node.neighbors if candidate.level < node.level]))+len(node.listTargets) 
                      for node in self.mc.net.listNodes])
        targets = []
        for target in self.mc.net.listTargets:
            for node in connected_nodes:
                if euclidean(target.location, node.location) <= node.send_range:
                    targets.append(target)
                    break
        t = len(targets)/ len(self.mc.net.listTargets)
        # energy factor
        energy_factor = np.sum(e*p/E)
        # priority factor
        priority_factor = np.sum(w*p)
        # target monitoring factor
        target_monitoring_factor = t
        
        return energy_factor, priority_factor, target_monitoring_factor  

def charge_time(self, mc, Sj):   

    Eth = 1
    theta = 1  
    Emax = Sj.capacity

    Es = Eth + theta * Emax

    # Calculate charging time for each sensor at the charging location
    optimal_charging_times = []
    for sensor_index, _ in enumerate(self.action_list):
        sensor_location = self.action_list[sensor_index]

        # Calculate per-second energy provided by the charging location
        distance_to_charging_location = euclidean(sensor_location, Sj.location)
        pc_jj = lambda d: self.mc.alpha / (distance_to_charging_location + mc.beta)**2
        tmove = distance_to_charging_location / mc.velocity
        pc_jk = mc.pc_jk(mc.location)
        # Calculate current energy consumption rate and cur_energy
        ej = mc.consumption_rate
        Ej_now = Sj.energy

        # Calculate remaining charging time
        tcu = mc.get_remaining_charging_time()

        charging_time = ((Es - Ej_now - ej*tcu) - (tmove * ej)) / (sum((pc_jk) + pc_jj - ej))
        optimal_charging_times.append(charging_time)

    # Sort the optimal charging times in descending order
    optimal_charging_times.sort(reverse=True)

    return optimal_charging_times
