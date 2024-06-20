from controller import Robot, Motor
import numpy as np
import math
import copy
import random

from collections import namedtuple, deque
from itertools import count
from math import acos
from math import atan2
from math import sin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# DQN PART====================================
# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# Class to store memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
        
# DQN model        
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 156)
        self.layer2 = nn.Linear(156, 156)
        self.layer3 = nn.Linear(156, 78)
        self.layer4 = nn.Linear(78, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)



# DQN INITIALIZATION PART ====================================================
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 6
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-3

action_space = [0, 1, 2]
n_actions = len(action_space)

# Will have to change when we add more values to the observation/state
n_observations = 9

# policy_net = DQN(n_observations, n_actions).to(device)
# target_net = DQN(n_observations, n_actions).to(device)
# target_net.load_state_dict(policy_net.state_dict())
policy_net = torch.load("policy0.pth")
target_net = torch.load("target0.pth")

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(40000)


steps_done = 0
# ==============================================================        



        
# Functions
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[ random.sample(action_space, 1)[0] ]], device=device, dtype=torch.long)        

# print(select_action([0, 0, 0]).item())



episode_durations = []



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
# print(torch.tensor([[random.sample([0,0,0],1)[0]]] , device = device, dtype=torch.long ) )
#==============================================

     
     
# Class represents the environment and the state of the robot in it
# Works similar to the gymnazium env
class Environment():
    def __init__(self, robot, ds, gps, compass, bumpers, mtSystem):
        self.robot = robot
        self.ds_sensors = ds
        self.gps = gps
        self.compass = compass
        self.bumpers = bumpers
        self.current_position = self.get_current_position()
        self.prev_position = self.current_position
        print("current")
        print(self.current_position)
        self.mtSystem = mtSystem
        
        self.action_history = deque([])
        
        # FIRST TESTING
        self.finish_coords = [1.03487, 0.822464]
    
    
    def step(self, action):
        self.mtSystem.apply_action(action)
        
        # Trying to make him not abuse turning
        if(action==1 or action==2):
            self.action_history.append(1)
        else:
            self.action_history.append(0)
        
        if(len(self.action_history)>10):
            self.action_history.popleft()
            
            
        self.robot.step(timestep)
        self.actualize_position()
        state = self.get_state()
        reward = self.get_reward()
        finished = self.is_at_finish()
        return [state, reward, finished]
        
    def reset(self):
        self.prev_position = self.current_position
        return self.get_state()
        
        
    def is_at_finish(self):
        return self.get_distance_to_finish(self.current_position) < 0.1
    
    # use get reward only with actualized positions
    # rn reward changes just when we get closer to the target
    def get_reward(self):
        if(self.is_at_finish()):
            return 400
        
        # bump check
        if(self.is_collision()):
            return -60
    
        prev_distance = self.get_distance_to_finish(self.prev_position)
        
        current_distance = self.get_distance_to_finish(self.current_position)
        # print("Distance")
        # print(current_distance)
        
        turn_coef = 1.0
        
        # if(sum(list(self.action_history)) > 7):
            # turn_coef = sum(list(self.action_history))/len(self.action_history)
        
        # Distance to target code
        reward = 10*(prev_distance-current_distance)
        
        # Angular deviation code
        rad_deviation = self.get_deviation()
        reward = (1/abs(rad_deviation)) * 0.1 
        
        # Distance sensors code
        distances = [sensor.getValue()/400.0 for sensor in self.ds_sensors]
        d_effects = [d/2.5 for d in distances]
        
        for i in range(len(d_effects)):
            if d_effects[i] < 0.98:
                if i in [0,4]:
                    d_effects[i] *= 0.9 # normal effect
                if i in [1,3]:
                    d_effects[i] *= 0.8 # bigger effect
                if i in [0]:
                    d_effects[i] *= 0.7
                
                
        # Rules
        if abs(reward) < 0.03:
            reward = 0.0
        else:
            if reward > 0.7:
                if(sum(list(self.action_history))/len(self.action_history) > 0.4):
                    turn_coef = (len(self.action_history)/2.5) / sum(list(self.action_history))
                    reward = 0.7 * turn_coef
                else:
                    reward = 0.7
                
                
        for e in d_effects:
            reward *= e
                
        # if min(obstacle_distances) < 1.2:
            # reward *= (min(obstacle_distances))/5.0
        
        return reward
      
         
        
    def get_distance_to_finish(self, position):
        return self.get_distance(self.finish_coords, position)
        
    def actualize_position(self):
        self.prev_position = copy.deepcopy(self.current_position)
        self.current_position = self.get_current_position()
        
    
    def get_distance(self, pos1, pos2):
        vector = [pos1[0]-pos2[0], pos1[1]-pos2[1]]
        distance =  math.sqrt(vector[0]**2 + vector[1]**2)
        return distance
        
    def get_current_position(self):
        return self.gps.getValues()[:2]
    
    # will return only X, Y and front ds sensor for now    
    def get_state(self):
        x_coord, y_coord, z_coord = self.gps.getValues()
        rad_deviation =self.get_deviation()
        #
        # Obstacle distances
        distances = [sensor.getValue()/400.0 for sensor in self.ds_sensors]
        # md_to_obstacle = self.ds_sensors[2].getValue()/400.0
        # rd_to_obstacle = self.ds_sensors[0].getValue()/400.0
        # ld_to_obstacle = self.ds_sensors[1].getValue()/400.0
        
        # Bump check
        collision = int(self.is_collision())
        
        # self.ds_sensors[2].getValue()/40.0
        return [x_coord, y_coord, rad_deviation, distances[0], distances[1], distances[2], distances[3], distances[4], collision]
   
    def get_deviation(self):

        # Get values from compass object 
        robot_vector = self.compass.getValues()
        robot_vector[0] = -robot_vector[0]
        # print("robot_vector: {0}".format(robot_vector))
        
        # Get finish direction vector
        finish_vector = [self.finish_coords[0]-self.get_current_position()[0], self.finish_coords[1]-self.get_current_position()[1]]
        # print("Finish: {0}".format(finish_vector))
        
        
        scalar = robot_vector[0]*finish_vector[0] + robot_vector[1]*finish_vector[1]
        robot_module =  math.sqrt(robot_vector[0]**2 + robot_vector[1]**2)
        finish_module =  math.sqrt(finish_vector[0]**2 + finish_vector[1]**2)
        
        cosA = (scalar)/(robot_module * finish_module)
        
        rad_deviation = acos(cosA)
        
        cross_product = robot_vector[0]*finish_vector[1] - robot_vector[1]*finish_vector[0]
        if(cross_product > 0):
            rad_deviation*=-1
        return rad_deviation

    def is_collision(self):
        for bumper in self.bumpers:
            if bool(bumper.getValue()):
                return True           
        return False



# Class created for much easier robot motor system manipulation.
# Simple methods from class may be used as an action space with Q table?
class MotorSystem():
    def __init__(self, left_motor, right_motor, rear_motor, front_motor, max_speed):
        self.left_motor = left_motor
        self.right_motor = right_motor
        self.max_speed = max_speed
        self.rear_motor = rear_motor
        self.front_motor = front_motor
        
    def apply_action(self, action):
        if(action == "Stop"):
            self.stop()
        elif(action == "Forward" or action == 0):
            self.forward()
        elif(action == "Left" or action == 1):
            self.turn_left()
        elif(action == "Right" or action == 2):
            self.turn_right()
        else:
            print("Given motor action is not found.")
        
    def stop(self):
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.rear_motor.setVelocity(0)
        self.front_motor.setVelocity(0)
    
    def forward(self):
        self.left_motor.setVelocity(self.max_speed*1.0)
        self.right_motor.setVelocity(self.max_speed*1.0)
        self.rear_motor.setVelocity(self.max_speed*1.0)
        self.front_motor.setVelocity(self.max_speed*1.0)
        
    def turn_left(self):
        self.left_motor.setVelocity(self.max_speed*(-1.0))
        self.right_motor.setVelocity(self.max_speed*(1.0))
        self.rear_motor.setVelocity(self.max_speed*0.0)
        self.front_motor.setVelocity(self.max_speed*0.0)
        
    def turn_right(self):
        self.left_motor.setVelocity(self.max_speed*(0.7))
        self.right_motor.setVelocity(self.max_speed*(-0.7))
        self.rear_motor.setVelocity(self.max_speed*0.0)
        self.front_motor.setVelocity(self.max_speed*0.0)
 





# MAIN CODE

# create the Robot instance
robot = Robot()
# run_robot(robot)
# defenition of the time step
timestep = 32

# defenition of the max speed
max_speed = 10

# initialize motors
speed = 0.0  # [rad/s]
left_motor = robot.getDevice('motor_1')
right_motor = robot.getDevice('motor_2')
rear_motor = robot.getDevice('motor_3')
front_motor = robot.getDevice('motor_4')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
rear_motor.setPosition(float('inf'))
front_motor.setPosition(float('inf'))

left_motor.setVelocity(max_speed)
right_motor.setVelocity(max_speed)
rear_motor.setVelocity(max_speed)
front_motor.setVelocity(max_speed)

# initialize GPS module
gps = robot.getDevice("gps")

# initialize distance dssensors
# 40 : 0.01m
ds = []
dsNames = ['d_far_left', 'd_left', 'd_front', 'd_right', 'd_far_right']
for i in range(5):
    ds.append(robot.getDevice(dsNames[i]))
    ds[i].enable(timestep)

# initialize the lidar
# lidar = robot.getDevice('lidar')
# lidar.enable(timestep)
# lidar.enablePointCloud()

mtSystem = MotorSystem(left_motor, right_motor, rear_motor, front_motor, max_speed)

moving_stage = 0

# enable the GPS
gps.enable(timestep)

initializedEnv = 0

# initialize the Compass
compass = robot.getDevice('compass')
# enable the Compass
compass.enable(timestep)

# initialize the TouchSensor
bumpers = [robot.getDevice('bumper_left'), robot.getDevice('bumper_front'), robot.getDevice('bumper_right')]
# bumper = robot.getDevice('bumper')
# enable the TouchSensor
for bumper in bumpers:
    bumper.enable(timestep)

finish_positions = [[-0.52, 0.89], [-0.89, -0.47], [0.78, 1.06], [0.43, -1.15], [0.17, -0.06]]

# DQN TRAINING ===========================================================

if (robot.step(timestep) != -1 and initializedEnv != 1):
    env = Environment(robot, ds, gps, compass, bumpers, mtSystem)
    initializedEnv = 1
    # mtSystem.turn_left()
    # for i in range(100):
        # mtSystem.turn_left()
        # robot.step(timestep)
    
        
    # mtSystem.stop()
    # robot.step(timestep)

    # if torch.cuda.is_available():
        # num_episodes = 6
    # else:
        # num_episodes = 6 
    # while (robot.step(timestep) != -1):
        # print("Deviation from the goal (rad):", env.get_deviation())
        
    num_episodes = 600
    num_iters = 2600
    
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        env.finish_coords = finish_positions[i_episode % 5]
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if num_iters < 800:
            num_iters = 800
        for t in count():
            # termination if cant reach the target
           
            if t > num_iters:
                reward = torch.tensor([-650], device=device)
                if i_episode % 10 == 0 and num_iters > 800:
                    num_iters *= 0.9
                break
            
                
            
            print("\n=======================================")
            print("SINGLE STEP: {0}".format(t))
            action = select_action(state)
            print("Action: {0}".format(action.item()))
            observation, reward, terminated = env.step(action.item())
            print("Target: {0}".format(env.finish_coords))
            print("New STATE(position only): {0}".format(observation))
            print("Reward: {0}".format(reward))
            print("Terminated: {0}".format(terminated))
            deviation = env.get_deviation() # methods for compass
            # print("Deviation from the goal (rad):", deviation)
            print(f"Is Collision: {env.is_collision()}") # controll collision status

            reward = torch.tensor([reward], device=device)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the policy network)
            optimize_model()
    
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
    
            if terminated:
                mtSystem.stop()
                robot.step(timestep)
                episode_durations.append(t + 1)
                break
           