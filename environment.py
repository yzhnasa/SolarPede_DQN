import random
import numpy as np
import time
from datetime import datetime
from reinforcement_learning import DQN


MAX_LASER_CURRENT = 5.5
MAX_LASER_FREQUENCY = 5000
MIN_LASER_CURRENT = 3
MIN_LASER_FREQUENCY = 600
STOP_LASER_CURRENT = 2.5
action_dim = 8  # rnn input size
state_dim = 3

GOAL_STATE_ERROR = 0.01

class Solarped(object):
    def __init__(self):
        self.agent = DQN(action_dim=action_dim, state_dim=state_dim)

    def reward(self, goal_state, next_state):
        x_goal = goal_state[0]
        y_goal = goal_state[1]
        angle_goal = goal_state[2]
        x_current = next_state[0]
        y_current = next_state[1]
        angle_current = next_state[2]
        reward = -np.sqrt(np.square(x_current-x_goal)+np.square(y_current-y_goal)) - np.abs(angle_current-angle_goal)
        return reward

    def select_action(self, current_state):
        action = self.agent.select_action(current_state, add_noise=False)
        return action

    def learn(self):
        if self.agent.is_memory_full():
            # print("learing")
            self.agent.learn()

    def store_experience(self, current_state, action, reward, next_state):
        self.agent.store_experience(current_state, action, reward, next_state)

    def is_done(self, goal_state, next_state):
        if np.sqrt(np.square(next_state[0]-goal_state[0])+np.square(next_state[1]-goal_state[1])) + np.abs(next_state[2]-goal_state[2]) < GOAL_STATE_ERROR:
            return True
        return False

    # def stop_robot(self):
    #     action = np.hstack((STOP_LASER_CURRENT, MIN_LASER_FREQUENCY))
    #     return action


    def save_agent(self):
        self.agent.save_model()
        # self.agent.save_critic()




current_state = 0
serpenbot = Solarped()
def init_training_control(init_state):
    global current_state
    current_state = np.array(init_state)
#    global serpenbot
#    serpenbot = Solarped()

def init_training_control_api(init_x, init_y, init_angle, actuator_number, state_number):
    global action_dim
    action_dim = actuator_number
    global state_dim
    state_dim = state_number
    init_state = [init_x, init_y, init_angle]
    init_training_control(init_state)

#def init_training_control_api(init_x, init_y, init_angle, actuator_number, state_number):
#    global action_dim
#    action_dim = actuator_number
#    global state_dim
#    state_dim = state_number
#    init_state = [init_x, init_y, init_angle]
#    global current_state
#    current_state = np.array(init_state)
#    global serpenbot
#    serpenbot = Solarped()

def training(goal_state, next_state):
    # Todo: return action (laser_current, laser_frequency)
    # Todo: if is_done(next_state), then turn off laser
    # print(next_state)
    goal_state = np.array(goal_state)
    next_state = np.array(next_state)
    global current_state
    action = serpenbot.select_action(next_state)
    reward = serpenbot.reward(goal_state, next_state)
    print("reward:", reward)

    serpenbot.store_experience(current_state, action, reward, next_state)
    current_state = next_state
    serpenbot.learn()

    # laser_current = np.round((MAX_LASER_CURRENT-MIN_LASER_CURRENT)*action[0]+MIN_LASER_CURRENT, 2)
    # laser_frequency = np.round((MAX_LASER_FREQUENCY-MIN_LASER_FREQUENCY)*action[1]+MIN_LASER_FREQUENCY, 0)
    # action = np.hstack((laser_current, laser_frequency))

    print("action:", action)

    return action

def training_api(goal_x, goal_y, goal_angle, next_x, next_y, next_angle):
    goal_state = [goal_x, goal_y, goal_angle]
    next_state = [next_x, next_y, next_angle]
    return training(goal_state, next_state)

def finish_training_api():
    serpenbot.save_agent()

def control(current_state):
    # Todo: return action (laser_current, laser_frequency)
    action = serpenbot.select_action(np.array(current_state))
    # laser_current = np.round((MAX_LASER_CURRENT - MIN_LASER_CURRENT) * action[0] + MIN_LASER_CURRENT, 2)
    # laser_frequency = np.round((MAX_LASER_FREQUENCY - MIN_LASER_FREQUENCY) * action[1] + MIN_LASER_FREQUENCY, 0)
    # action = np.hstack((laser_current, laser_frequency))
    return action

def control_api(current_x, current_y, current_angle):
    current_state = [current_x, current_y, current_angle]
    return control(current_state)

def is_done_episode(goal_state, next_state):
    return serpenbot.is_done(goal_state, next_state)

def is_done_episode_api(goal_x, goal_y, goal_angle, next_x, next_y, next_angle):
    goal_state = [goal_x, goal_y, goal_angle]
    next_state = [next_x, next_y, next_angle]
    return is_done_episode(goal_state, next_state)

def stop_robot():
    return serpenbot.stop_robot()

if __name__ == '__main__':
    random.seed(datetime.now())
    x = random.uniform(-1, 1) * 16
    y = random.uniform(-1, 1) * 16
    angle = random.random() * 360

    # global action_dim
    #action_dim = 8
    # global state_dim
    #state_dim = 3

    robot_state = [x, y, angle]
    # init_training_control(robot_state)
    goal_x = 10
    goal_y = 10
    goal_angle = 100
    goal_state = [goal_x, goal_y, goal_angle]
    init_training_control_api(goal_x, goal_y, goal_angle, 8, 3)

    for i in range(10000):
        x = random.random() * 1080
        y = random.random() * 720
        angle = random.random() * 360
        robot_state = [x, y, angle]
        start = time.time()
        # action = training(goal_state, robot_state)
        action = training_api(goal_x, goal_y, goal_angle, x, y, angle)
        end = time.time()
        print("execution time:", end - start)

    finish_training_api()

