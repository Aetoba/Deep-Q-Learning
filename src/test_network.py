import pickle
import tensorflow as tf
from model.qnetwork import Q_cnn
from utils.run_network import train, loss_on_task
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
Q = Q_cnn(0.0001, 6)
gpu_options = tf.GPUOptions()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())


### Get samples ###
actions = []
rewards = []
count = 0
done = False
with open('./data/not_runs/labels.pickle', 'rb') as f:
    while not done:
        try:
            action, reward = pickle.load(f)
            actions.append(action)
            rewards.append(np.sign(reward))
            count += 1
        except:
            done = True

done = False
states = []
with open('./data/not_runs/states.pickle', 'rb') as f:
    while not done:
        try:
            states.append(pickle.load(f))
        except:
            done = True

print("There are " + str(count) + " samples")
not_zeros = [i for i in range(len(rewards)) if rewards[i] != 0]
print("Of which " + str(len(not_zeros)) + " have a non-zero reward")

not_zeros_states = np.array(states)[not_zeros]
not_zeros_rewards = np.array(rewards)[not_zeros]
not_zeros_actions = np.array(actions)[not_zeros]



def get_loss(states, rewards, actions):
    i = 0
    tot_loss = 0
    done = False
    while not done:
        batch_x = states[i:i+32]
        batch_y = rewards[i:i+32]
        batch_a = actions[i:i+32]
        i += 32
        if len(batch_x) == 0:
            done = True
            break
        tot_loss += loss_on_task(sess, Q, batch_x, batch_y, batch_a)

    return tot_loss/count

def get_nonzero_loss():
    return loss_on_task(sess, Q, not_zeros_states, not_zeros_rewards, not_zeros_actions)/len(not_zeros)


### Main training loop ###
epochs = 100

avg = get_loss(states, rewards, actions)
avgs = [avg]

avg = get_nonzero_loss()
not_zeros_avgs = [avg]
print(avg)
for j in range(epochs):
    i = 0
    done = False
    while not done:
        batch_x = states[i:i+32]
        batch_y = rewards[i:i+32]
        batch_a = actions[i:i+32]
        i += 32
        if len(batch_x) == 0:
            done = True
            break 

        train(sess, Q, batch_x, batch_y, batch_a)
    
    avg = get_loss(states, rewards, actions)
    avgs.append(avg)

    avg = get_nonzero_loss()
    not_zeros_avgs.append(avg)
    print(avg)

plt.plot(avgs)
plt.plot(not_zeros_avgs)
plt.show()

