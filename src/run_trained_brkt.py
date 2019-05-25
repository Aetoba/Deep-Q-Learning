import tensorflow as tf
import gym

from utils.exploit_learning import Exploit
from model.qnetwork import Q_cnn


print("Type anything and press enter to save frames, otherwise just press enter")
capture = input() != ""


sess = tf.Session()
Q = Q_cnn(0.00025, 4)
env = gym.make('BreakoutDeterministic-v4')
saver = tf.train.Saver()
saver.restore(sess, './data/run_brkt/model.ckpt')


exploit = Exploit(sess, Q, env, 4, 4500)

exploit.run(cap=capture, cap_prepro=capture, render=True)