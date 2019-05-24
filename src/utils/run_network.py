import numpy as np

def make_preds(sess, Q, inputs):
    return sess.run(Q.output, feed_dict={
        Q.input_layer: inputs
    })

def make_chosen_preds(sess, Q, inputs, actions):
    return sess.run(Q.chosen, feed_dict={
        Q.input_layer: inputs,
        Q.action: actions
    })

def max_action_preds(sess, Q, inputs):
    preds = sess.run(Q.output, feed_dict={
        Q.input_layer: inputs
    })
    return np.argmax(preds, axis=1), np.max(preds, axis=1)

def train(sess, Q, inputs, targets, actions):
    loss, _ = sess.run([Q.loss, Q.optimizer], feed_dict={
        Q.input_layer: inputs,
        Q.target: targets,
        Q.action: actions
    })

    return loss

def loss_on_task(sess, Q, inputs, targets, actions):
    loss = sess.run(Q.loss, feed_dict={
        Q.input_layer: inputs,
        Q.target: targets,
        Q.action: actions
    })

    return loss