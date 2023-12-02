import numpy as np
import tensorflow as tf
from rlcard.agents import RandomAgent
import rlcard

class PPOAgent:
    def __init__(self, model, epochs, batch_size, learning_rate, ent_coef, clip_ratio=0.2, vf_coef=0.5):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.ent_coef = ent_coef
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef

    def _compute_loss(self, old_probs, advantages, values, new_probs):
        advantages = tf.stop_gradient(advantages)
        ratios = tf.exp(tf.math.log(new_probs + 1e-10) - tf.math.log(old_probs + 1e-10))
        surr1 = ratios * advantages
        surr2 = tf.clip_by_value(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        value_loss = tf.reduce_mean(tf.square(values - advantages))
        entropy_loss = -tf.reduce_mean(new_probs * tf.math.log(new_probs + 1e-10))
        total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
        return total_loss

    def predict(self, state):
        logits, _ = self.model.predict(state)
        action_probs = tf.nn.softmax(logits)
        action = np.random.choice(np.arange(len(action_probs[0])), p=action_probs[0])
        return action, action_probs[0, action]

    def train(self, states, actions, rewards, old_probs):
        for epoch in range(self.epochs):
            idx = np.arange(len(states))
            np.random.shuffle(idx)

            for i in range(0, len(states), self.batch_size):
                batch_idx = idx[i:i + self.batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_rewards = rewards[batch_idx]
                batch_old_probs = old_probs[batch_idx]

                with tf.GradientTape() as tape:
                    logits, values = self.model(batch_states)
                    new_probs = tf.nn.softmax(logits)

                    advantages = batch_rewards - values
                    loss = self._compute_loss(batch_old_probs, advantages, values, new_probs)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

