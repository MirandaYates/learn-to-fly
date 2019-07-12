import sys
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd

from collections import defaultdict
from plane_env import PlaneEnv
import plotting

matplotlib.style.use('ggplot')

env = PlaneEnv()

#number of episodes run
EPOCHS = 500
EP_MOD = 101

# number of time it must reach pitch=0 and altitude=goal
DONE_STATE = 100


def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
	""" 
	Creates an epsilon-greedy policy based 
	on a given Q-function and epsilon. 
	
	Returns a function that takes the state 
	as an input and returns the probabilities 
	for each action in the form of a numpy array 
	of length of the action space(set of possible actions). 
	"""
	def policyFunction(state): 

		Action_probabilities = np.ones(num_actions, 
				dtype = float) * epsilon / num_actions 
				
		best_action = np.argmax(Q[state]) 
		Action_probabilities[best_action] += (1.0 - epsilon)
		return Action_probabilities 

	return policyFunction 


def qLearning(env, num_episodes, discount_factor = 0.6, 
							alpha = 0.1, epsilon = 0.1): 
	""" 
	Q-Learning algorithm: Off-policy TD control. 
	Finds the optimal greedy policy while improving 
	following an epsilon-greedy policy"""
	
	# Action value function 
	# A nested dictionary that maps 
	# state -> (action -> action-value). 
	Q = defaultdict(lambda: np.zeros(env.action_space.n)) 

	# Keeps track of useful statistics 
	stats = plotting.EpisodeStats( 
		episode_lengths = np.zeros(num_episodes), 
		episode_rewards = np.zeros(num_episodes),
		alpha = alpha,
		discount_factor = discount_factor)

	#contains lists of altitudes
	all_alt_lists = []
	
	# Create an epsilon greedy policy function 
	# appropriately for environment action space 
	policy = createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n) 

	print(env.observation_space)
	#initialize capture altitude for EPOCH
	alt_capture = False
	# For every episode 
	for ith_episode in range(num_episodes): 
		
		# Reset the environment and pick the first action 
		state = env.reset() 
		count = 0
		
		#set altitude capture boolean to True for chosen Epochs
		if ith_episode % EP_MOD == 0:
			alt_capture = True
			alt_list = []

		for t in itertools.count(): 

			# get probabilities of all actions from current state 
			action_probabilities = policy(state)

			# choose action according to 
			# the probability distribution 
			action = np.random.choice(np.arange( 
					len(action_probabilities)), 
					p = action_probabilities) 

			# take action and get reward, transit to next state 
			next_state, reward, done, _ = env.step(action) 
			
			#if capturing altitude for this EPOCH 
			if alt_capture:

				#add altitude to list
				alt_list.append(env.get_pitch(next_state))

			# Update statistics 
			stats.episode_rewards[ith_episode] += reward 
			stats.episode_lengths[ith_episode] = t 
			
			# TD Update 
			best_next_action = np.argmax(Q[next_state])	 
			td_target = reward + discount_factor * Q[next_state][best_next_action] 
			td_delta = td_target - Q[state][action] 
			Q[state][action] += alpha * td_delta 

			# done is True if episode terminated 
			if done: 
				count += 1

			# reaches done state 10 times
			if count > DONE_STATE:
				break

			# won't go forever
			if t > 1000:
				break
				
			state = next_state

		if alt_capture:

			#add to graph/list
			alt_capture = False
			all_alt_lists.append(alt_list)

	#plotting.plot_alt(all_alt_lists, EP_MOD)
	return Q, stats

stats_list = []
for a in range(1, 5):
	Q, stats = qLearning(env, EPOCHS, alpha=a/5, discount_factor=0.6, epsilon=0.1) 
	stats_list.append(stats)
plotting.plot_comparison(stats_list, "alpha ")

