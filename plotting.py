import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards", "alpha", "discount_factor"])

def plot_episode_stats(stats, alpha, discount, e, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Rewards")

    title_string = "Episode Rewards over Time"
    subtitle_string = "alpha = " + str(alpha) + ", discount factor = " + str(discount) + ", epsilon = " + str(e)
    plt.suptitle(title_string, y=.98, fontsize=14)
    plt.title(subtitle_string, fontsize=10)

    if noshow:
        plt.close(fig1)
        plt.savefig("Graphs/RewardTimed/:" + str(alpha) + "df:" + str(discount) + "e:" + str(e) + ".png")
    else:
        plt.show(fig1)
        plt.savefig("Graphs/RewardTimed/a:" + str(alpha) + "df:" + str(discount) + "e:" + str(e) + ".png")

    fig2 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig2)
        plt.savefig("Graphs/Episode_Length_over_Time.png")
    else:
        plt.show(fig2)
        plt.savefig("Graphs/Episode_Length_over_Time.png")

    return fig1, fig2


def plot_alt(all_alt_lists, ep_num):
    # multiple line plot
    i = 0
    for l in all_alt_lists:
        plt.plot(range(len(l)), l, label="Episode "+str(i*ep_num))
        i += 1
    plt.legend()
    plt.show(plt)
    plt.savefig("Graphs/test.png")

def plot_comparison(stats, label):
    for s in stats:
        plt.plot(range(len(s.episode_rewards)), s.episode_rewards, alpha=0.5, label=label+ str(s.alpha))
    plt.legend()
    plt.show(plt)
    plt.savefig("Graphs/test2.png")