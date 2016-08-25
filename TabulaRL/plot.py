from gridworld_plot import plot 

action_directions = [
        plot.CENTER, 
        plot.UP, 
        plot.RIGHT, 
        plot.DOWN, 
        plot.LEFT]

def plotQ(env, mdp, agent, timestep=0):
    fig, ax = plot.grid_plot(env)
    tq = agent.qVals
    q = { s : tq[s,0] for s in range(env.nState) }

    plot.plotQ(ax, env, q, plot.plot_labeled_arrows(action_directions))
    plot.plotR(ax, env, mdp.R)
    plot.plotRBelief(ax, env, agent.R_prior)
    fig.show()
