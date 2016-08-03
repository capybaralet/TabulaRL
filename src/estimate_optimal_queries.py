"""

To start with, we assume that T is known (probably it's simple to extend to unknown T)

We do the following (k times):
    Sample a reward function, R ~ P(R)
        Estimate optimal returns via planning with R
    Sample N queries from R for every (s,a) pair
        For every n in {1, ..., N}:
            Compute a new posterior P(R | first n sampled queries)
            Estimate optimal returns via planning over the new posterior
            
    We'd like to have a way to make a different number of queries to different (s,a).
        But naively, there are exponentially many different combinations.
            If we used the linear-bandit type approach, then that might tell use which (s,a) we'd want to query at each tstep
            so then we can run the linear bandit thing for different numbers of tsteps, and just search over the number of tsteps!

^ We can put this whole thing "in the loop" and use it to update our query strategy online!

TODO: 
    * implement everything
    * Compare to the thing where we actually run PSRL on the sampled R to compare the different values of n
    * Evaluate sensitivity to all random elements in the algo:
        ** Sampling R
        ** Sampling queries from R
        ** Running PSRL



can we use Rmax as a heuristic for what N should be??
    i.e. we'd like to turn the query_cost and gamma into epsilon and delta (for the PAC guarantees of Rmax), and then chose N based on that.

"""

def do_planning(mdp):

def estimate_returns(policy, mdp):

def 



