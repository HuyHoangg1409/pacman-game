# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            new_values = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                    continue

                best_action = self.computeActionFromValues(state)
                max_qvalue = self.computeQValueFromValues(state, best_action)
                new_values[state] = max_qvalue
            self.values = new_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value=0
        transitions = self.mdp.getTransitionStatesAndProbs(state,action)

        for nextState, prob in transitions:
            reward = self.mdp.getReward(state,action,nextState)
            v_next = self.discount * self.getValue(nextState)
            q_value += prob*(reward + v_next)
        return q_value
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        if len(actions) == 0:
            return None
        max_value = float('-inf')
        action_choice = None
        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_value:
                max_value = q_value
                action_choice = action
        return action_choice
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        prodecessors = {}
        for state in self.mdp.getStates():
            prodecessors[state] = set()
        
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue
            
            for action in self.mdp.getPossibleActions(state):
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)

                for nextState, prop in transitions:
                    if prop > 0:
                        prodecessors[nextState].add(state)
        
        pq = util.PriorityQueue()
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            max_value = float('-inf')
            for action in self.mdp.getPossibleActions(state):
                q_value = self.computeQValueFromValues(state, action)
                if q_value > max_value:
                    max_value = q_value
            
            diff = abs(self.values[state] - max_value)
            pq.push(state, -diff)

        for i in range(self.iterations):
            if pq.isEmpty():
                break

            s = pq.pop()
            if not self.mdp.isTerminal(s):
                max_value_s = float('-inf')
                for action in self.mdp.getPossibleActions(s):
                    q_value_s = self.computeQValueFromValues(s, action)
                    if q_value_s > max_value_s:
                        max_value_s = q_value_s
                self.values[s] = max_value_s


            for parent in prodecessors[s]:
                max_value = float('-inf')
                for action in self.mdp.getPossibleActions(parent):
                    q_value = self.computeQValueFromValues(parent, action)
                    if q_value > max_value:
                        max_value = q_value
                diff = abs(self.values[parent] - max_value)

                if diff > self.theta:
                    pq.update(parent, -diff)


