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
        next_values = util.Counter()
        for iteration in range(0, self.iterations):
            for state in self.mdp.getStates():
                values = []
                if self.mdp.isTerminal(state):
                    values.append(0)

                for action in self.mdp.getPossibleActions(state):
                    values.append(self.computeQValueFromValues(state, action))

                next_values[state] = max(values)

            self.values = next_values.copy()


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
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        values = []
        for transition in transitions:
            reward = self.mdp.getReward(state, action, transition[0])
            values.append(transition[1] * (reward + self.discount * self.values[transition[0]]))

        return sum(values)

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

        if not actions :
            return None
        
        is_set = False;
        q_value = 0;
        next_action = None;
        for action in actions:
            value = self.computeQValueFromValues(state,action);
            if not is_set:
                q_value = value;
                next_action = action;
                is_set = True;
            else:
                if value > q_value:
                    q_value = value;
                    next_action = action;
        return next_action;

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates();
        for i in range(self.iterations):
            state = states[i % len(states)];
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state);
                maximum = max([self.getQValue(state,a) for a in actions]);
                self.values[state]=maximum;

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
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
    def get_predecessors(self):
        predecessors = {};
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state,action):
                        if next_state in predecessors:
                            predecessors[next_state].add(state);
                        else:
                            predecessors[next_state] = {state};
        return predecessors;
    
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        p_q = util.PriorityQueue();
        pres = self.get_predecessors();
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                q_max = max([self.getQValue(s,action) for action in self.mdp.getPossibleActions(s)]);
                difference = abs(self.values[s]- q_max);
                p_q.push(s,-difference);
        for i in range(self.iterations):
            if p_q.isEmpty():
                break;
            state = p_q.pop();
            if not self.mdp.isTerminal(state):
                q_max = max([self.getQValue(state,action) for action in self.mdp.getPossibleActions(state)]);
                self.values[state] = q_max;
            for pre in pres[state]:
                if not self.mdp.isTerminal(pre):
                    q_max = max([self.getQValue(pre,action) for action in self.mdp.getPossibleActions(pre)]);
                    difference = abs(self.values[pre] - q_max);
                    if difference > self.theta:
                        p_q.update(pre,-difference)
       


