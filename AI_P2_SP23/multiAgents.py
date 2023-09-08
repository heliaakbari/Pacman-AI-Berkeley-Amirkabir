# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
import search
from searchAgents import mazeDistance

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # nearest ghost next state
        GhostDistancesNext = [];
        for ghostState in newGhostStates:
            GhostDistancesNext.append(manhattanDistance(ghostState.getPosition(),newPos));
        nearestGhostNext = min(GhostDistancesNext);

        # nearest food now
        FoodDistancesNow = [];
        for food in currentGameState.getFood().asList():
            FoodDistancesNow.append(manhattanDistance(currentGameState.getPacmanPosition(), food));
        nearestFoodDistanceNow = min(FoodDistancesNow);
        
        #nearest food next state
        FoodDistancesNext = [];
        for food in newFood.asList():
            FoodDistancesNext.append(manhattanDistance(newPos, food));
        if len(FoodDistancesNext)==0:
            NearestFoodDistanceNext = 0;
        else:
            NearestFoodDistanceNext = min(FoodDistancesNext)

        #first priority: always keep some distance from the ghosts
        if nearestGhostNext <= 3:
            return 0
        #second priority: eat food
        elif currentGameState.getFood().count() - newFood.count() > 0:
            return 3
        #third priority: get to the nearest food
        elif nearestFoodDistanceNow - NearestFoodDistanceNext > 0:
            return 2
        else:
            return 1
        
def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state,agent,depth):
            v = -100000;
            legalActions = state.getLegalActions(agent);
            SuccessorStates = [];
            for action in legalActions:
                SuccessorStates.append(state.generateSuccessor(agent, action));
            for successor in SuccessorStates:
                if gameState.getNumAgents() -1 == agent:
                    v = max(v,value(successor,0,depth-1));
                else :
                    v = max(v,value(successor,agent+1,depth));
            return v;
    
        def minValue(state,agent,depth):
            v = +100000;
            legalActions = state.getLegalActions(agent);
            SuccessorStates = [];
            for action in legalActions:
                SuccessorStates.append(state.generateSuccessor(agent, action));
            for successor in SuccessorStates:
                if gameState.getNumAgents() -1 == agent:
                    v = min(v,value(successor,0,depth-1));
                else :
                    v = min(v,value(successor,agent+1,depth));
            return v;
    
        def value(state,agent,depth):
            #terminal states
            if state.isLose():
                return(self.evaluationFunction(state));
            if state.isWin():
                return(self.evaluationFunction(state));
            if depth==0:
                return(self.evaluationFunction(state));
        
            if agent == 0:
                return maxValue(state,agent,depth);
            else :
                return minValue(state,agent,depth);

        #first time:
        finalAction = 0;
        v = -100000;
        legalActions = gameState.getLegalActions(0);
        SuccessorStates = [];
        for action in legalActions:
            SuccessorStates.append((gameState.generateSuccessor(0, action),action));
        for successor in SuccessorStates:
            if gameState.getNumAgents() -1 == 0:
                valueFunc = value(successor[0],0,self.depth-1);
                if v < valueFunc:
                    v = valueFunc;
                    finalAction = successor[1];
            else :
                valueFunc = value(successor[0],1,self.depth);
                if v < valueFunc :
                    v = valueFunc;
                    finalAction = successor[1];
        
        return finalAction;

                

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha= -10000;
        beta = 10000;
        def maxValue(state,agent,depth,alpha1,beta):
            v = -10000;
            legalActions = state.getLegalActions(agent);
            SuccessorStates = [];
            alpha = alpha1;
            if not legalActions:
                return self.evaluationFunction(state)
            for action in legalActions:
                successor = state.generateSuccessor(agent, action);
                if gameState.getNumAgents() -1 == agent:
                    v = max(v,value(successor,0,depth-1,alpha,beta));
                else :
                    v = max(v,value(successor,agent+1,depth,alpha,beta));
                if v > beta : return v;
                alpha = max(alpha,v);
            return v;
    
        def minValue(state,agent,depth,alpha,beta1):
            v = +10000;
            legalActions = state.getLegalActions(agent);
            SuccessorStates = [];
            beta = beta1;
            if not legalActions:
                return self.evaluationFunction(state)
            for action in legalActions:
                successor = state.generateSuccessor(agent, action);
                if gameState.getNumAgents() -1 == agent:
                    v = min(v,value(successor,0,depth-1,alpha,beta));
                
                else :
                    v = min(v,value(successor,agent+1,depth,alpha,beta));
                if v < alpha : return v;
                beta = min(beta,v);
            return v;
    
        def value(state,agent,depth,alpha,beta):
            #terminal states
            if state.isLose():
                return(self.evaluationFunction(state));
            if state.isWin():
                return(self.evaluationFunction(state));
            if depth==0:
                return(self.evaluationFunction(state));
        
            if agent == 0:
                return maxValue(state,agent,depth,alpha,beta);
            else :
                return minValue(state,agent,depth,alpha,beta);

        #first time:
        finalAction = 0;
        v = -10000;
        legalActions = gameState.getLegalActions(0);
        SuccessorStates = [];
        for action in legalActions:
            successor =gameState.generateSuccessor(0, action); 
            valueFunc = 0;
            if gameState.getNumAgents() -1 == 0:
                valueFunc = value(successor,0,self.depth-1,alpha,beta);
                if v < valueFunc:
                    v = valueFunc;
                    finalAction = action;
            else :
                valueFunc = value(successor,1,self.depth,alpha,beta);
                if v < valueFunc :
                    v = valueFunc;
                    finalAction = action;
            if valueFunc > beta : return finalAction;
            alpha = max(alpha,valueFunc);
            
        return finalAction;


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def maxValue(state,agent,depth):
            v = -100000;
            legalActions = state.getLegalActions(agent);
            SuccessorStates = [];
            for action in legalActions:
                SuccessorStates.append(state.generateSuccessor(agent, action));
            for successor in SuccessorStates:
                if gameState.getNumAgents() -1 == agent:
                    v = max(v,value(successor,0,depth-1));
                else :
                    v = max(v,value(successor,agent+1,depth));
            return v;
    
        def ExpValue(state,agent,depth):
            v = 0;
            legalActions = state.getLegalActions(agent);
            SuccessorStates = [];

            for action in legalActions:
                SuccessorStates.append(state.generateSuccessor(agent, action));
            for successor in SuccessorStates:
                if gameState.getNumAgents() -1 == agent:
                    v = v+value(successor,0,depth-1);
                else :
                    v = v+value(successor,agent+1,depth);
            return float(v)/float(len(legalActions));
    
        def value(state,agent,depth):
            #terminal states
            if state.isLose():
                return(self.evaluationFunction(state));
            if state.isWin():
                return(self.evaluationFunction(state));
            if depth==0:
                return(self.evaluationFunction(state));
        
            if agent == 0:
                return maxValue(state,agent,depth);
            else :
                return ExpValue(state,agent,depth);

        #first time:
        finalAction = 0;
        v = -100000;
        legalActions = gameState.getLegalActions(0);
        SuccessorStates = [];
        for action in legalActions:
            SuccessorStates.append((gameState.generateSuccessor(0, action),action));
        for successor in SuccessorStates:
            if gameState.getNumAgents() -1 == 0:
                valueFunc = value(successor[0],0,self.depth-1);
                if v < valueFunc:
                    v = valueFunc;
                    finalAction = successor[1];
            else :
                valueFunc = value(successor[0],1,self.depth);
                if v < valueFunc :
                    v = valueFunc;
                    finalAction = successor[1];
        
        return finalAction;

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    heuristic = 10000;
    GhostDistances = [];
    for ghostState in ghostStates:
        ghostX = int(ghostState.getPosition()[0]);
        ghostY = int(ghostState.getPosition()[1]);
        GhostDistances.append(mazeDistance((ghostX,ghostY),pacmanPosition,currentGameState));
    if GhostDistances :
        nearestGhost = min(GhostDistances);
    else :
        nearestGhost = 10;
    # nearest food now
    FoodDistances = [];
    for food in foods.asList():
        FoodDistances.append(mazeDistance(pacmanPosition, food,currentGameState));
    if not FoodDistances:
        return 100000 + currentGameState.getScore();
    else:
        nearestFoodDistance = min(FoodDistances);
    #first priority: always keep some distance from the ghosts
    if nearestGhost <= 1:
        heuristic += -10000;
    heuristic -= 100*len(currentGameState.getCapsules());
    #second priority: eat food
    heuristic -= 10*nearestFoodDistance;
    heuristic -= 50*len(foods.asList());
    heuristic += currentGameState.getScore();
    return heuristic;


# Abbreviation
better = betterEvaluationFunction
