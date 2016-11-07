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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        manhattan_to_food = []
        manhattan_to_ghost = []

        for dot in newFood:
            distance = manhattanDistance(newPos, dot)
            manhattan_to_food.append(distance)

        for ghost in newGhostStates:
            distance = manhattanDistance(newPos, ghost.getPosition())
            manhattan_to_ghost.append(distance)

        if newScaredTimes[0] > 0:
            return 10 + successorGameState.getScore()
        else:
            if len(manhattan_to_food) > 0 and len(manhattan_to_ghost) > 0:
                score = min(manhattan_to_ghost) - min(manhattan_to_food) + successorGameState.getScore()
                if min(manhattan_to_ghost) == 1 and min(manhattan_to_food) == 1:
                    return -1
                else:
                    return score
            else:
                return successorGameState.getScore()


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
        """
        next_action = "North"
        number_of_agents = gameState.getNumAgents()
        # score returned after evaluating all of the actions until depth is reached and returns score
        # after that depth
        def after_depth_action(gameState, index, depth):

            index = index % number_of_agents
            legal_actions = gameState.getLegalActions(index)
            score_after_number_of_depth_moves = 0

            if depth == self.depth:
                return self.evaluationFunction(gameState)

            if index == 0:
                if not legal_actions:
                    return self.evaluationFunction(gameState)
                temp = -1000
                for action in legal_actions:
                    next_val = after_depth_action(gameState.generateSuccessor(index, action), index +1, depth)
                    if next_val > temp:
                        temp = next_val
                        score_after_number_of_depth_moves = temp
            else:
                if not legal_actions:
                    return self.evaluationFunction(gameState)
                temp = 1000
                if index == number_of_agents -1:
                    for action in legal_actions:
                        next_val = after_depth_action(gameState.generateSuccessor(index, action), index + 1, depth + 1)
                        if next_val < temp:
                            temp = next_val
                            score_after_number_of_depth_moves = temp

                else:
                    if not legal_actions:
                        return self.evaluationFunction(gameState)

                    for action in legal_actions:
                        next_val = after_depth_action(gameState.generateSuccessor(index, action), index + 1, depth)
                        if next_val < temp:
                            temp = next_val
                            score_after_number_of_depth_moves = temp
            return score_after_number_of_depth_moves

        best_score = -1000
        for action in gameState.getLegalActions(0):
            temp = after_depth_action(gameState.generateSuccessor(0, action), 1, 0)
            if temp > best_score:
                best_score = temp
                next_action = action
        return next_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        next_action = "North"
        number_of_agents = gameState.getNumAgents()

        def after_depth_action(gameState, index, depth, alpha, beta):

            index = index % number_of_agents
            legal_actions = gameState.getLegalActions(index)
            score_after_number_of_depth_moves = 0

            if depth == self.depth:
                return self.evaluationFunction(gameState)

            if index == 0:
                if not legal_actions:
                    return self.evaluationFunction(gameState)
                temp = -1000
                for action in legal_actions:
                    next_val = after_depth_action(gameState.generateSuccessor(index, action), index + 1, depth, alpha, beta)
                    if next_val > temp:
                        temp = next_val
                        score_after_number_of_depth_moves = temp
                    # next 4 lines additional to minimax agent
                    if temp > beta:
                        return temp
                    if temp > alpha:
                        alpha = temp

            else:
                if not legal_actions:
                    return self.evaluationFunction(gameState)
                temp = 1000

                if index == number_of_agents - 1:
                    for action in legal_actions:
                        next_val = after_depth_action(gameState.generateSuccessor(index, action), index + 1, depth + 1, alpha, beta)
                        if next_val < temp:
                            temp = next_val
                            score_after_number_of_depth_moves = temp
                        # next lines are additional to minimax agent
                        if temp < alpha:
                            return temp
                        if temp < beta:
                            beta = temp

                else:
                    if not legal_actions:
                        return self.evaluationFunction(gameState)

                    for action in legal_actions:
                        next_val = after_depth_action(gameState.generateSuccessor(index, action), index + 1, depth, alpha, beta)
                        if next_val < temp:
                            temp = next_val
                            score_after_number_of_depth_moves = temp
                            # added next lines only
                        if temp < alpha:
                            return temp
                        if temp < beta:
                            beta = temp

            return score_after_number_of_depth_moves

        # alpha and beta are additional to minimax agent
        best_score = -1000
        alpha = -1000
        beta = 1000

        for action in gameState.getLegalActions(0):
            temp = after_depth_action(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            # updating alpha
            if temp > alpha:
                alpha = temp
        # ------------------------------
            if temp > best_score:
                best_score = temp
                next_action = action
        return next_action


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
        next_action = "North"
        number_of_agents = gameState.getNumAgents()

        """
        ONLY DIFFERENCE WILL BE WHEN TAKING THE MINIMUM VALUE WE WILL DIVIDE EACH LEGAL ACTION'S RETURNED
        VALUE BY THE TOTAL NUMBER OF LEGAL ACTIONS SO EACH LEGAL ACTION HAS THE SAME WEIGHT IN PERCENTAGES

        """
        def after_depth_action(gameState, index, depth):

            index = index % number_of_agents
            legal_actions = gameState.getLegalActions(index)
            score_after_number_of_depth_moves = 0

            if depth == self.depth:
                return self.evaluationFunction(gameState)

            if index == 0:
                if not legal_actions:
                    return self.evaluationFunction(gameState)
                temp = -1000
                for action in legal_actions:
                    next_val = after_depth_action(gameState.generateSuccessor(index, action), index + 1, depth)
                    if next_val > temp:
                        temp = next_val
                        score_after_number_of_depth_moves = temp
            else:
                if not legal_actions:
                    return self.evaluationFunction(gameState)
                temp = 1000
                if index == number_of_agents - 1:
                    for action in legal_actions:
                        next_val = after_depth_action(gameState.generateSuccessor(index, action), index + 1, depth + 1)
                        # only changed part is the following (in comparison to minimax agent):
                        temp += next_val / len(legal_actions)
                    score_after_number_of_depth_moves = temp

                else:
                    if not legal_actions:
                        return self.evaluationFunction(gameState)

                    for action in legal_actions:
                        next_val = after_depth_action(gameState.generateSuccessor(index, action), index + 1, depth)
                        # only changed part is the following (in comparison to minimax agent):
                        temp += next_val / len(legal_actions)
                    score_after_number_of_depth_moves = temp

            return score_after_number_of_depth_moves

        best_score = -1000
        for action in gameState.getLegalActions(0):
            temp = after_depth_action(gameState.generateSuccessor(0, action), 1, 0)
            if temp > best_score:
                best_score = temp
                next_action = action
        return next_action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    if currentGameState.isWin():
        return 1000000
    if currentGameState.isLose():
        return -1000000


    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    score = currentGameState.getScore()
    manhattan_to_food = []
    manhattan_to_ghost = []

    for dot in newFood:
        distance = manhattanDistance(newPos, dot)
        manhattan_to_food.append(distance)

    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        manhattan_to_ghost.append(distance)

    if newScaredTimes[0] > 0:
        score += newScaredTimes[0] * 10
    else:
        if len(manhattan_to_food) > 0 and len(manhattan_to_ghost) > 0:
            score += min(manhattan_to_ghost) - min(manhattan_to_food) * 2
            if min(manhattan_to_ghost) == 1 and min(manhattan_to_food) == 1:
                return -1
        else:
            return score


    return score


# Abbreviation
better = betterEvaluationFunction
