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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        #FOR FOOD
        minFoodDist = 1E9
        for x in range(newFood.width):
            for y in range(newFood.height):
                if newFood[x][y]:
                    foodManhattanDist = abs(x - newPos[0]) + abs(y - newPos[1])

                    if foodManhattanDist < minFoodDist:
                        minFoodDist = foodManhattanDist
        inverseMinFoodDist = 1.0 /(minFoodDist+1)

        #FOR GHOSTS
        minGhostDist = 1E9

        maxScaredTimeFactor = 0
        for i in range(0,len(newGhostStates)):
            ghostPos = newGhostStates[i].getPosition()
            ghostManhattanDist = abs(ghostPos[0] - newPos[0]) + abs(ghostPos[1] - newPos[1])
            ghostScaredTimer = newScaredTimes[i]

            if ghostManhattanDist < ghostScaredTimer:
                if (ghostScaredTimer - ghostManhattanDist) > maxScaredTimeFactor:
                    maxScaredTimeFactor = ghostScaredTimer - ghostManhattanDist
            else:
                if ghostManhattanDist < minGhostDist:
                    minGhostDist = ghostManhattanDist


        inverseMinGhostDist = 1.0/(minGhostDist+1)


        if minGhostDist < 3:
            inverseMinGhostDist = inverseMinGhostDist * 100
        totalScore = successorGameState.getScore() + inverseMinFoodDist - 0.9*inverseMinGhostDist + 0.1*maxScaredTimeFactor

        return totalScore









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
        finalLevel = gameState.getNumAgents() * self.depth

        def minimax(game_state, level):
            agentIndex = level % gameState.getNumAgents()
            if level >= finalLevel:
                print
                "Error with minimax function: levels_left should not be <= 0."
                return None

            elif level == finalLevel - 1:  # must be a ghost, minimizer
                bestAction = ''
                lowestEvalNum = 999999

                for action in game_state.getLegalActions(agentIndex):
                    newState = game_state.generateSuccessor(agentIndex, action)

                    evalNum = self.evaluationFunction(newState)

                    if evalNum < lowestEvalNum:
                        lowestEvalNum = evalNum
                        bestAction = action

                return (lowestEvalNum, bestAction)

            elif level % gameState.getNumAgents() == 0:  # Pacman's turn, maximizer
                bestAction = 'Stop'
                highestEvalNum = -999999

                for action in game_state.getLegalActions(agentIndex):
                    newState = game_state.generateSuccessor(agentIndex, action)

                    if newState.isWin() or newState.isLose():
                        evalNum = self.evaluationFunction(newState)
                    else:
                        evalNum, _ = minimax(newState, level + 1)

                    if evalNum > highestEvalNum:
                        highestEvalNum = evalNum
                        bestAction = action

                return (highestEvalNum, bestAction)

            else:  # Ghosts' turn, minimizer
                bestAction = 'Stop'
                lowestEvalNum = 999999

                for action in game_state.getLegalActions(agentIndex):
                    newState = game_state.generateSuccessor(agentIndex, action)

                    if newState.isWin() or newState.isLose():
                        evalNum = self.evaluationFunction(newState)
                    else:
                        evalNum, _ = minimax(newState, level + 1)

                    if evalNum < lowestEvalNum:
                        lowestEvalNum = evalNum
                        bestAction = action

                return (lowestEvalNum, bestAction)

        evalNum, action = minimax(gameState, 0)  # initially starts the level at 0
        return action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        finalLevel = gameState.getNumAgents() * self.depth

        def alphabeta(game_state, level, bestMin, bestMax):
            agentIndex = level % gameState.getNumAgents()
            if level >= finalLevel:
                return None

            elif level == finalLevel - 1:  # must be a ghost, minimizer
                bestAction = ''
                lowestEvalNum = float("inf")

                for action in game_state.getLegalActions(agentIndex):
                    newState = game_state.generateSuccessor(agentIndex, action)

                    evalNum = self.evaluationFunction(newState)
                    if evalNum < lowestEvalNum:
                        lowestEvalNum = evalNum
                        bestAction = action

                    if lowestEvalNum < bestMax: return (lowestEvalNum, bestAction)
                    bestMin = min(bestMin, lowestEvalNum)

                return (lowestEvalNum, bestAction)

            elif level % gameState.getNumAgents() == 0:  # Pacman's turn, maximizer
                bestAction = 'Stop'
                highestEvalNum = float("-inf")

                for action in game_state.getLegalActions(agentIndex):
                    newState = game_state.generateSuccessor(agentIndex, action)

                    if newState.isWin():
                        evalNum = self.evaluationFunction(newState)
                    elif newState.isLose():
                        evalNum = self.evaluationFunction(newState)
                    else:
                        evalNum, _ = alphabeta(newState, level + 1, bestMin, bestMax)

                    if evalNum > highestEvalNum:
                        highestEvalNum = evalNum
                        bestAction = action
                    if highestEvalNum > bestMin: return (highestEvalNum, bestAction)
                    bestMax = max(bestMax, highestEvalNum)

                return (highestEvalNum, bestAction)

            else:  # Ghosts' turn, minimizer
                bestAction = 'Stop'
                lowestEvalNum = float("inf")

                for action in game_state.getLegalActions(agentIndex):
                    newState = game_state.generateSuccessor(agentIndex, action)
                    if newState.isWin():
                        evalNum = self.evaluationFunction(newState)
                    elif newState.isLose():
                        evalNum = self.evaluationFunction(newState)
                    else:
                        evalNum, _ = alphabeta(newState, level + 1, bestMin, bestMax)

                    if evalNum < lowestEvalNum:
                        lowestEvalNum = evalNum
                        bestAction = action
                    if lowestEvalNum < bestMax: return (lowestEvalNum, bestAction)
                    bestMin = min(bestMin, lowestEvalNum)

                return (lowestEvalNum, bestAction)

        evalNum, action = alphabeta(gameState, 0, float("inf"), float("-inf"))  # initially starts the level at 0
        return action


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
        finalLevel = gameState.getNumAgents() * self.depth

        def expectimax(game_state, level):
            agentIndex = level % gameState.getNumAgents()

            if level >= finalLevel:
                print
                "Error with minimax function: levels_left should not be <= 0."
                return None

            elif level == finalLevel - 1:  # must be a ghost, random action chooser!
                sumEvalNums = 0.0  # making it a float
                actionsList = game_state.getLegalActions(agentIndex)

                for action in actionsList:
                    newState = game_state.generateSuccessor(agentIndex, action)
                    evalNum = self.evaluationFunction(newState)
                    sumEvalNums += evalNum

                return (
                sumEvalNums / len(actionsList), '')  # returning a blank action because the action doesn't matter

            elif level % gameState.getNumAgents() == 0:  # Pacman's turn, maximizer
                bestAction = 'Stop'
                highestEvalNum = -999999

                for action in game_state.getLegalActions(agentIndex):
                    newState = game_state.generateSuccessor(agentIndex, action)

                    if newState.isWin() or newState.isLose():
                        evalNum = self.evaluationFunction(newState)
                    else:
                        evalNum, _ = expectimax(newState, level + 1)

                    if evalNum > highestEvalNum:
                        highestEvalNum = evalNum
                        bestAction = action

                return (highestEvalNum, bestAction)

            else:
                sumEvalNums = 0.0  # making it a float
                actionsList = game_state.getLegalActions(agentIndex)

                for action in actionsList:
                    newState = game_state.generateSuccessor(agentIndex, action)

                    if newState.isWin() or newState.isLose():
                        evalNum = self.evaluationFunction(newState)
                    else:
                        evalNum, _ = expectimax(newState, level + 1)

                    sumEvalNums += evalNum

                return (
                sumEvalNums / len(actionsList), '')  # returning a blank action because the action doesn't matter

        evalNum, action = expectimax(gameState, 0)  # initially starts the level at 0
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """





    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()  # capsules aka power pellets
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]

    minFoodDist = 1E9
    minCapsuleDist = 1E9
    minGhostDist = 1E9

    # for food
    foodNotEaten = 0
    for x in range(food.width):
        for y in range(food.height):
            if food[x][y]:
                foodManDist = abs(x - pacmanPosition[0]) + abs(y - pacmanPosition[1])
                foodNotEaten +=1

                if foodManDist < minFoodDist:
                    minFoodDist = foodManDist
    if minFoodDist != 0:
        invMinFoodDist = 1.0 / minFoodDist
    else:
        invMinFoodDist = 1.0 / 1

    # for capsules
    for capsulePos in capsules:
        capsuleManDist = abs(capsulePos[0] - pacmanPosition[0]) + abs(capsulePos[1] - pacmanPosition[1])
        if capsuleManDist < minCapsuleDist:
            minCapsuleDist = capsuleManDist
    if minCapsuleDist != 0:
        invMinCapsuleDist = 1.0 / minCapsuleDist
    else:
        invMinCapsuleDist = 1.0 / 1

    if minCapsuleDist < 3:
        invMinCapsuleDist = invMinCapsuleDist * 100


    #for ghosts
    ghostCountsNear = 0
    maxScaredTimeFactor = 0

    for i in range(0, len(ghostStates)):
        ghostPos = ghostPositions[i]
        ghostManhattanDist = abs(ghostPos[0] - pacmanPosition[0]) + abs(ghostPos[1] - pacmanPosition[1])
        ghostScaredTimer = scaredTimers[i]

        if ghostManhattanDist < ghostScaredTimer:
            if (ghostScaredTimer - ghostManhattanDist) > maxScaredTimeFactor:
                maxScaredTimeFactor = ghostScaredTimer - ghostManhattanDist
        else:
            ghostCountsNear = ghostCountsNear + 1

            if ghostManhattanDist < minGhostDist:
                minGhostDist = ghostManhattanDist
    if minGhostDist != 0:
        inverseMinGhostDist = 1.0 / minGhostDist
    else:
        inverseMinGhostDist = 1.0 / 1


    if minGhostDist < 3:
        inverseMinGhostDist = inverseMinGhostDist * 100

    # evaluation function 1
    totalScore = currentGameState.getScore() + invMinFoodDist - inverseMinGhostDist + invMinCapsuleDist - len(
        capsules) * 1000 + maxScaredTimeFactor * 0.1

    #evaluation function 2
    # totalScore = currentGameState.getScore() + invMinFoodDist - inverseMinGhostDist + invMinCapsuleDist - len(
    #     capsules) * 1000 + maxScaredTimeFactor * 0.1 - ghostCountsNear* 100
    #
    # #evaluation function 3
    # totalScore = currentGameState.getScore() + invMinFoodDist - inverseMinGhostDist + invMinCapsuleDist - len(
    #     capsules) * 1000 + maxScaredTimeFactor * 0.1 - foodNotEaten*10

    # #evaluation function 3
    # totalScore = currentGameState.getScore() + invMinFoodDist - inverseMinGhostDist + invMinCapsuleDist - len(
    #     capsules) * 1000 + maxScaredTimeFactor * 0.1 - foodNotEaten *100-ghostCountsNear*100

    return totalScore


# Abbreviation
better = betterEvaluationFunction
