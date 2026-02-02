package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent that implements the Value Iteration algorithm to
 * compute an optimal policy for playing Tic-Tac-Toe based on a Markov Decision
 * Process (MDP) model.
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map stores the values of states
	 */
	Map<Game, Double> valueFunction = new HashMap<>();

	/**
	 * The discount factor
	 */
	double discount = 0.9;

	/**
	 * The MDP model
	 */
	TTTMDP mdp = new TTTMDP();

	/**
	 * the number of iterations to perform - feel free to change this/try out
	 * different numbers of iterations
	 */
	int k = 50;

	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent() {
		super();
		mdp = new TTTMDP();
		this.discount = 0.9;
		initValues();
		train();
	}

	/**
	 * Constructor that initializes the agent with an existing policy
	 * 
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
	}

	public ValueIterationAgent(double discountFactor) {
		this.discount = discountFactor;
		mdp = new TTTMDP();
		initValues();
		train();
	}

	/**
	 * Initializes the valueFunction map with initial values (V0), setting the value
	 * of all states to 0
	 */
	public void initValues() {
		List<Game> allGames = Game.generateAllValidGames('X'); // All valid games where it is X's turn or terminal
		for (Game g : allGames) {
			this.valueFunction.put(g, 0.0); // Initially, all states have a value of 0
		}
	}

	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward,
			double drawReward) {
		this.discount = discountFactor;
		mdp = new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}

	/**
	 * Computes the value of taking a specific action in a given state.
	 */
	private double computeActionValue(Game g, Move action) {
		double actionVal = 0.0;
		List<TransitionProb> transition = mdp.generateTransitions(g, action);

		// Calculating the expected value of the action based on the transitions
		for (TransitionProb tp : transition) {
			double prob = tp.prob;
			double reward = tp.outcome.localReward;
			Game state = tp.outcome.sPrime;
			actionVal += prob * (reward + discount * valueFunction.get(state));
		}
		return actionVal;
	}

	/**
	 * Performs k iterations of value iteration to compute the value of each state.
	 */
	public void iterate() {
		for (int i = 0; i < k; i++) {
			// Creating a new map to hold the updated values for this iteration
			Map<Game, Double> updatedValue = new HashMap<>();

			// Iterating over each game state in the current value function
			for (Game gameState : valueFunction.keySet()) {
				// keeping the state value unchanged if it's Terminal
				if (mdp.isTerminal(gameState)) {
					updatedValue.put(gameState, valueFunction.get(gameState));
					continue;
				}

				double maxValue = Double.NEGATIVE_INFINITY;
				// Evaluating each possible action from the current state
				for (Move action : gameState.getPossibleMoves()) {
					double actionValue = computeActionValue(gameState, action);
					maxValue = Math.max(maxValue, actionValue);
				}
				// Updating the new value function for the current state
				updatedValue.put(gameState, maxValue);
			}

			// Replacing the old value function with the updated one
			valueFunction = updatedValue;
		}
	}

	/**
	 * Extracts the policy from the value function. After running this method, the
	 * agent's policy will be updated according to the computed values.
	 * 
	 * @return The policy corresponding to the value function
	 */
	public Policy extractPolicy() {
		HashMap<Game, Move> policyMap = new HashMap<>();

		// Iterating through each state in the value function
		for (Game gState : valueFunction.keySet()) {
			// Only consider non-terminal states for policy extraction
			if (!mdp.isTerminal(gState)) {
				Move bestMove = null;
				double maxVal = Double.NEGATIVE_INFINITY;

				// Evaluating each possible action from the current state
				for (Move action : gState.getPossibleMoves()) {
					double actValue = computeActionValue(gState, action);

					// Updating the best move if the current action is better
					if (actValue > maxVal) {
						maxVal = actValue;
						bestMove = action;
					}
				}

				// Storing the best move for the current state in the policy map
				policyMap.put(gState, bestMove);
			}
		}
		return new Policy(policyMap);
	}

	/**
	 * This method solves the mdp using your implementation of
	 * {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}.
	 */
	public void train() {
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in
		 * {@link ValueIterationAgent#valueFunction} and set the agent's policy
		 * 
		 */

		super.policy = extractPolicy();

		if (this.policy == null) {
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			// System.exit(1);
		}

	}

	public static void main(String a[]) throws IllegalMoveException {
		// Test method to play the agent against a human agent
		ValueIterationAgent agent = new ValueIterationAgent();
		HumanAgent human = new HumanAgent();

		Game game = new Game(agent, human, human); // Agent plays as 'X', Human as 'O'
		game.playOut(); // Simulate and print the game
	}
}
