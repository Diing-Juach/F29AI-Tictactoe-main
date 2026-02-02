package ticTacToe;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A policy iteration agent. You should implement the following methods: (1)
 * {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation
 * step from your lectures (2) {@link PolicyIterationAgent#improvePolicy}: this
 * is the policy improvement step from your lectures (3)
 * {@link PolicyIterationAgent#train}: this is a method that should
 * runs/alternate (1) and (2) until convergence.
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration:
 * Convergence of the Values of the current policy, and Convergence of the
 * current policy to the optimal policy. The former happens when the values of
 * the current policy no longer improve by much (i.e. the maximum improvement is
 * less than some small delta). The latter happens when the policy improvement
 * step no longer updates the policy, i.e. the current policy is already
 * optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current
	 * policy (policy evaluation).
	 */
	HashMap<Game, Double> policyValues = new HashMap<Game, Double>();

	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}.
	 */
	HashMap<Game, Move> curPolicy = new HashMap<Game, Move>();

	double discount = 0.9;

	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;

	/**
	 * The (convergence) delta
	 */
	double delta = 0.1;

	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol
	 * files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp = new TTTMDP();
		initValues();
		initRandomPolicy();
		train();

	}

	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * 
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);

	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP
	 * paramters (rewards, transitions, etc) as specified in {@link TTTMDP}
	 * 
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {

		this.discount = discountFactor;
		this.mdp = new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * 
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward,
			double drawReward) {
		this.discount = discountFactor;
		this.mdp = new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all
	 * states to 0 (V0 under some policy pi ({@link #curPolicy} from the lectures).
	 * Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to
	 * do this.
	 * 
	 */
	public void initValues() {
		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames)
			this.policyValues.put(g, 0.0);

	}

	/**
	 * generating a random Initial policy
	 */
	public void initRandomPolicy() {
		// Generating all valid game states where it's 'X's turn
		List<Game> valideGame = Game.generateAllValidGames('X');
		Random random = new Random();

		// Iterating through each valid game state
		for (Game game : valideGame) {
			List<Move> possibleMoves = game.getPossibleMoves();
			if (!possibleMoves.isEmpty()) {
				// Randomly selecting a move from the list of possible moves
				curPolicy.put(game, possibleMoves.get(random.nextInt(possibleMoves.size())));
			}
		}
	}

	/**
	 * Calculates the value of a given state based on the action taken.
	 */
	private double computeValue(Game g, Move action) {
		double value = 0.0;
		List<TransitionProb> transition = mdp.generateTransitions(g, action);

		// Iterating through each transition probability
		for (TransitionProb tp : transition) {
			Game state = tp.outcome.sPrime;
			double reward = tp.outcome.localReward;
			double prob = tp.prob;

			// Updating the value based on the transition's probability, reward, and
			// discounted future value
			value += prob * (reward + discount * policyValues.get(state));
		}

		// Returning the computed value of the action
		return value;
	}

	/**
	 * Performs policy evaluation steps until the maximum change in values is less
	 * than delta
	 */
	protected void evaluatePolicy(double delta) {
		double maxChange;

		do {
			maxChange = 0;
			HashMap<Game, Double> newPolicyValues = new HashMap<>(policyValues);

			// Iterating through all states in the current policy
			for (Game g : curPolicy.keySet()) {
				if (!mdp.isTerminal(g)) {
					Move action = curPolicy.get(g);
					double value = computeValue(g, action);
					double change = Math.abs(value - policyValues.get(g));

					// Calculating the change in value for this state
					maxChange = Math.max(maxChange, change);

					// Updating the new value for this state into the new policy values map
					newPolicyValues.put(g, value);
				}
			}
			// Replacing the old policy values with the updated ones
			policyValues = newPolicyValues;

		} while (maxChange > delta); // Continuing until convergence
	}

	/**
	 * Finds the best action for a given game state based on the current policy
	 * values.
	 */
	private Move bestAction(Game g) {
		Move bestMove = null;
		double bestValue = Double.NEGATIVE_INFINITY;

		// Iterating through all possible actions for the current game state
		for (Move action : g.getPossibleMoves()) {
			double actionValue = computeValue(g, action);

			// Updating the optimal action if this action has a higher value
			if (actionValue > bestValue) {
				bestValue = actionValue;
				bestMove = action;
			}
		}

		// Returning the action with the highest value
		return bestMove;
	}

	/**
	 * This method should be run AFTER the
	 * {@link PolicyIterationAgent#evaluatePolicy} train method to improve the
	 * current policy according to {@link PolicyIterationAgent#policyValues}. You
	 * will need to do a single step of expectimax from each game (state) key in
	 * {@link PolicyIterationAgent#curPolicy} to look for a move/action that
	 * potentially improves the current policy.
	 * 
	 * @return true if the policy improved. Returns false if there was no
	 *         improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy() {
		boolean policyChanged = false;

		// Iterating through each game state in the policy values
		for (Game game : policyValues.keySet()) {

			// Only consider non-terminal states for policy improvement
			if (!mdp.isTerminal(game)) {
				Move bestMove = bestAction(game);

				// Updating the policy If the best action is different from the current policy 
				if (bestMove != null && !bestMove.equals(curPolicy.get(game))) {
					curPolicy.put(game, bestMove);
					policyChanged = true;
				}
			}
		}

		// Returning whether the policy was updated
		return policyChanged;
	}

	/**
	 * This method should perform policy evaluation and policy improvement steps
	 * until convergence (i.e. until the policy no longer changes), and so uses your
	 * {@link PolicyIterationAgent#evaluatePolicy} and
	 * {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train() {
		boolean policyChanged;
		do {
			// Perform policy evaluation
			evaluatePolicy(delta);

			// Perform policy improvement and check if the policy has changed
			policyChanged = improvePolicy();

		} while (policyChanged); // Continue until the policy no longer improves

		// After the training is complete, set the final policy
		super.policy = new Policy(curPolicy); // Set the updated policy to the parent Agent class
	}

	public static void main(String[] args) throws IllegalMoveException {
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi = new PolicyIterationAgent();

		HumanAgent h = new HumanAgent();

		Game g = new Game(pi, h, h);

		g.playOut();

	}

}
