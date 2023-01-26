using for solving Markov Decision Process
MDP is a tuple of <States, Actions(capable moves), Transitions(Prob(S,a,S')), r, R>
	* Transitions is the prob that if you are at state S and you take action a that you will end up in state S'
	* r is representative of gamma which is the Discount Factor
	* R is the reward system

The most common type of Reinforcement learning is Q-Learning
	- Temperal difference
	- Q(S, A) keeps track of the q-value of each action in each state 
	- Boils down to the reward you get in S + discounted future reward
- Bell mans equation
![[Bellman.jpeg]]

Bell man's is similiar to an squared Error loss function in a neural network

Tabular Q Learning - uses dictionaries to hold Q values but gets pretty large after a while 

The most common form of this is called "Deep Q Learning" which replaces the dictionary with a neural net, can be useful if you have discrete number sets.

Solution: Policy (pi) 
A Policy is a function that states action  (S -> A)
Used to solve seqential problems. 
A* is usually the first place to look when trying to solve this kind of problem.

Good frame works and guides for setting these up:
	AiGym, Cliffwalker, Pull-Cart
	Using Simple Q Learners are a good place to start


Notes 
Use openCV for image Processing and maybe convert into a tensor,
Start with facial recognition data sets setting up a CNN
Tensorflow (data pipelines) and pytorch (data loader) are good places to start




