![Ghosts ML](https://github.com/Whatar/GhostsML/blob/main/imgs/GhostsML.gif)

Italian version of this documentation file is available [here](https://github.com/Whatar/GhostsML/blob/main/README.it.md)

# Artificial Intelligence methods for non-player characters in video games

## 0How to use the project

Enter the source folder with:

`cd src`

To run the trained version of ghosts against pacman, run the command:

`python ./run.py`

To re-train pacman:

`python ./run.py -t` or `python ./run.py --train`

To enable debugging (does not use the trained version of ghosts):

`python ./run.py -d` or `python ./run.py --debug`

To display the 4 ghosts:

`python ./run.py -4` or `python ./run.py --4ghosts`

## 1. Introduction

In the context of video game development, the application of Artificial Intelligence (AI) to create non-player characters (NPCs) presents a fascinating challenge These characters must demonstrate intelligent and responsive behaviors, thus enriching the game experience. In this thesis, we explore the use of Deep Q-Network (DQN), a machine learning technique, to achieve an alternative version to the AI of ghosts in the famous Pac-Man game.

### 1.1 Project Goals

The main goals of this thesis project are as follows

 - **Obtain an Automated Ghost AI**: An attempt will be made to develop an AI for ghosts in the classic arcade game Pac-Man and will try through training and epsilong greedy logic, to make it able to navigate the environment by moving two ghosts (blinky and pinky), looking for pacman.

 - **Use of Deep Q-Network (DQN)**: Machine learning capabilities will be exploited to find a coordination solution between two ghosts in order to catch Pacman.

 - **Performance evaluation**: We will conduct an accurate performance evaluation of DQN-trained ghosts. We will use specific metrics to measure their in-game behavior.

### 1.2 Background and relevance of the use of ML in video game AI development

Traditionally, video game AIs have been programmed with algorithms such as the behavior tree and other kinds of fixed rules, which we call heuristics for simplicity, which in many cases has been more than enough to provide a good gaming experience

 Sometimes, however, these heuristic methods have resulted in AIs that are predictable and somewhat limited in their response to player actions The use of ML allows for the creation of AIs that are more rational and adaptive and that can in some situations, enhance player immersion.

 There has been research in this area for some time, from divisions such as [LaForge](https://www.ubisoft.com/en-us/studio/laforge) at ubisoft, which has produced some interesting papers in this regard, such as this one on an [Autonomous Driving AI](https://arxiv.org/abs/1912.11077) for Watch Dogs

### 1.3 Structure of Topics

In this project we observe Pac-Man programmed using algorithms that make use of heuristics. to collect all the pellets and escape when in the vicinity of a ghost The ghosts are moved by a single DQN, and have as their goal to minimize the sum of their distances from Pac-Man, and capture him.

 We will look at, in order: the project structure, evaluation metrics, methodology used, implementation, problems and solutions, results and evaluation, future considerations, and conclusion.

## 2. Fundamentals of the Project

### 2.1 The game environment

The maze looks and behaves like that of the original game, but some aspects of the game have been modified to simplify the implementation of AI In particular, fruit is ignored by all types of agents (pacman/ghost), and power pellets have been removed, this is because in the original game they modify the behavior of the ghosts, causing them to flee back to their initial cell- Leaving this behavior would have resulted in a kind of time jump from the time when the DQN sees the state prior to the power pellet being eaten, and the next one, where at the end of the original AI's return home, it would find itself teleported. Which would have made the training more unstable.

 From the library, node management was inherited, but it is not sufficiently granular for the purposes of the project, so cell management was implemented that would first allow pacman to be able to be directed to the pellets, and then could provide more detailed information to the ghosts so they could be directed to pacman.

 The maze is equivalent to a 28x36 matrix, where each cell is a square of 16x16 pixels.The maze consists of 4 types of cells:

- **Walls**: Walls are the cells that cannot be crossed by any character They are represented by a dark blue square.

 - **Passages**: Passages are the cells that can be crossed by all characters. They are represented by a white square.

 - **Gates**: Gates are the cells through which one can pass to be teleported to the other side of the maze.

 Pacman and ghosts are equally fast, but ghosts have a slower reaction time, this is to reduce the size of the search space, and to make training easier.

 Specifically, pacman can decide every frame which action to perform, while ghosts (whose move is chosen by a DQN for both) can decide every 10 frames (0.2 seconds at 50fps).

### 2.2 Evaluation Metrics

To evaluate the effectiveness of the implementation of artificial intelligence algorithms in Pac-Man's nonplaying characters, it is essential to define appropriate evaluation metrics These metrics allow us to measure the performance, behavior, and challenge level of our NPCs.

The evaluation metrics set are:

1.  **Reward**: Reward is a measure of an agent's performance in a given state.

 2.  **Pellets captured by pacman**: The number of pellets captured by pacman (in 5 lifetimes), is a useful measure to indicate the pressure that ghosts have put on pacman.

## 3. Methodology

### 3.1 Description of the Learning Method

#### Model Architecture

After some testing, the following model was found to be sufficiently effective in solving the problem:

- **L1:** 6 inputs ( x, y ofPinky, Blinky, Pacman)

- **L2-3:** 256 Neurons

- **L4:** 16 outputs (4 possible directions for pinky, multiplied by 4 possible directions for blinky)

The activation function used for intermediate layers is ReLU

#### Deep Q-Network (DQN)

The core of the implemented machine learning approach is the Deep Q-Network (DQN) algorithm DQN is a form of RL that uses deep neural networks to approximate the Q-function, which quantifies the expected value of actions in a given state. This Q function is critical for making optimal decisions.

 Deep learning theory applied to arcade games, specifically atari, is a topic covered in detail [in this paper](https://arxiv.org/abs/1312.5602), which provides a comprehensive theoretical basis. For the implementation, however, official guidance from [PyTorch for DQNs](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) was followed.

 The core of the DQN is the neural network, which approximates the Q function The neural network takes as input the game state and returns a Q value for each possible action. The action with the highest Q value is selected and used to make the final decision. This is when the neural network is trained and has learned to correctly estimate Q-values for each action.

neural network implementation, similar to the official PyTorch implementation, looks like this:

    class DQN(nn.Module):
        def __init__(self, n_observations, n_actions):
            super(DQN, self).__init__()
            self.layer1 = nnLinear(n_observations, 256)
            self.layer2 = nn.Linear(256, 256)
            self.layer3 = nn.Linear(256, 256)
            self.layer4 = nn.Linear(256, n_actions)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))

            return self.layer4(x)

The action selection function:

    def select_action(state, game, dt):
        global steps_done
        sample = random.random()
        if gametraining:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
        else:
            eps_threshold = 0.00
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad()
                # Calculate Q-values for all actions
                q_values = policy_net(state)

                action = q_values.max(1)[1].view(1, 1)

                return action
        else:
            return randomAction(state, game, dt)

The random action is obtained by choosing a random integer in the range of possible actions

    def randomAction(state, game, dt):
        # random        action
action = random.randrange(n_actions)
        # possibleActions = list(range(n_actions))
        # while not isValidAction(action, game, dt):
        # q_values = policy_net(state)

        # q_values[0][action] = -float('inf')

        # possibleActions.remove(action)
        # action = random.choice(possibleActions)

        return torch.tensor([[action]], device=device, dtype=torchlong)

Originally tests were performed with intermediate filters within the random selection function, to prevent an action with obvious negative implications from being selected, such as choosing a direction that leads at least one of the two ghosts to collide against a wall, but after a few training sessions it turned out that this strategy leads to worse results than penalizing in the reward this behavior, so the validation part of the action was disabled

## Stages during Agent Training

1.**Acquisition of the State of

- Initially, the agent detects and acquires the current state of the environment, which will be the input to the neural network

 2.**Execution of Action with Epsilon-Greedy Strategy:

- An action is selected to be executed in the environment, using an epsilon-greedy strategy This approach balances exploration by random actions with exploitation of current estimates from the neural network.

 3.**Reward Calculation and Neural Network Update with optimize_model():**

- After the action is executed, the reward is calculated (to be seen later in detail)

- Next, the neural network is updated using the `optimize_model()` function

Analyzing the optimize_model() function, the steps performed are as follows:

1.**Check on memory size:**

- If the memory size (`memory`) is smaller than the batch size (`BATCH_SIZE`), there is not enough accumulated experience to perform an adequate model update In such a case, the function stops the execution of the optimization phase without performing any further operations.

 2.**Extraction of random transitions from memory:**

- Random extraction of transitions from memory helps decorrelate training data, improving learning stability and algorithm convergence
 - A transition is encoded in this format: `Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))` or (_S, A, \_S_′, R\_).State coding is explained in chapter [3.2]()

3.**Batch data preparation:

- Transposing the transitions provides a more suitable format for training the neural network, allowing for more efficient batch managementThat is, you unpack the transactions, which are placed in a list of this type:
  [[state1, action1, reward1, next_state1],[state2, action2, reward2, next_state2]]`
  and place them in a more easily managed list of this type
  `[[state1, state2],[action1, action2], [reward1, reward2], [next_state1, next_state2]]`

- The creation of the `non_final_mask` is necessary to identify the next states that have not been terminals, helping to correctly calculate the expected Q values

 4.**Calculation of expected Q-values

- The calculation of expected Q-values for subsequent states is based on the target network, providing training stability through the target fixing technique

 - The use of a mask ensures that the values for the terminal states do not contribute to the calculation of the final expected Q values.

 5.**Loss calculation (loss):

- Huber loss is used in place of quadratic loss to mitigate the effects of outliers in the training data, providing greater robustness to the algorithm, some additional details and references to insights available [here](https://arxiv.org/pdf/2108.12627.pdf)
  In particular, Huber loss unlike quadratic loss is less affected by extreme values, as can be seen in the image below.
  ![Huber Loss](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Huber_loss.svg/720px-Huber_loss.svg.png)

- Application of the mask ensures that the loss is calculated only for non-final states in the batch, focusing the model update on the relevant examples

 6.**Model Optimization:

- Clearing gradients is essential before each optimization step to avoid gradient buildup in the model

 - Clipping gradients is used to avoid numerical instability problems and to stabilize training by limiting the magnitude of gradients during back-propagation.PyTorch provides a very intuitive method to perform clipping:
  `torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)`

- Optimization by updating the model weights is the final step to improve the predictive ability of the model

 Regarding the replay memory, this is a data structure that stores the experiences accumulated during the game, where we save the aforementioned transitions. The replay memory is implemented as a circular buffer, which stores experiences as `(state, action, reward, next_state)` tuples. These tuples are randomly drawn from memory during the training phase.

### 3.2 Data Preprocessing

The game state is entered as input to the DQN in the form of a 6-element array, the position (x, y) of pacman, blinky, pinky divided by the width/height of the cells

gives the index of the cell where pacman and agents are located.

    state = np.zeros(6)
    state[0] = self.pacman.position.x / TILEWIDTH
    state[1] = self.pacman.position.y / TILEHEIGHT
    state[2] = self.blinky.position.x / TILEWIDTH
    state[3] = self.ghosts.blinky.position.y / TILEHEIGHT
    state[4] = self.ghosts.pinky.position.x / TILEWIDTH
    state[5] = self.ghosts.pinky.position.y / TILEHEIGHT

### 3.2.2 Normalization

The normalization process looks something like this:

state[0] = state[0] / 28

state[1] = state[1] / 36

state[2] = state[2] / 28

state[3] = state[3] / 36

state[4] = state[4] / 28

state[5] = state[5] / 36

We divide by the number of vertical and horizontal cells

### 33 Reward Function

The reward is linear if the sum of the BFS distances of the ghosts from pacman is less than 20, otherwise it is quadratic

In addition, there are some penalties and bonuses:

- **Bonus**: If pacman dies, the reward is 100

- **Penalty**: If the game is not finished, malus of 1, if the game is finished and pacman is alive, the malus is 100, also for each stationary ghost the malus is 50

Code:

blinkyDistance = game.cells.getBFSDistance(game.ghosts.blinky.cell, game.pacman.cell)

pinkyDistance = game.cells.getBFSDistance(game.ghosts.pinky.cell, game.pacman.cell)

# Treatment of invalid distances

if pinkyDistance == -1:

pinkyDistance = 20 # 20 is the starting distance

if blinkyDistance == -1:

blinkyDistance = 20 # 20 is the starting distance

# Calculating the total distance

distance = pinkyDistance + blinkyDistance

# Calculating the reward based on the distance

if distance < 20:

reward = 20 - distance

else:

reward = -(distance\*\2)/100

# Penalty if the ghosts did not move

if selfpinky.lastPosition and self.pinky.position == self.pinky.lastPosition:

reward -= 50

if self.blinky.lastPosition and self.blinky.position == self.blinky.lastPosition:

reward -= 50

# Update last positions of ghosts

self.blinky.lastPosition = self.blinkyposition

 self.pinky.lastPosition = self.pinky.position

# Handle reward if pacman dies

if game.pacmandead:

reward = 100

game.pacman.dead = False

terminated = True

else:

if not terminated:

reward -= 1

else:

reward -= 100

reward = torch.tensor([reward], device=device)

### 3.4 Optimization and Parameters

A diagram of the changes during the optimization phase

![Exel](https://github.com/Whatar/GhostsML/blob/main/imgs/exel.png)

## 4Implementation

### 4.1 Libraries and Tools Used

#### 4.1.1 Pacman Code

Pacman Code is a library developed by [Jonathan Richards](https://www.youtube.com/@jonathanrichards7969) that I used as the basis for the Pacman game I thank him profusely for his generosity in offering a complete overview of the code on his [site](https://pacmancode.com).

#### 4.1.2 Libraries

- Python 3.10.7

- PyTorch 2.0.1

- matplotlib 3.7.1

- numpy 1.23.5

### 4.2 Code Description

The relevant parts are

**run.py** the main process from which the various modules are called to build the environment, initialize pacman and ghosts, and start the game

**pacman.py** the pacman management module, commandable by either a player or Ai heuristic with 4 levels of strength:

- pacman lv1: Seeks the cell farthest from the two ghosts and sets it as its own target, every 3 seconds it changes targets (this slow update often forces it to clash against ghosts)

- pacman lv2:Optimized for the fastest pellet collection possible, ignoring ghosts, often able to win against original ghosts simply because of the large amount of turns performed at intersections

- pacman lv3: Able to flee ghosts from the moment they approach, but limited by a virtual resource (called mana), with a 5-second rechargeCurrently configured to be able to escape for 32 frames before having to recharge mana

- pacman lv4: Has no mana limit

 **ghosts.py** the ghost management module and the DQN algorithm. The two most relevant functions are complete_training, where the reward calculation is performed and the new transition data added to memory, and optimize_model, where the gradient and target_network update is calculated.

### 4.3 Problems and Solutions

**Asynchrony between Game and AI**: A significant problem that has been addressed is the asynchrony between the game time in Pac-Man and the AI update Initially, we were trying to update the target_network after a game update subsequent to taking each action, but this created synchronization problems, since a single game update can correspond to 1 frame, while actual changes in game state occur over longer times.A diagram of the problem and the solution:

Before the change:

[ **ACTION** | **LEARN** | FRAME | FRAME | FRAME | FRAME | **ACTION** | **LEARN**]

After the change:

[ **ACTION** | FRAME | FRAME | FRAME | FRAME | **LEARN** | **ACTION** ]

 Basically, the game situation just prior to the time when another action needs to be performed is analyzed.

### 5 Results and evaluation

A graph representing the average total reward per agent episode during training, over 100 games, against pacman lv3 In gray the variance.

![reward_lv2](https://github.com/Whatar/GhostsML/blob/main/imgs/ghosts_nn_10runs.png)

Against pacman lv1, you can see that variance is much lower

![reward_lv2](https://github.com/Whatar/GhostsML/blob/main/imgs/ghosts_nn_easy.png)

Some graphs showing the results of various configurations, representing pacman's score over 100 games

Pacman lv1: BLUE
Pacman lv2: ORANGE
Pacman lv3: GREEN
Pacman lv4: RED

The 4 levels of pacman against the original ghosts:

![original_ai](https://github.com/Whatar/GhostsML/blob/main/imgs/score_original.png)

The 4 levels of pacman against the DQN ghosts

![original_ai](https://github.com/Whatar/GhostsML/blob/main/imgs/score_AI.png)

the 4 levels of pacman against an experimental version with 4 ghosts (where there are two DQNs, one for blinky and pinky, one for the other two ghosts)

![original_ai](https://github.com/Whatar/GhostsML/blob/main/imgs/score_4Ghosts.png)

### 6 Future considerations and conclusion

#### 6.1 AlternativeTechniques

 One alternative technique is the Double Deep Q Network (DDQN) technique, which is discussed extensively in this [paper](https://arxiv.org/abs/1509.06461), which is useful for stabilizing training and avoiding overfitting, which is actually a problem encountered, when during the training of the ghosts against level 4 pacman, given the difficulty in capturing the target, the ghosts often preferred to settle for jamming it in a small section of the maze in order to maintain a small distance, but without actually eating it

 However, it is not ruled out that instability could be decreased by more careful engineering of the reward, for example by trying to identify traits that characterize the decision not to get too close to pacman (such as moving in the opposite direction), and penalize them.

#### 6.2 Implications and future applications

We could see how complex it is to debug such a machine learning application, especially in a custom environment that does not follow gymnasium guidelines This is limiting because gymnasium's open source libraries are often dated and no longer compatible with current versions of gym.

 For this reason it is believed, that at the beginning of such a project, it is worthwhile to invest some time in building solid logging tools that can show what is really happening during the training.

 There are also some interesting experiments that could be started from this project. For example, it might be very interesting to look for a reward function that could allow ghosts to chase pacman but not necessarily catch him too early, to allow him to move freely for a certain period at the beginning of the game, and then increase the pressure.

 This turned out to be particularly tricky, considering the ease with which, the DQN algorithm of this design tends to crash on local minima, so it is hard to imagine how it could, during the course of a game, discover access to a new source of reward that was not initially present.

 Alternatively, one could add a malus to capture depending on the number of pellets in play. Clearly this would require adding this information to the input of the DNQN (although, again a DNQ would probably not be suitable for this more complex scenario).

 Finally, one could configure the reward to be maximum at a certain distance from the pacman, and decrease this distance every x frames, but even this makes the training more unstable.

 Still, it turns out to be an interesting theory, which could actually allow the ghosts to apply gradually more pressure to the player. 
