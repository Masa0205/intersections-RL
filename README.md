# Reinforcement Learning for traffic control(Intersections)

## Introduction
Here is my source code for training to choice priority lane in intersections without traffic light. By using QL and DQN.  
 I used SUMO for simulation and TraCI for data acquisition, and Pytorch for NN construction and training.
 Up to now, I have successfully learned to use QL and DQN at single intersections. 
I will add Double-DQN, Dueling-DQN, MQRL...
 ## How to use the code
- **Train or Test** QL agent by running `python QL.py`
- **Train or Test** DQN agent by running `python DQN.py`  
 You will be asked for the required information, so enter each of the following. 
(using mode, the number of vehicles, episode...) 
simulation can be viewed when you choice the test mode. Also, after training is completed, reawrd and teleport detected graph are outmaticaly saved.
