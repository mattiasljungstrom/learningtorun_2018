

# README - DDPG

An asynchronous DDPG Reinforcement Learning algorithm with multiple Actor-Critic pairs.

Fourth place in NeurIPS 2018: AI for Prosthetics Challenge: 
https://www.crowdai.org/challenges/neurips-2018-ai-for-prosthetics-challenge/leaderboards


## Initial code based on

* DDPG base code and farm/farmer/noise code is based on: https://github.com/ctmakro/stanford-osrl
* Memory code based on baseline code: https://github.com/openai/baselines/blob/master/baselines/ddpg/memory.py


## New Features

* All code optimized and rewritten to use only tensorflow 1.8.
* Added training and inference with multiple actor-critic pairs.
* New observation setup in observation_2018.py
* New reward shaping in reward_mod.py
* Simplified code to remove dependencies.
* Added new more light-weight visualization with more statistics.
* Added evaluation code in test_multi.py and test_env.ipynb


## Dependencies

conda create -n opensim -c kidzik opensim python=3.6.1  
source activate opensim

(if you don't have git)  
sudo apt-get install git

conda install -c conda-forge lapack  
pip install git+https://github.com/stanfordnmbl/osim-rl.git

pip install tensorflow (==1.8.0)  
pip install ipython  
pip install Pyro4  

(optional)  
pip install matplotlib  
pip install pymsgbox  


## Run in two different consoles

1> python farm.py

2> python -m IPython -i ddpg_multi_tf.py  
2> r(20000)

Use test_multi.py and test_end.ipynb to evaluate results.

On Windows python leaves processes, kill them with:  
taskkill /F /IM python.exe /T


## Notes

Farm code could be improved and is slightly buggy, leaves processes, etc.  
Decided to focus on improving model and training instead.



