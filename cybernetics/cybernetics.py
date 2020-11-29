import numpy as np
import pandas as pd

def create_game(size, ran_range=10):
    '''
    This function takes a size tuple and creates a game matrix of size=(x,y).
    
    ran_range is to be passed from frontend use of train.
    '''
    game_matrix = np.random.randint(ran_range, size=size)
    print('Created game matrix of size', size)
    rows = [i+1 for i in range(size[0])]
    return pd.DataFrame(data = game_matrix, columns=[i+1 for i in range(game_matrix.shape[1])], index=rows)

def environment_play(game,dist):
    '''
    This function takes a game and a distribution and returns an environmental "play" according to the distribution over the environment's actions in the game matrix.
    '''
    return np.random.choice(game.index, size=1, p=dist).item()

def regulator_action(game,probs):
    '''
    This function takes a game and a probability distribution and returns a regulator action to try to "control" the environmental action ("disturbance").
    '''
    return np.random.choice(game.columns, size=1, p=probs).item()

def prob_calc(regulator_dict):
    '''
    This function takes a regulator dictionary (composed of actions and pre-probabilities, i.e. a "Polya urn" filled with "action-balls") and converts the urn into probabilities.
    '''
    sum_reg = np.array(sum([regulator_dict[i] for i in regulator_dict]))
    return np.array([regulator_dict[i]/sum_reg for i in regulator_dict])

def squeeze(regulator_dict, urn_list):
    '''
    This update function takes two arguments.
    
    regulator_dict: a regulator defined as a dictionary of key labels (plays or columns) 
    and integer values (from a distribution or urn).
    
    urn_list: a list of integers interpreted as the composition of a Polya urn.
    
    The function calculates the mean of the urn composition, and compares each value
    in the regulator_dict with the mean.  
    
    The resulting regulator_dict is updated by incrementing values smaller than the
    mean, and decrementing values greater than the mean.
    
    '''
    mean = np.mean(urn_list)
    for i in regulator_dict:
        if regulator_dict[i] >= mean:
            regulator_dict[i] -= 1
            #print('squeeze down:', regulator_dict[i])
        else:
            regulator_dict[i] += 1
            #print('squeeze up:', regulator_dict[i])

def update(regulator_dict,action,out,goals,urn_list,skweez=False):
    '''
    This update function takes as arguments a regulator, its action, the output in the game matrix of that action, the regulator's goals, and an "urn" composing the regulator's disposition toward actions.  It compares the state of the world output (entry in game matrix) as a result of the action with the regulator's goals.  

    A success variable is set to 1 if the output is in alignment with the regulator's goals.

    Returns success (0 or 1).
    '''
    success = 0
    if out in goals:
        #print(action)
        #print("success: reinforced the regulator's action", action, "from", regulator_dict[action], "to", regulator_dict[action]+len(regulator_dict))
        regulator_dict[action] += len(regulator_dict)**(1/2)
        success += 1
        print("success!")
        #print('now we need to recalculate the probabilities according to the reinforced urn')
    elif skweez:
        print('fail: squeezing.')
        squeeze(regulator_dict, urn_list)
    return success

