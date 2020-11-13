import numpy as np
import pandas as pd

def create_game(size):
    game_matrix = np.random.randint(10, size=size)
    rows = [i+1 for i in range(len(game_matrix))]
    print(rows)
    return pd.DataFrame(data = game_matrix, columns=[i+1 for i in range(game_matrix.shape[1])], index=rows)

def environment_play(game,dist):
    '''
    
    '''
    return np.random.choice(game.index, size=1, p=dist).item()

def regulator_action(game,probs):
    return np.random.choice(game.columns, size=1, p=probs).item()

def prob_calc(regulator_dict):
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

def update(regulator_dict,action,out,goal,urn_list,skweez=False):
    success = 0
    if out == goal:
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

def train(game_size,goal,epochs,skweez):
    game = create_game(game_size)
    print(game)
    urn = np.random.randint(100, size=len(game.columns))
    probs = np.array([(i/sum(urn)) for i in urn])
    #print("probs:",probs)
    regulator = dict(zip(game.columns,urn))
    print("regulator:",regulator)
    dist = np.random.dirichlet(alpha=game.index)
    successes = 0
    i=1
    while i <= epochs:
        
        print("Epoch: ",i)
        
        # Environment chooses play.
        play = environment_play(game,dist)
        
        # Regulator chooses action.
        action = regulator_action(game,probs)
        
        # Compute state of the world that is output (index in game matrix)
        out = game.loc[play,action]
        #print("out:",out)
        
        # Update regulator.
        successes += update(regulator,action,out,goal,urn,skweez=skweez)
        print("successes per epoch:",successes / i)
        
        # Recalculate regulator probabilities.
        probs = prob_calc(regulator)
        #print("updated probs:",probs)
        
        
        #Increment i.
        i += 1
    return regulator

