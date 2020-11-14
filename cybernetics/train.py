import numpy as np
import pandas as pd

from .cybernetics import create_game
from .cybernetics import environment_play
from .cybernetics import regulator_action
from .cybernetics import prob_calc
from .cybernetics import squeeze
from .cybernetics import update

def train(game_size,goals,epochs,ran_range=10,skweez=None):
    '''
    The train function is the primary function for the frontend user of this package.

    The user must provide a game_size=(x,y), where x,y are integers (not tested for very large sizes!).
    
    The user may provide a list of multiple goals=[i,j,k,...].

    The user must specify the number of epochs (environment plays x regulator actions x updates) they wish to train the regulator.
    
    The ran_range of the random game matrix defaults to 10.
    '''
    game = create_game(game_size,ran_range)
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
        successes += update(regulator,action,out,goals,urn,skweez=skweez)
        print("successes per epoch:",successes / i)
        
        # Recalculate regulator probabilities.
        probs = prob_calc(regulator)
        #print("updated probs:",probs)
        
        
        #Increment i.
        i += 1
    return regulator

