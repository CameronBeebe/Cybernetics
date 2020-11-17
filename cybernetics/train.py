import numpy as np
import pandas as pd

from .cybernetics import create_game
from .cybernetics import environment_play
from .cybernetics import regulator_action
from .cybernetics import prob_calc
from .cybernetics import squeeze
from .cybernetics import update

def train(game_size=None,goals=[],epochs=5,ran_range=10,game=None,skweez=None):
    '''
    The train function is the primary function for the frontend user of this package.

    The user must provide a game_size=(x,y), where x,y are integers (not tested for very large sizes!).
    
    The user may provide a list of multiple goals=[i,j,k,...].

    The user must specify the number of epochs (environment plays x regulator actions x updates) they wish to train the regulator.
    
    The ran_range of the random game matrix defaults to 10.
    '''
    if (game is not None) and (game_size==None):
        train_game = game
    else:
        train_game = create_game(game_size,ran_range)
    print(train_game)
    urn = np.random.randint(100, size=len(train_game.columns))
    probs = np.array([(i/sum(urn)) for i in urn])
    #print("probs:",probs)
    regulator = dict(zip(train_game.columns,urn))
    print("regulator:",regulator)
    dist = np.random.dirichlet(alpha=train_game.index)
    successes = 0
    i=1
    while i <= epochs:
        
        print("Epoch: ",i)
        
        # Environment chooses play.
        play = environment_play(train_game,dist)
        
        # Regulator chooses action.
        action = regulator_action(train_game,probs)
        
        # Compute state of the world that is output (index in game matrix)
        out = train_game.loc[play,action]
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

