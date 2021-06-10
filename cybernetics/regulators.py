import numpy as np
import pandas as pd

class Regulator():
    '''
    Class defining general attributes and behaviors of regulator objects.
    '''
    
    def __init__(self):
        # Object of regulator to achieve.
        self.goals = []
    
    def update_protocol(self):
        pass
        
    def fit(self):
        # Process of improving performance towards regulatory goal.
        pass
        
        
class Ashby(Regulator):
    '''
    Subclass of Regulator class to illustrate Ashby's basic game theoretic foundation for cybernetic regulators.
    '''
    #from cybernetics.train import train
    
    #def __init__(self,goals,game_size,epochs,ran_range,game):
    def __init__(self):
        Regulator.__init__(self)
        self.goals = []
        self.game_size = (1,1)
        self.epochs = 5
        self.ran_range = 10
        self.game = None
        self.skweez = None
        self.history = dict()
        
        
    def train(self):
        from cybernetics.cybernetics import create_game
        from cybernetics.cybernetics import environment_play
        from cybernetics.cybernetics import regulator_action
        from cybernetics.cybernetics import prob_calc
        from cybernetics.cybernetics import squeeze
        from cybernetics.cybernetics import update
        
        print('self.game:',self.game)
        
            
        if self.game is not None:
            print('Game given')
            print('setting game size to self.game.shape')
            self.game_size = self.game.shape
        else:
            print('Creating game of size', self.game_size, 'and range', self.ran_range)
            self.game = create_game(self.game_size,self.ran_range)
            
            
        print(self.game)
        urn = np.random.randint(100, size=len(self.game.columns))
        probs = np.array([(i/sum(urn)) for i in urn])
        #print("probs:",probs)
        regulator = dict(zip(self.game.columns,urn))
        print("regulator:",regulator)
        dist = np.random.dirichlet(alpha=self.game.index)
        successes = 0
        i=1
        
        history = self.history
        
        while i <= self.epochs:

            # Environment chooses play.
            play = environment_play(self.game,dist)

            # Regulator chooses action.
            action = regulator_action(self.game,probs)

            # Compute state of the world that is output (index in game matrix)
            out = self.game.loc[play,action]
            #print("out:",out)

            # Update regulator.
            successes += update(regulator,action,out,self.goals,urn,skweez=self.skweez)
            
            # Print only every 5 epochs for brevity.
            # REMOVED FOR LARGE EPOCH TESTING.
            #if i % 5 == 0:
            #    print("Epoch: ",i)
            #    print("successes per epoch:",successes / i)

            # Recalculate regulator probabilities.
            probs = prob_calc(regulator)
            #print("updated probs:",probs)

            # Log history dict for plotting.
            history[i] = successes / i
            
            #Increment i.
            i += 1
        return print('Trained regulator:', regulator)
    
