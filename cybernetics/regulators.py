import numpy as np
import pandas as pd
from mpi4py import MPI
import pyspark

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
    def __init__(self,goals=[], game_size=(1,1), epochs=5, ran_range=10, game=None, skweez=None, history=dict(), game_df=None, parallelize=False, mpi=False, pyspark=False, comm=False, rank=False):
        Regulator.__init__(self)
        self.goals = goals
        self.game_size = game_size
        self.epochs = epochs
        self.ran_range = ran_range
        self.game = game
        self.skweez = skweez
        self.history = history
        self.game_df = game_df
        
        # For parallelization
        self._parallelize = parallelize
        self._mpi = mpi
        self._pyspark = pyspark
        
        # For MPI
        self._comm = comm
        self._rank = rank
        
        # Parallelization logic.
        if self._parallelize:
            print('Parallelization engaged.')
            if self._mpi:
                print('MPI engaged.')
                print('I am rank {}.'.format(self._rank))
                #self._comm = MPI.COMM_WORLD
                #self._rank = self._comm.Get_rank()
            elif self._pyspark:
                print('Pyspark engaged.')
        
        
    def create_game(self):
        '''
        This function takes a size tuple attribute and creates a game matrix of size=(x,y).
        
        It also checks for parallelization and acts accordingly.

        '''
        
        if self._mpi:
            
            if self._rank == 0:
                print('MPI logic creating game only for root rank.')
                
                # Calculate game only once (in root rank)
                game_matrix = np.random.randint(self.ran_range, size=self.game_size)
                print('Created game matrix of size {}.  Returning game dataframe...'.format(self.game_size))
                rows = [i+1 for i in range(self.game_size[0])]
            
            else:
                game_matrix = None
                rows = None
                
            # Broadcast data from root
            game_matrix = self._comm.bcast(game_matrix, root=0)
            rows = self._comm.bcast(rows, root=0)
            
            # Set object game attribute for each rank
            self.game = game_matrix

            # Set object game_df attribute for each rank
            self.game_df = pd.DataFrame(data = game_matrix, columns=[i+1 for i in range(game_matrix.shape[1])], index=rows)

            return self.game_df
            
        else:
            print('No ranks.')
            game_matrix = np.random.randint(self.ran_range, size=self.game_size)
            print('Created game matrix of size {}.  Returning game dataframe...'.format(self.game_size))
            rows = [i+1 for i in range(self.game_size[0])]

            # Set object game attribute
            self.game = game_matrix

            # Set object game_df attribute
            self.game_df = pd.DataFrame(data = game_matrix, columns=[i+1 for i in range(game_matrix.shape[1])], index=rows)

            return self.game_df

    def environment_play(self,dist):
        '''
        This function takes a game and a distribution and returns an environmental "play" according to the distribution over the environment's actions in the game matrix.
        '''
        return np.random.choice(self.game_df.index, size=1, p=dist).item()

    def regulator_action(self,probs):
        '''
        This function takes a game and a probability distribution and returns a regulator action to try to "control" the environmental action ("disturbance").
        '''
        return np.random.choice(self.game_df.columns, size=1, p=probs).item()

    def prob_calc(self,regulator_dict):
        '''
        This function takes a regulator dictionary (composed of actions and pre-probabilities, i.e. a "Polya urn" filled with "action-balls") and converts the urn into probabilities.
        '''
        sum_reg = np.array(sum([regulator_dict[i] for i in regulator_dict]))
        return np.array([regulator_dict[i]/sum_reg for i in regulator_dict])

    def squeeze(self,regulator_dict, urn_list):
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

    def update(self,regulator_dict,action,out,goals,urn_list,skweez=False):
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
            # REMOVED PRINTS FOR LARGE TESTING
            #print("success!")
            #print('now we need to recalculate the probabilities according to the reinforced urn')
        elif skweez:
            print('fail: squeezing.')
            squeeze(regulator_dict, urn_list)
        return success


        
        
    def train(self):
        
        print('self.game:',self.game)
        
        # Get game.    
        if self.game is not None:
            print('Game given')
            print('setting game size to self.game.shape')
            self.game_size = self.game.shape
        elif self.parallelize:
            if self.mpi:
                print('mpi in training')
        else:
            print('Creating game of size', self.game_size, 'and range', self.ran_range)
            #self.game = self.create_game(self.game_size,self.ran_range)
            self.game = self.create_game()
            
            
        print(self.game)
        urn = np.random.randint(100, size=len(self.game_df.columns))
        probs = np.array([(i/sum(urn)) for i in urn])
        #print("probs:",probs)
        
        regulator = dict(zip(self.game_df.columns,urn))
        print("regulator:",regulator)
        
        dist = np.random.dirichlet(alpha=self.game_df.index)
        print('printing dist: {}'.format(dist))
        
        successes = 0
        i=1
        
        history = self.history
        
        while i <= self.epochs:
            
            if self.goals == []:
                print('No goals. Breaking.')
                break

            # Environment chooses play.
            play = self.environment_play(dist)
            #print('printing play: {}'.format(play))

            # Regulator chooses action.
            action = self.regulator_action(probs)

            # Compute state of the world that is output (index in game matrix)
            out = self.game_df.loc[play,action]
            #print("out:",out)

            # Update regulator.
            successes += self.update(regulator,action,out,self.goals,urn,skweez=self.skweez)
            
            # Print only every 5 epochs for brevity.
            # REMOVED FOR LARGE EPOCH TESTING.
            #if i % 5 == 0:
            #    print("Epoch: ",i)
            #    print("successes per epoch:",successes / i)

            # Recalculate regulator probabilities.
            probs = self.prob_calc(regulator)
            #print("updated probs:",probs)

            # Log history dict for plotting.
            history[i] = successes / i
            
            #Increment i.
            i += 1
        return print('Trained regulator:', regulator)
    
