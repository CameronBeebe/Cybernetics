An implementation of the game theoretic foundation of cybernetic systems theory as outlined by W. R. Ashby.

Cybernetics, as outlined by Ashby, provides a foundational framework to understand the effectiveness of common machine learning techniques.  In particular, connectionist artificial neural networks can be understood as cybernetic regulators which are error-controlled, or which learn through reinforcement.

At the moment there is just basic functionality of a toy model to illustrate the basic concepts, it is limited in its applicability (and performance).

The ultimate goal of this package is to provide a common class structure for regulatory objects.

TO DO:

1.  Ability to input game, not just game_size (so can compare multiple runs on same game).  

DONE, NEEDS LOGIC CHECK / TEST in train.py AND DOCSTRING UPDATED

2.  Game analysis: check goals against random game composition (i.e. calculate availability of success to compare to regulator performance)

STARTED TESTING.  

2a.  Create visualization for performance and comparison to potential performance.

3.  Clean up verbosity for notebook (inline accuracy meter?)

4.  Create options for different forms of reinforcement.

5.  Add normalization function to urn (value in regulator dictionary)?

6.  Prototype class structures for regulator objects in general.  

BEGAN

7.  Separate classes into separate .py file

8.  Parallelize: put logic into train(?) file for both MPI and pyspark

9.  Integrate everything into classes, remove old files

10.  Add quality control: check to make sure something in list of goals are in game so trivially unproductive games are not run.  Be able to accept a list of goals

11.  Create (at least in MPI version) a per-rank goal functionality.  That is, each rank has its own goal, and depending on user input the functionality checks for a rank that has trained for that goal and uses that rank's regulator to spit out a response.

12.  Create a method that just spits out one single action in response to an environmental disturbance (row).  I.e. just one train forward pass.