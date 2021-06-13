# Run from terminal in env with commands like "mpirun -n 4 python mpi_regulator_test.py"
from cybernetics.regulators import Ashby

ash = Ashby(epochs=10000)

ash.game_size = (10,10)

ash.goals.append(3)

ash.parallelize = True
ash.mpi = True

ash.train()

