# Run from terminal in env with commands like "mpirun -n 4 python mpi_regulator_test.py"
from cybernetics.regulators import Ashby
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

ash = Ashby(epochs=10000, parallelize=True, mpi=True, comm=comm, rank=rank)

ash.game_size = (10,10)

ash.goals.append(3)

#ash.parallelize = True
#ash.mpi = True

#ash.create_game()

ash.train()

