#### b. Write a small script ```mpi_sum.py``` which calculates the sum over all ranks and prints the result from the process with rank 0.
#Hint: Have a look at the tutorials from the mpi4py documentation 
# page: [https://mpi4py.readthedocs.io/en/stable/tutorial.html](https://mpi4py.readthedocs.io/en/stable/tutorial.html)

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Get process rank
size = comm.Get_size()  # Get total number of processes

# Each process gets its own value (for demonstration, we use its rank)
local_value = rank + 1  # Each rank contributes its own value

# Sum up all local values across all processes
total_sum = comm.reduce(local_value, op=MPI.SUM, root=0)

# Print the result from the root process (rank 0)
if rank == 0:
    print(f"Total sum across all ranks: {total_sum}")
