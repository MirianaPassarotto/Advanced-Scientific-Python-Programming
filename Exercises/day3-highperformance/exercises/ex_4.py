
## 4. MPI parallelization

#### a. Write a simple MPI script ```mpi_ranks.py``` that prints the rank of the different processes when running 

#mpirun python mpi_ranks.py


from mpi4py import MPI 
comm = MPI.COMM_WORLD 
rank = comm.Get_rank() 
print("hello world from process ", rank)