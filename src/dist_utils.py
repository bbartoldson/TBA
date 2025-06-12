import os
import socket
from mpi4py import MPI
import torch
import cupy

def init_distributed_env(accelerate_ranks=None, accelerate_kwargs=None):
    """
    - Initializes MPI.
    - Configures environment variables so that the ranks in `accelerate_ranks` 
      can form a smaller "Accelerate world" for multi-GPU training.
    - Returns:
        comm, world_rank, world_size, accelerator
    """
    from accelerate import Accelerator
    
    if accelerate_kwargs is None:
        accelerate_kwargs = {}
        
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    if accelerate_ranks is None:
        accelerate_ranks = [0]
    
    # Ensure accelerate_ranks is sorted for consistent indexing
    accelerate_ranks = sorted(accelerate_ranks)
    
    # Rank 0 determines master address and port, then broadcasts to all
    if world_rank == 0:
        assert world_rank in accelerate_ranks, "Rank 0 must be in accelerate ranks"
        master_addr = socket.gethostname()
        master_port = next_free_port(29500)  # Use same port for all accelerate ranks
        coordination_info = {'master_addr': master_addr, 'master_port': master_port}
    else:
        coordination_info = None
    
    # Broadcast coordination info to all ranks
    coordination_info = comm.bcast(coordination_info, root=0)
    master_addr = coordination_info['master_addr']
    master_port = coordination_info['master_port']
    
    accelerator = None
    
    if world_rank in accelerate_ranks:
        # Configure environment for accelerate ranks
        local_accel_rank = accelerate_ranks.index(world_rank)
        local_rank = local_accel_rank  # For multi-GPU: rank 0->GPU 0, rank 1->GPU 1, etc.
        
        # Set up distributed training environment variables
        os.environ["WORLD_SIZE"] = str(len(accelerate_ranks))
        os.environ["RANK"] = str(local_accel_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        
        print(f"Accelerate rank {world_rank}: WORLD_SIZE={len(accelerate_ranks)}, "
              f"RANK={local_accel_rank}, LOCAL_RANK={local_rank}, "
              f"MASTER_ADDR={master_addr}, MASTER_PORT={master_port}", flush=True)
        
        # Create the Accelerator for this rank
        try:
            accelerator = Accelerator(**accelerate_kwargs)
            print(f"Accelerator created successfully for rank {world_rank}", flush=True)
        except Exception as e:
            print(f"Failed to create accelerator for rank {world_rank}: {e}", flush=True)
            raise
            
    else:
        # Configure environment for non-accelerate ranks (searchers)
        # Isolate them from the accelerate world
        searcher_rank = world_rank - len(accelerate_ranks)  # 0-indexed among searchers
        isolated_port = next_free_port(30000 + searcher_rank * 10)  # Different port range
        
        os.environ["WORLD_SIZE"] = "1"  # Each searcher is in its own "world"
        os.environ["RANK"] = "0" 
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] =  socket.gethostname() #master_addr  # Can use same addr
        os.environ["MASTER_PORT"] = str(isolated_port)  # But different port
        
        print(f"Searcher rank {world_rank}: Isolated from accelerate world, "
              f"using port {isolated_port}", flush=True)
    
    return comm, world_rank, world_size, accelerator

def next_free_port(start_port=29500):
    """Find the next available port starting from start_port"""
    import socket
    
    for port in range(start_port, start_port + 100):  # Try 100 ports
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find free port starting from {start_port}")


def broadcast_weights_orig(model, comm: MPI.Comm, root_mpi_rank: int, role: str):
    """
    Broadcast all of `model`'s parameters from `root_mpi_rank`
    to every other MPI rank. If you're running on GPU,
    we must temporarily copy parameters to CPU for the broadcast.

    TODO: make this faster for larger-scale models (e.g. >=7B params)
    """
    world_rank = comm.Get_rank()
    for name, param in model.named_parameters():
        # conversion from bfloat needed for Bcast, it's a no-op if float32 already
        param_cpu = param.data.to(torch.float32).cpu().numpy() 
        # Broadcast in-place from root
        comm.Bcast(param_cpu, root=root_mpi_rank)
        # Non-root ranks copy data back into model param
        if world_rank != root_mpi_rank and role!= 'trainer':
            # Convert back to original dtype and device
            param.data = torch.from_numpy(param_cpu).to(
                dtype=param.data.dtype,
                device=param.data.device
            )


import torch
import pickle
import os
from pathlib import Path
import time

def broadcast_weights(model, comm, root_mpi_rank, role ):
    """
    Broadcast weights via Lustre filesystem.
    Requires proper Lustre striping for best performance.
    """
    lustre_path = '/p/lustre5/bartolds/tmp'
    rank = comm.Get_rank()
    world_size = comm.Get_size() 
    root_rank = root_mpi_rank

    lustre_dir = Path(lustre_path)
    lustre_dir.mkdir(exist_ok=True)
    
    # Use unique identifiers to avoid conflicts
    job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    weights_file = lustre_dir / f"weights_{job_id}_{rank}.pt"
    ready_file = lustre_dir / f"ready_{job_id}"


    job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    weights_file = lustre_dir / f"weights_{job_id}.pt"
    ready_file = lustre_dir / f"ready_{job_id}"
    
    try:
        if rank == root_rank:
            print(f"[rank {rank}] Writing weights...")
            
            # Save as a list of tensors to avoid name mismatches
            param_list = [param.cpu().clone() for param in model.parameters()]
            torch.save(param_list, weights_file)
            
            # Force file system sync
            os.sync()
            time.sleep(0.1)
            ready_file.touch()
            os.sync()
            
            print(f"[rank {rank}] Weights written")
        
        # All ranks wait
        comm.Barrier()
        
        if role!='trainer':
            print(f"[rank {rank}] Waiting for weights...")
            
            # Wait for ready signal
            timeout = 60
            start_time = time.time()
            
            while not ready_file.exists():
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Rank {rank}: Timeout waiting for weights file")
                time.sleep(0.1)
            
            print(f"[rank {rank}] Loading weights...")
            
            # Load weights as list
            param_list = torch.load(weights_file, map_location='cpu')
            
            # Copy parameters in order (avoiding name issues)
            for param, saved_param in zip(model.parameters(), param_list):
                param.data.copy_(saved_param.to(param.device))
            
            print(f"[rank {rank}] Weights loaded")
        
        # Final barrier
        comm.Barrier()
        
    finally:
        # Cleanup
        if rank == root_rank:
            try:
                weights_file.unlink(missing_ok=True)
                ready_file.unlink(missing_ok=True)
            except:
                pass
