from axonn import axonn as ax
import os
from typing import Any, Dict, Optional
import torch
import numpy as np
import random
from axonn.intra_layer import clip_grad_norm_, sync_gradients, optimize_communication
from contextlib import contextmanager, nullcontext
from functools import partial
from lightning.fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors

def rank_prefixed_message(msg, rank):
    return f"Rank={rank} | {msg}"

def divide(a, b):
    assert a%b == 0
    return a // b

class AxoNNFabric():
    def __init__(self, init_method="slurm", tensor_parallel_grid=[1,1,1], loggers=None, training=True):
        if init_method not in ["slurm"]:
            raise ValueError("AxoNNFabric can only initialize from slurm as of now")
        
        if init_method == "slurm":
            rank = int(os.getenv("SLURM_PROCID", "0"))
            world_size = int(os.getenv("SLURM_NTASKS", "1"))
            torch.distributed.init_process_group(rank=rank, 
                                                 world_size=world_size,
                                                 backend='nccl')

        G_intra_r, G_intra_c, G_intra_d = tensor_parallel_grid
        tensor_parallel_size = G_intra_r * G_intra_d * G_intra_c
        data_parallel_size = divide(self.world_size, tensor_parallel_size)

        loggers = loggers if loggers is not None else []
        self._loggers = loggers if isinstance(loggers, list) else [loggers] 

        self.print(f"> Data Parallel Size = {data_parallel_size}")
        self.print(f"> Tensor Parallel Size = {tensor_parallel_size}")
        self.print(f"> Tensor Parallel Grid = {G_intra_r} x {G_intra_c} x {G_intra_d}")

        ax.init(
            G_inter=1,
            G_data=data_parallel_size,
            G_intra_r=G_intra_r,
            G_intra_c=G_intra_c,
            G_intra_d=G_intra_d,
        )

        self.local_rank = self.global_rank % torch.cuda.device_count()
        self.do_depth_parallel_grad_sync = True
        torch.cuda.set_device(self.local_rank)

        self._unwrapped_model = None
        self.training = training

    def run_bw_test(self):
        SZ = int(16 * 2048 * 4096)
        msg = torch.rand(SZ, 1, dtype=torch.bfloat16, device="cuda")
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        times = []
        for _ in range(20):
            st.record()
            torch.distributed.all_reduce(msg)
            en.record()
            torch.cuda.synchronize()
            times.append(st.elapsed_time(en))
            #print(times[-1]/2)

        size = SZ * 2  ## bytes
        g = torch.distributed.get_world_size()
        bw = 2 * (g - 1) / g * size / 1e9 / np.mean(times[-10:]) * 1000
        if torch.distributed.get_rank() == 0:
            print(
                f"All-reduce bus bw for {g} GPUs is {bw:.3f} GBPS for message size {size/1e9:.3f} GB"
            )
            print(f"time = {np.mean(times[-10:])} ms")

    def launch(self):
        pass

    def print(self, msg, ranks=[0]):
        if torch.distributed.get_rank() in ranks:
            print(msg, flush=True)

    def log_dict(self, metrics, step = None):
        if self.global_rank == 0:
            metrics = convert_tensors_to_scalars(metrics)
            for logger in self._loggers:
                logger.log_metrics(metrics=metrics, step=step)

    @property
    def device(self):
        return torch.device('cuda')
    
    @property
    def global_rank(self):
        return torch.distributed.get_rank()

    @property
    def world_size(self):
        return torch.distributed.get_world_size()

    @property
    def data_parallel_global_rank(self):
        return ax.config.data_parallel_rank

    @property
    def data_parallel_world_size(self):
        return ax.config.G_data

    @property
    def global_rank_for_creating_dataloader(self):
        return ax.config.G_intra_d * ax.config.data_parallel_rank + ax.config.intra_layer_depth_parallel_rank 

    @property
    def global_world_size_for_creating_dataloader(self):
        return ax.config.G_intra_d * ax.config.G_data

    @property
    def row_tensor_parallel_global_rank(self):
        return ax.config.intra_layer_row_parallel_rank

    @property
    def column_tensor_parallel_global_rank(self):
        return ax.config.intra_layer_column_parallel_rank

    @property
    def depth_tensor_parallel_global_rank(self):
        return ax.config.intra_layer_depth_parallel_rank
        
    @property
    def depth_tensor_parallel_global_world_size(self):
        return ax.config.G_intra_d

    @property
    def data_parallel_process_group(self):
        return ax.comm_handle.coll_nccl_comm 

    @property
    def dtype(self):
        return torch.bfloat16

    @property
    def logger(self):
        return self._loggers[0]

    @property
    def strategy(self):
        return "AxoNNFabric"

    @property 
    def unwrapped_model(self):
        return self._unwrapped_model

    def _validate_launched(self): 
        pass

    def setup_dataloaders(self, *args, **kwargs):
        pass

    def seed_everything(self, seed: Optional[int] = None, workers:bool = False):
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        if seed is None:
            env_seed = os.environ.get("PL_GLOBAL_SEED")
            if env_seed is None:
                seed = 0
                self.print(f"Warning: No seed found, seed set to {seed}")
            else:
                try:
                    seed = int(env_seed)
                except ValueError:
                    seed = 0
                    self.print(f"Warning: Invalid seed found: {repr(env_seed)}, seed set to {seed}")
        elif not isinstance(seed, int):
            seed = int(seed)

        if not (min_seed_value <= seed <= max_seed_value):
            self.print(f"Warning: {seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
            seed = 0

        print(rank_prefixed_message(f"Seed set to {seed}", self.global_rank))
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

        return seed

    def setup(self, model):
        # wrap DDP 
        model = model.cuda()
        if self.training:
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            bucket_cap_mb = num_params * 4

            self._unwrapped_model = model

            return torch.nn.parallel.DistributedDataParallel(model, 
                                                              device_ids=[self.local_rank], 
                                                              process_group=self.data_parallel_process_group,
                                                              bucket_cap_mb=bucket_cap_mb)
        else:
            self._unwrapped_model = model
            return model

    def setup_optimizers(self, optimizers):
        return optimizers

    def all_reduce(self, data, reduce_op="mean"):
        '''
        All reduce over data parallel groups
        '''
        op = torch.distributed.ReduceOp.AVG
        if reduce_op == "sum":
            op = torch.distributed.ReduceOp.SUM

        # I am not a 100% sure if we want to do this. This is a lightning utility that is 
        # used in the all_reduce of the fabric that converts all data into tensors and transfers it to self.device
        # We definitely need the transfer to self.device but might not need the conversion to tensors
        data = convert_to_tensors(data, device=self.device)

        for process_group in [ax.comm_handle.coll_nccl_comm, ax.comm_handle.depth_intra_layer_parallel_group]:
            torch.distributed.all_reduce(data, op=op, group=process_group)
        return data

    def all_gather_object(self, obj):
        """
        All gather for serializable python objects over data parallel groups
        """
        # FIXME this is @jwkirchenbauer crap
        dst_list = [None for _ in range(torch.distributed.get_world_size(group=self.data_parallel_process_group))]
        torch.distributed.all_gather_object(dst_list, obj, group=self.data_parallel_process_group)
        return dst_list
    
    def all_gather(self, data):
        """
        All gather over data parallel groups
        """
        if not isinstance(data, torch.Tensor):
            return self.all_gather_object(data)

        # FIXME this is @jwkirchenbauer crap
        # See above.
        data = convert_to_tensors(data, device=self.device)

        # https://github.com/Lightning-AI/pytorch-lightning/blob/main/src/lightning/fabric/utilities/distributed.py#L235
        from torch.distributed.nn.functional import all_gather

        data = data.contiguous()  # https://github.com/pytorch/pytorch/issues/73515
        gathered_data = torch.stack(all_gather(data, group=self.data_parallel_process_group))

        return gathered_data

    def barrier(self):
        torch.distributed.barrier()

    def broadcast(self, obj, src: int = 0):
        if not torch.distributed.is_initialized():
            return obj

        obj = [obj]
        torch.distributed.broadcast_object_list(obj, src)
        return obj[0]


    def backward(self, model, loss):
        loss.backward() 
        if self.do_depth_parallel_grad_sync:
            sync_gradients(model)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.depth_tensor_parallel_global_world_size)

    def clip_gradients(self, model, optimizer, max_norm):
        norm = clip_grad_norm_(model.parameters(), 
                               max_norm, 
                               norm_type=2.0)
        return norm

    def get_tensor_parallel_prefix_for_checkpoint(self):
        row_tp_rank = self.row_tensor_parallel_global_rank
        column_tp_rank = self.column_tensor_parallel_global_rank
        depth_tp_rank = self.depth_tensor_parallel_global_rank
        return f"tp_row_{row_tp_rank}_col_{column_tp_rank}_depth_{depth_tp_rank}"


    def save(self, checkpoint_file, state):
        if self.data_parallel_global_rank == 0:
            save_state_dict = {}
            for key, value in state.items():
                if key not in ["optimizer", "model"]:
                    save_state_dict[key] = value
                else:
                    save_state_dict[key] = value.state_dict()
            
            checkpoint_folder = os.path.dirname(checkpoint_file)
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            if os.path.exists(checkpoint_file):
                raise ValueError(f"Checkpoint {checkpoint_file} already exists")
            torch.save(save_state_dict, checkpoint_file)

    def load(self, resume, state):
        checkpoint = torch.load(resume)
        for key, value in checkpoint.items():
            if key not in ["optimizer", "model"]:
                state[key] = value
            else:
                if key == 'model':
                    state[key].load_state_dict(value)
                else:
                    state[key].load_state_dict(value)

    @contextmanager
    def no_backward_sync(self, model, enabled=True):
        old_require_backward_grad_sync =  model.require_backward_grad_sync
        old_do_depth_parallel_grad_sync = self.do_depth_parallel_grad_sync
        model.require_backward_grad_sync = not enabled
        self.do_depth_parallel_grad_sync = not enabled
        try:
            yield None
        finally:
            model.require_backward_grad_sync = old_require_backward_grad_sync
            self.do_depth_parallel_grad_sync = old_do_depth_parallel_grad_sync


    # TODO: Depreciate. Should be using all reduce for this
    def calculate_world_loss(self, on_gpu_loss):
        for process_group in [ax.comm_handle.coll_nccl_comm, ax.comm_handle.depth_intra_layer_parallel_group]:
            torch.distributed.all_reduce(on_gpu_loss, group=process_group)
            on_gpu_loss = on_gpu_loss / torch.distributed.get_world_size(process_group)
        return on_gpu_loss


    def optimize_communication(self, model, enabled=True):
        if enabled:
            return partial(optimize_communication,
                    overlap_all_reduce=True,
                    overlap_reduce_scatter=False, # not compatible with DDP
                    overlap_all_gather=True,
                    model_object_for_overlapping_allgathers=model) 
        else:
            return nullcontext
