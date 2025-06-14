from typing import List

import torch
from spde_model import SPDEModel
from spde_parameters import Parameters
from torch.profiler import ProfilerActivity, profile


def run_profiling(
    spde: SPDEModel,
    Y: torch.Tensor,
    A: torch.Tensor,
    logparams_list: List[torch.Tensor],
) -> None:
    # Enable profiling for both forward and backward passes

    with profile(
        activities=[ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        spde.params.set_from_logparam_val_list(logparams_list)

        # # Your forward and backward pass
        ll = spde.inference.log_likelihood(Y, A)  # Forward pass
        ll.backward()  # Backward pass

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


def run_profiling_for_ADAM_opt(
    spde: SPDEModel,
    Y: torch.Tensor,
    A: torch.Tensor,
    logparams_list: List[torch.Tensor],
) -> None:
    # Enable profiling for both forward and backward passes

    optimizer = torch.optim.Adam(logparams_list, lr=0.05)
    with profile(
        activities=[ProfilerActivity.CPU],
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_stack=True,
    ) as prof:
        spde.params.set_from_logparam_val_list(logparams_list)

        # Refresh the model
        spde.refresh()

        # # Your forward and backward pass
        ll = -spde.inference.log_likelihood(Y, A)  # Forward pass

        ll.backward()  # Backward pass

        optimizer.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
