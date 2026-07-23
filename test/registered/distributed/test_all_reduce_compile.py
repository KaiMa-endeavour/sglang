"""Regression test: `GroupCoordinator.all_reduce` must compile once under Dynamo.

The all-reduce method selection (`should_custom_ar` -> `_pick_algo`) compares
tensor byte sizes against tuned thresholds. If that selection runs inside a
Dynamo trace with a dynamic token dim, every comparison guards on the symbolic
shape, so each new shape bucket recompiles the graph. Under the tc_piecewise
prefill backend (`fullgraph=True`, default `recompile_limit=8`) this escalated
to `FailOnRecompileLimitHit` and killed server startup for TP models
(GLM-5.2-NVFP4, Kimi-K2.5/K2.6). The fix defers method selection to runtime
inside the opaque `outplace_all_reduce` custom op ("auto" method).

This test compiles a function that all-reduces tensors spanning ~28KB to
~115MB (crossing every tuned threshold on SM90/SM100) and asserts a single
Dynamo compile plus NCCL-exact numerics.

Pattern follows `test/registered/eplb/test_lplb_distributed.py` —
`torch.multiprocessing.spawn` is required because `torch.distributed`
plays poorly with arbitrary subprocess launchers.
"""

import pytest
import torch
import torch.distributed as dist

from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")

NUM_GPUS = 2
HIDDEN_DIM = 7168
TOKEN_COUNTS = [2, 4, 16, 64, 256, 1024, 2048, 4096, 8192]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < NUM_GPUS,
    reason=f"This test requires at least {NUM_GPUS} CUDA devices",
)
def test_all_reduce_compiles_once_across_shapes():
    torch.multiprocessing.spawn(
        _worker_main,
        args=(NUM_GPUS,),
        nprocs=NUM_GPUS,
    )


def _worker_main(local_rank: int, world_size: int):
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12365",  # Distinct from other tests' ports.
        }
    )
    init_distributed_environment(
        world_size=world_size, rank=local_rank, local_rank=local_rank
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    group = get_tensor_model_parallel_group().device_group

    try:
        warmup = torch.zeros(1, device=device)
        dist.all_reduce(warmup, group=group)
        torch.cuda.synchronize()

        compile_count = 0

        def counting_backend(gm, example_inputs):
            nonlocal compile_count
            compile_count += 1
            return gm.forward

        def fn(x):
            return tensor_model_parallel_all_reduce(x + 1)

        compiled = torch.compile(fn, fullgraph=True, backend=counting_backend)

        for tokens in TOKEN_COUNTS:
            # Integer values keep custom all-reduce bit-identical to NCCL.
            x = torch.randint(
                1, 8, (tokens, HIDDEN_DIM), device=device, dtype=torch.int32
            ).to(torch.bfloat16)
            torch._dynamo.maybe_mark_dynamic(x, 0)
            out = compiled(x)
            ref = x + 1
            dist.all_reduce(ref, group=group)
            assert torch.equal(out, ref), f"all-reduce mismatch at tokens={tokens}"

        assert compile_count == 1, (
            f"expected a single dynamic-shape compile, got {compile_count}: "
            "all-reduce method selection is leaking shape guards into Dynamo"
        )
    finally:
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
