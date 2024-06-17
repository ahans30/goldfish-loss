# Hand-made frontier launch solution
"""Specs:
* Callable from python
* Automate good settings for HPC
* Handle file transfer and environment unpacking on all nodes
* Check a fixed git commit [optional]
* Load all required modules automatically
* Handle interconnect
* Handle default job dependencies, split into a series of 2h jobs, based on singleton execution

Todo:
* Handle automatic dependencies, such as cooldown and eval
"""
import os
import socket
import getpass
import time
import secrets
import subprocess

import argparse


from dataclasses import dataclass

SHELL = os.environ["SHELL"]

if SHELL == "/bin/bash":
    RC_FILE = os.path.expanduser("~/.bashrc")
elif SHELL == "/bin/zsh":
    RC_FILE = os.path.expanduser("~/.zshrc")
else:
    raise ValueError("Unsupported shell.")


def frontier_max_minutes(num_nodes: int):
    if num_nodes > 184:
        return 720
    elif num_nodes > 92:
        return 360
    else:
        return 120


def load_standard_modules():
    return """
echo $(date -u) "Loading modules"
module load PrgEnv-cray 
module load amd-mixed/5.6.0
module load craype-accel-amd-gfx90a
module load libfabric
module load libtool
module load miniforge3
"""


def get_comms_and_slingshot(installdir="${HOME}/tiny_plugins_rccl/lib", enable_net_gdr=False, rccl_flag_variant=False):
    # net_gdr is twice as fast on frontier, but may lead to hangs [...]
    # ENV variables are also documented in https://www.olcf.ornl.gov/wp-content/uploads/2021/04/HPE-Cray-MPI-Update-nfr-presented.pdf # noqa
    return f"""
### MPI
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}:${{CRAY_MPICH_ROOTDIR}}/gtl/lib"
### AMD GPU
export HSA_FORCE_FINE_GRAIN_PCIE=1
### Slingshot
export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}:{installdir}"
export FI_CXI_ATS=0
{'export NCCL_CROSS_NIC=1' if not rccl_flag_variant else ''}
{'export NCCL_NET_GDR_LEVEL=3' if enable_net_gdr else ''}
{'export NCCL_SOCKET_IFNAME=hsn0' if rccl_flag_variant else ''}
{'export GLOO_SOCKET_IFNAME=hsn0' if rccl_flag_variant else ''}
"""


def activate_env(env_path=r"${HOME}/frontier_conda"):
    return f"""
echo $(date -u) "Activating environment"
source deactivate > /dev/null 2>&1
source activate {env_path}
"""


def cast_and_unpack_env(env_full_path="frontier_env_packed.tar.gz", num_nodes=8):
    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
    # your application may pick up partially complete shared library files, which would give you confusing errors.
    # pigz -d as tar -xzf alternative is available on frontier
    # but srun -N {num_nodes} --ntasks-per-node=1 bash -c 'pigz -dc /mnt/bb/${{USER}}/{env_name}.tar.gz | tar -xf - -C /mnt/bb/${{USER}}/{env_name}' # noqa
    # (so, with pigz) is not much faster
    # OLCF DOC: https://docs.olcf.ornl.gov/software/python/sbcast_conda.html
    # Note: Disabling previously present env, just in case. This is valid even if no environment is set
    if not env_full_path.endswith(".tar.gz"):
        raise ValueError("Invalid environment archive path provided.")
    dirname, filename = os.path.split(env_full_path)
    env_name = filename.rsplit(".tar.gz", 1)[0]
    return f"""
echo $(date -u) "Copying environment to each node"
source deactivate > /dev/null 2>&1
ENV_NAME={env_name}

sbcast -pf {dirname}/{env_name}.tar.gz /mnt/bb/${{USER}}/{env_name}.tar.gz
if [ ! "$?" == "0" ]; then
    echo "SBCAST failed!"
    exit 1
fi

# Untar the environment file (only need 1 task per node to do this)
srun -N{num_nodes} --ntasks-per-node 1 mkdir /mnt/bb/${{USER}}/{env_name}
echo $(date -u) "Untaring environment"
srun -N{num_nodes} --ntasks-per-node 1 tar -xzf /mnt/bb/${{USER}}/{env_name}.tar.gz -C  /mnt/bb/${{USER}}/{env_name}

# Unpack the env
echo $(date -u) "Unpacking environment"
source activate /mnt/bb/${{USER}}/{env_name}
srun -N{num_nodes} --ntasks-per-node 1 conda-unpack
"""


def set_generic_env_flags(
    run_name="debug-run",
    gpus_per_node=8,
    master_port=29500,
    debug_flags_python=True,
    debug_flags_interconnect=False,
    host_on_rank_zero=True,
    output_dir="$(pwd)/output/$RUN_NAME",
):
    # afaik RCCL does not pick up NCCL_DEBUG_SUBSYS=WARN, so those flags are removed
    # it does pick up NCCL_DEBUG and the libfabric plugin picks up FI_LOG_LEVEL
    if host_on_rank_zero:
        master_address = "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)"
    else:
        master_address = "$(hostname)"

    return f"""
export RUN_NAME="{run_name}"

NNODES=$SLURM_JOB_NUM_NODES
export MASTER_ADDR={master_address}
export MASTER_PORT={master_port}
export WORLD_SIZE=$(( NNODES * {gpus_per_node} )) 
# debugging flags (optional)
{'export LOGLEVEL=INFO' if debug_flags_python else ''}
{'export PYTHONFAULTHANDLER=1' if debug_flags_python else ''}
{'export NCCL_DEBUG=WARN' if debug_flags_interconnect else ''}
{'export FI_LOG_LEVEL=warn' if debug_flags_interconnect else ''}
# wandb 
wandb offline
# frontier specific:
export OMP_NUM_THREADS=7 
# Dirs
export OUTPUT_DIR={output_dir}
export LOGDIR=${{OUTPUT_DIR}}/logs
mkdir -p $LOGDIR

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "Logging to $LOGDIR"
"""


def get_repo_at_specific_commit(commit_id=None):
    """Current behavior: Copy repo into current working directory, wherever it may be."""
    archive_path = (
        f"https://github.com/tomg-group-umd/lit-gpt-dev/archive/"
        f"{f'@{commit_id}' if commit_id is not None else 'main'}.tar.gz"
    )
    return f"""
cd $OUTPUT_DIR
wget -O lit-gpt.tar.gz {archive_path}
tar -xzf lit-gpt.tar.gz -C  lit-gpt-dev-{commit_id}
cd lit-gpt-dev-{commit_id}
"""


def assemble_sbatch_file(
    output_dir=None,
    run_name="debug-run",
    python_invocation="pretrain_umd/train.py",
    nodes=8,
    budget_minutes=120,
    rccl_installdir="${HOME}/tiny_plugins_rccl/lib",
    env_packed=None,
    env_path=None,
    email=None,
    gpus_per_node=8,
    detach_repo_at_commit=None,
    gpu_bind=False,
    repetitions=1,
    dependency=None,
):
    assert (env_packed in [None, ""]) ^ (
        env_path in [None, ""]
    ), "Exactly one of env_packed or env_path must be provided."

    hours = budget_minutes // 60
    minutes = budget_minutes - hours * 60
    sock = socket.socket()
    # Find a free socket:
    sock.bind(("", 0))
    free_socket_frontier = sock.getsockname()[1]
    sock.close()
    # Prealloc logfile and output folder
    logdir = f"{output_dir}/logs"
    os.makedirs(logdir, exist_ok=True)

    return rf"""#!/bin/bash
#SBATCH --account=csc569
#SBATCH --time={hours}:{minutes:02d}:00
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --constraint=nvme

#SBATCH --array=1-{repetitions}%1 
{f'#SBATCH --dependency={dependency}' if dependency is not None else ''}

#SBATCH --job-name={run_name}
#SBATCH --output={logdir}/%x_%A_%a.log
#SBATCH --error={logdir}/%x_%A_%a.log
#SBATCH --open-mode=append
{f'#SBATCH --mail-user={email}' if email is not None else ''}
{'#SBATCH --mail-type=FAIL,ARRAY_TASKS' if email is not None else ''}

echo $(date -u) "Preparing run..."
{load_standard_modules()}
{cast_and_unpack_env(env_packed, num_nodes=nodes) if env_packed not in [None, ''] else ''}
{activate_env(env_path) if env_path not in [None, ''] else ''}
{get_comms_and_slingshot(rccl_installdir) if rccl_installdir not in [None, ''] else ''}
{set_generic_env_flags(run_name=run_name,gpus_per_node=gpus_per_node, master_port=free_socket_frontier, output_dir=output_dir)}
{detach_repo_at_commit(commit_id=detach_repo_at_commit) if detach_repo_at_commit not in [None, ''] else ''}

echo $(date -u) "Starting run..."
srun -l -N {nodes} -n {gpus_per_node * nodes} -c7 --ntasks-per-node={gpus_per_node} \
    --gpus-per-node={gpus_per_node} {'--gpus-per-task=1 --gpu-bind=closest' if gpu_bind else ''} \
    python -u {python_invocation}
echo $(date -u) "Job execution finished."
"""


@dataclass
class SLURMLaunch:

    output_dir: str = None
    sub_output_dir_name: str = None
    default_python_invocation: str = "pretrain_umd/train.py"
    nodes: int = 8
    budget_minutes: int = 120
    rccl_installdir: str = "${HOME}/tiny_plugins_rccl/lib"
    env_packed: str = None
    env_path: str = None
    email: str = None
    gpus_per_node: int = 8
    detach_repo_at_commit: int = None
    repetitions: int = None
    dependency: str = None

    def sbatch_repr(self, run_name: str, python_invocation: str):
        authkey = secrets.token_urlsafe(5)
        unique_run_name = f"{run_name}_{authkey}"

        if self.sub_output_dir_name is None:
            self.sub_output_dir_name = unique_run_name

        if self.output_dir is None:
            self.output_dir = f"{os.getcwd()}/output/{self.sub_output_dir_name}"
        else:
            self.output_dir = f"{self.output_dir}/{self.sub_output_dir_name}"

        if self.repetitions is None:
            computed_repetitions = 1 + self.budget_minutes // frontier_max_minutes(self.nodes)
        else:
            computed_repetitions = self.repetitions
        assert computed_repetitions > 0, "Repetitions must be a positive integer."

        sbatch_file = assemble_sbatch_file(
            output_dir=self.output_dir,
            run_name=run_name,
            python_invocation=python_invocation,
            nodes=self.nodes,
            budget_minutes=min(self.budget_minutes, frontier_max_minutes(self.nodes)),
            rccl_installdir=self.rccl_installdir,
            env_packed=self.env_packed,
            env_path=self.env_path,
            email=self.email,
            gpus_per_node=self.gpus_per_node,
            detach_repo_at_commit=self.detach_repo_at_commit,
            repetitions=computed_repetitions,
            dependency=self.dependency,
        )
        return sbatch_file, unique_run_name

    def execute(
        self,
        run_name="debug-run",
        python_invocation="pretrain_umd/train.py",
        dryrun=True,
        debug_qos=True,
        launch_immediately=False,
    ):
        sbatch_file, unique_run_name = self.sbatch_repr(run_name, python_invocation)
        sbatch_file_name = f"{run_name}_launch.sbatch"
        sbatch_file_path = f"{self.output_dir}/{sbatch_file_name}"
        with open(sbatch_file_path, "w") as file:
            file.write(sbatch_file)
        print("Launch Specs are:")
        print(sbatch_file)

        username = getpass.getuser()
        print(f"Preparing job as user {username}" f" for launch from {socket.gethostname()} in 10 seconds...")
        print(f"This will allocate {self.nodes} nodes, so {self.nodes * self.gpus_per_node} GPUS in total.")

        if not dryrun:
            print(
                f"An array with {1 + self.budget_minutes // frontier_max_minutes(self.nodes)} jobs will be launched to SLURM, "
                f"with singleton dependencies."
            )
            print(f"Terminate job {unique_run_name} if necessary ...")
            try:
                if not launch_immediately:
                    time.sleep(5)
            except KeyboardInterrupt:
                os.remove(sbatch_file_path)
                print("Interrupt registered. No job was launched.")
                return

            output_status = subprocess.run(
                [
                    "/usr/bin/sbatch",
                    f"--qos={'debug' if debug_qos else 'normal'}",
                    f"{sbatch_file_path}",
                ],
                capture_output=True,
            )
            if len(output_status.stderr) > 0:
                raise ValueError(output_status.stderr)
            process_id = output_status.stdout.decode("utf-8").split("batch job ")[1].split("\n")[0]
            print(f"Launched job array with process id {process_id}.")

        else:
            print(
                f"An array of {1 + self.budget_minutes // frontier_max_minutes(self.nodes)} jobs would be launched to SLURM "
                f", with singleton dependencies."
            )
            print(f"No jobs are launched now. You can inspect the sbatch file at {sbatch_file_path}.")
            # subprocess.run(["/usr/bin/cat", sbatch_file_path], capture_output=True)


def parse_and_execute():
    """Parser, specificially turned to files that look like the pretrain_umd.train.py file.
    You can always replace this parse with your
    own construction of a SLURMLaunch object, to which you can pass your desired python invocation,
    or you can use --custom_invocation=debug_script.py or so to entirely overwrite the launcher."""

    parser = argparse.ArgumentParser(description="Dispatch a particular launch onto frontier.")
    # Base
    parser.add_argument("--run_name", default="frontier-debug", type=str, help="Name that will be displayed in squeue")
    parser.add_argument("--output_dir", default=None, type=str, help="The output dir.")
    parser.add_argument(
        "--sub_output_dir_name", default=None, type=str, help="The dir where sbatch files and logs are written."
    )
    parser.add_argument("--config", default=None, type=str, help="Which config? If None, no config is passed.")
    parser.add_argument("--extra_args", default="", type=str, help="Extra arguments to train.py as --arg=X")
    parser.add_argument("--email", default=None, type=str, help="Your email.")
    parser.add_argument("--dryrun", action="store_true", help="The sbatch file is only written and can be modified.")
    parser.add_argument("--debug_qos", action="store_true", help="Launch onto debug queue.")
    # Core Requests
    parser.add_argument("--budget_minutes", default=120, type=int, help="Requested runtime in minutes")
    parser.add_argument("--budget_hours", default=0, type=int, help="Requested runtime in hours.")
    parser.add_argument("--budget_days", default=0, type=int, help="Requested runtime in days.")
    parser.add_argument("--nodes", default="1", type=int, help="Requested number of nodes.")
    parser.add_argument("--gpus_per_node", default=8, type=int, help="Requested number of GPUs per node.")
    parser.add_argument("--repetitions", default=None, type=int, help="Manual number of repetitions.")
    parser.add_argument(
        "--dependency", default=None, type=str, help="Specify whether to launch w/ dependency eg. singleton."
    )
    # Job details
    parser.add_argument("--python_script", default="pretrain_umd/train.py", type=str, help="Pretrain script.")
    parser.add_argument("--rccl_installdir", default=None, type=str, help="Where did you install the RCCL plugin?")
    parser.add_argument("--env_packed", default=None, type=str, help="Where is the packed env?")
    parser.add_argument("--env_path", default=None, type=str, help="Where is the _unpacked_ env?")
    parser.add_argument(
        "--detach_repo_at_commit",
        default=None,
        type=int,
        help="If a github commit id is provided, the repo is detached and an independent copy is created from this commit.",
    )
    parser.add_argument(
        "--custom_invocation",
        default=None,
        type=str,
        help="Overwrite all other arguments and provide a custom python invocation",
    )
    parser.add_argument(
        "--launch_immediately", action="store_true", help="Launch the job immediately w/ no sleep time."
    )
    args = parser.parse_args()

    actual_budget_minutes = args.budget_minutes + 60 * args.budget_hours + 60 * 24 * args.budget_days

    if args.custom_invocation is None:
        fully_assembled_invocation = (
            f"{args.python_script} "
            f"{f'--config={args.config} ' if args.config is not None else ''}"
            f"--run_name={args.run_name} --out_dir=$OUTPUT_DIR {args.extra_args}"
        )
    else:
        fully_assembled_invocation = args.custom_invocation

    # Define launch settings, environment and SLURM directives at construction time
    launch_object = SLURMLaunch(
        output_dir=args.output_dir,
        sub_output_dir_name=args.sub_output_dir_name,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        budget_minutes=actual_budget_minutes,
        rccl_installdir=args.rccl_installdir,
        env_packed=args.env_packed,
        env_path=args.env_path,
        email=args.email,
        detach_repo_at_commit=args.detach_repo_at_commit,
        repetitions=args.repetitions,
        dependency=args.dependency,
    )
    # Execute a particular python command (here `fully_assembled_invocation`) with obj.execute()
    launch_object.execute(
        python_invocation=fully_assembled_invocation,
        run_name=args.run_name,
        dryrun=args.dryrun,
        debug_qos=args.debug_qos,
        launch_immediately=args.launch_immediately,
    )


if __name__ == "__main__":
    parse_and_execute()


# Notes:
# * Should we also sbcast the RCCL plugin?


# ###### Legacy options ############ #


def set_output_and_log_dir():
    return f"""
export OUTPUT_DIR="$(pwd)/output/$RUN_NAME"
mkdir -p $OUTPUT_DIR
echo "OUTPUT_DIR: $OUTPUT_DIR"

# Create the log file.
LOGDIR=${{OUTPUT_DIR}}/logs
mkdir -p $LOGDIR

echo "Logging to $LOGDIR"
"""  # noqa to keep the format consistent across all bash snippets in the file
