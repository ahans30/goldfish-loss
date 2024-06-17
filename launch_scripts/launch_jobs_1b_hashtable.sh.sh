# k = 3, 4, 8, 32, 128, inf

# 3-goldfish
python launch_scripts/launcher.py \
    --env_packed ${INSTALLDIR}/goldfish_loss_env_packed.tar.gz \
    --rccl_installdir="${HOME}/tiny_plugins_rccl/lib" \
    --config="launch_scripts/config/tinyllama-1b.yaml" \
    --budget_minutes=0 --budget_hours 10 --nodes=16 --email=ahans1@umd.edu \
    --run_name=tinyllama_1b_redpajama_wiki2k_20B_goldfish3_hash-table \
    --extra_args="--k_goldfish=3 --goldfish_strategy=hash-table"

# 4-goldfish
python launch_scripts/launcher.py \
    --env_packed ${INSTALLDIR}/goldfish_loss_env_packed.tar.gz \
    --rccl_installdir="${HOME}/tiny_plugins_rccl/lib" \
    --config="launch_scripts/config/tinyllama-1b.yaml" \
    --budget_minutes=0 --budget_hours 10 --nodes=16 --email=ahans1@umd.edu \
    --run_name=tinyllama_1b_redpajama_wiki2k_20B_goldfish4_hash-table \
    --extra_args="--k_goldfish=4 --goldfish_strategy=hash-table"


# 8-goldfish
python launch_scripts/launcher.py \
    --env_packed ${INSTALLDIR}/goldfish_loss_env_packed.tar.gz \
    --rccl_installdir="${HOME}/tiny_plugins_rccl/lib" \
    --config="launch_scripts/config/tinyllama-1b.yaml" \
    --budget_minutes=0 --budget_hours 10 --nodes=16 --email=ahans1@umd.edu \
    --run_name=tinyllama_1b_redpajama_wiki2k_20B_goldfish8_hash-table \
    --extra_args="--k_goldfish=8 --goldfish_strategy=hash-table"

# 32-goldfish
python launch_scripts/launcher.py \
    --env_packed ${INSTALLDIR}/goldfish_loss_env_packed.tar.gz \
    --rccl_installdir="${HOME}/tiny_plugins_rccl/lib" \
    --config="launch_scripts/config/tinyllama-1b.yaml" \
    --budget_minutes=0 --budget_hours 10 --nodes=16 --email=ahans1@umd.edu \
    --run_name=tinyllama_1b_redpajama_wiki2k_20B_goldfish32_hash-table \
    --extra_args="--k_goldfish=32 --goldfish_strategy=hash-table"


# 128-goldfish
python launch_scripts/launcher.py \
    --env_packed ${INSTALLDIR}/goldfish_loss_env_packed.tar.gz \
    --rccl_installdir="${HOME}/tiny_plugins_rccl/lib" \
    --config="launch_scripts/config/tinyllama-1b.yaml" \
    --budget_minutes=0 --budget_hours 10 --nodes=16 --email=ahans1@umd.edu \
    --run_name=tinyllama_1b_redpajama_wiki2k_20B_goldfish128_hash-table \
    --extra_args="--k_goldfish=128 --goldfish_strategy=hash-table"

# 128-goldfish
python launch_scripts/launcher.py \
    --env_packed ${INSTALLDIR}/goldfish_loss_env_packed.tar.gz \
    --rccl_installdir="${HOME}/tiny_plugins_rccl/lib" \
    --config="launch_scripts/config/tinyllama-1b.yaml" \
    --budget_minutes=0 --budget_hours 10 --nodes=16 --email=ahans1@umd.edu \
    --run_name=tinyllama_1b_redpajama_wiki2k_20B_goldfish128_hash-table \
    --extra_args="--k_goldfish=128 --goldfish_strategy=hash-table"

# inf-goldfish or standard loss
python launch_scripts/launcher.py \
    --env_packed ${INSTALLDIR}/goldfish_loss_env_packed.tar.gz \
    --rccl_installdir="${HOME}/tiny_plugins_rccl/lib" \
    --config="launch_scripts/config/tinyllama-1b.yaml" \
    --budget_minutes=0 --budget_hours 10 --nodes=16 --email=ahans1@umd.edu \
    --run_name=tinyllama_1b_redpajama_wiki2k_20B_goldfish128_hash-table