user=$(whoami)
echo "User: $user"
python scripts/convert_lit_checkpoint.py --checkpoint_path /lustre/orion/csc569/scratch/$user/lit-gpt-dev/out/lit-tiny-llama-1.1b/step-00120000.pth --output_path /lustre/orion/csc569/scratch/$user/lit-gpt-dev/transformer_ckpts/lit-tiny-llama-1.1b-120k-steps-500B-tokens --model_name tiny-llama-1.1b 
cd /lustre/orion/csc569/scratch/$user/lit-gpt-dev/transformer_ckpts/lit-tiny-llama-1.1b-120k-steps-500B-tokens
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-step-50K-105b/resolve/main/special_tokens_map.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-step-50K-105b/resolve/main/tokenizer_config.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-step-50K-105b/resolve/main/tokenizer.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-step-50K-105b/resolve/main/tokenizer.model
cd /lustre/orion/csc569/scratch/$user/lit-gpt-dev
python scripts/push_to_hub.py --model_name tiny-llama-1.1b-120k-steps-500B-tokens  --model_path /lustre/orion/csc569/scratch/$user/lit-gpt-dev/transformer_ckpts/lit-tiny-llama-1.1b-120k-steps-500B-tokens --token_id $HF_TOKEN