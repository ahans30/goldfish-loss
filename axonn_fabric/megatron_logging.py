import torch

def get_mem():
    curr =  torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    peak =  torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    return curr, peak

def get_tflops(config, batch_size):
    N = config.n_layer
    B = batch_size
    S = config.block_size
    V = config.padded_vocab_size
    H = config.n_embd
    IH = config.intermediate_size


    linear_flops = N*(32*B*S*H*H + 24 * B * S * H * IH)
    attention_flops = N*(16 * B * S * S * H)
    head_flops = 6 * B * S * H * V
    if config.gradient_checkpointing:
        flops = linear_flops + attention_flops + head_flops
    else:
        flops = 3/4*(linear_flops + attention_flops) + head_flops

    return flops/1e12

def pretty_log(iteration, 
               train_iters,
               consumed_train_samples,
               elapsed_time_per_iteration,
               learning_rate,
               batch_size,
               train_loss,
               grad_norm=None,
               model_name=None,
               config=None):
    log_string = '> global batch {:8d}/{:8d} |'.format(
        iteration, train_iters)
    log_string += ' consumed samples: {:12d} |'.format(
        consumed_train_samples)
    log_string += ' elapsed time per global batch (ms): {:.1f} |'.format(
        elapsed_time_per_iteration * 1000.0)
    log_string += ' learning rate: {:.3E} |'.format(learning_rate)
    log_string += ' global batch size: {:5d} |'.format(batch_size)
    log_string += ' loss: {:.5f} |'.format(train_loss)
    #log_string += ' loss scale: {:.1f} |'.format(loss_scale)
    if grad_norm is not None:
        log_string += ' grad norm: {:.3f} |'.format(grad_norm)
    #if num_zeros_in_grad is not None:
    #    log_string += ' num zeros: {:.1f} |'.format(num_zeros_in_grad)
    #if params_norm is not None:
    #    log_string += ' params norm: {:.3f} |'.format(params_norm)
    #log_string += ' number of skipped iterations: {:3d} |'.format(
    #    total_loss_dict[skipped_iters_key])
    #log_string += ' number of nan iterations: {:3d} |'.format(
    #    total_loss_dict[nan_iters_key])
    #log_string += ' theoretical FLOP/s: {:.3f} TFLOP/s | '.format(get_flops(elapsed_time_per_iteration))
    #log_string += ' model size: {:.3f} B params | '.format(get_params())
    curr, peak = get_mem()
    log_string += ' memory used by tensors {:.3f} GB (peak {:.3f} GB) |'.format(curr, peak)
    if model_name is not None:
        log_string += f' model name {model_name} |'
    if config is not None:
        log_string += f' {get_tflops(config, batch_size)/elapsed_time_per_iteration/torch.distributed.get_world_size():.2f} TFLOP/s per GPU'
    return log_string
