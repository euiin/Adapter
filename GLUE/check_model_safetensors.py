from safetensors import safe_open


# Example usage
tensors= {}
with safe_open("checkpoint/roberta-base_rte_alora_A_rand_hB_zero_r_8_seed_49/model/adapter_model.safetensors", framework="pt", device=0) as f:
    breakpoint()
    for k in f.keys():
        tensors[k] = f.get_tensor(k) # loads the full tensor given a key

print(tensors)
breakpoint()
print('finished')
