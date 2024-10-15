import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt

from safetensors import safe_open

import transformers
import torch.nn.functional as F

def extract_layer_number(key):
    parts = key.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None

def resize_tensor(tensor, size):
    if tensor.size == size:
        return tensor
    elif tensor.size > size:
        return tensor[:size]
    else:
        return np.pad(tensor, (0, size - tensor.size), 'constant')

def calculate_tensor_cosine_similarity(tensors, tensors_large):
    cosine_similarities = {}
    
    for key in tensors.keys():
        layer_number = extract_layer_number(key)
        if layer_number is not None:
            large_key = key.replace(f'.{layer_number}.', f'.{layer_number}.')
            if large_key in tensors_large:
                # Flatten the tensors to 1D arrays
                tensor1 = tensors[key].flatten().cpu()
                tensor2 = tensors_large[large_key].flatten().cpu()
                
                # Resize or pad the tensors to the same shape
                max_size = max(tensor1.size(0), tensor2.size(0))
                tensor1 = torch.from_numpy(resize_tensor(tensor1.numpy(), max_size))
                tensor2 = torch.from_numpy(resize_tensor(tensor2.numpy(), max_size))
                
                # Calculate the cosine similarity
                cosine_similarity = F.cosine_similarity(tensor1, tensor2, dim=0).item()
                cosine_similarities[key] = cosine_similarity
    
    return cosine_similarities

# Example usage
tensors_base = {}
with safe_open("checkpoint/roberta-base_rte_vera_qv_80/model/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors_base[k] = f.get_tensor(k) # loads the full tensor given a key

tensors_large = {}
with safe_open("checkpoint/roberta-base_cola_vera_qv_20/model/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors_large[k] = f.get_tensor(k) # loads the full tensor given a key

cosine_similarities = calculate_tensor_cosine_similarity(tensors_base, tensors_large)

# Plotting the cosine similarities
keys = list(cosine_similarities.keys())
similarities = list(cosine_similarities.values())

fig_name = 'bottom'

plt.figure(figsize=(12, 12))

bars = plt.bar(keys, similarities, color='skyblue', width=0.6, label='Cosine Similarity')
plt.xlabel('Tensor Keys', fontsize=14)
plt.ylabel('Cosine Similarity', fontsize=14)
plt.title('Cosine Similarity between Tensors with Matching Keys : ' + fig_name, fontsize=16)
plt.xticks(rotation=90, fontsize=10)  # Rotate the x-axis labels for better readability
plt.yticks(fontsize=10)
plt.legend(fontsize=12) 

# Manually adjust the margins
plt.subplots_adjust(bottom=0.3, top=0.9)

# Save the plot as an image file
plt.savefig('cosine_similarity_plot_' + fig_name + '.png')

plt.show()

# Separate keys based on ending with 'lambda_b' or 'lambda_d'
keys_lambda_b = [key for key in cosine_similarities.keys() if key.endswith('lambda_b')]
keys_lambda_d = [key for key in cosine_similarities.keys() if key.endswith('lambda_d')]

# Plotting the cosine similarities for keys ending with 'lambda_b'
similarities_lambda_b = [cosine_similarities[key] for key in keys_lambda_b]

plt.figure(figsize=(12, 12))
plt.bar(keys_lambda_b, similarities_lambda_b, color='skyblue', width=0.6, label='Cosine Similarity (lambda_b)')
plt.xlabel('Tensor Keys', fontsize=14)
plt.ylabel('Cosine Similarity', fontsize=14)
plt.title('Cosine Similarity between Tensors with Matching Keys : ' + fig_name + ' (lambda_b)', fontsize=16)
plt.xticks(rotation=90, fontsize=10)  # Rotate the x-axis labels for better readability
plt.yticks(fontsize=10)
plt.legend(fontsize=12)  # Add a legend

# Manually adjust the margins
plt.subplots_adjust(bottom=0.3, top=0.9)

# Save the plot as an image file
plt.savefig('cosine_similarity_plot_' + fig_name + '_lambda_b.png')

plt.show()

# Plotting the cosine similarities for keys ending with 'lambda_d'
similarities_lambda_d = [cosine_similarities[key] for key in keys_lambda_d]

plt.figure(figsize=(12, 12))
plt.bar(keys_lambda_d, similarities_lambda_d, color='lightcoral', width=0.6, label='Cosine Similarity (lambda_d)')
plt.xlabel('Tensor Keys', fontsize=14)
plt.ylabel('Cosine Similarity', fontsize=14)
plt.title('Cosine Similarity between Tensors with Matching Keys : ' + fig_name + ' (lambda_d)', fontsize=16)
plt.xticks(rotation=90, fontsize=10)  # Rotate the x-axis labels for better readability
plt.yticks(fontsize=10)
plt.legend(fontsize=12)  # Add a legend

# Manually adjust the margins
plt.subplots_adjust(bottom=0.3, top=0.9)

# Save the plot as an image file
plt.savefig('cosine_similarity_plot_' + fig_name + '_lambda_d.png')

plt.show()