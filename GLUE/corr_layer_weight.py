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

def calculate_tensor_correlations(tensors, tensors_large):
    correlation_tensors = {}
    
    for key in tensors.keys():
        layer_number = extract_layer_number(key)
        if layer_number is not None:
            large_key = key.replace(f'.{layer_number}.', f'.{layer_number}.')
            if large_key in tensors_large:
                # Flatten the tensors to 1D arrays
                tensor1 = tensors[key].flatten().cpu().numpy()
                tensor2 = tensors_large[large_key].flatten().cpu().numpy()
                
                # Resize or pad the tensors to the same shape
                max_size = max(tensor1.size, tensor2.size)
                tensor1 = resize_tensor(tensor1, max_size)
                tensor2 = resize_tensor(tensor2, max_size)
                
                # Calculate the Pearson correlation coefficient
                correlation = np.corrcoef(tensor1, tensor2)[0, 1]
                correlation_tensors[key] = correlation
    
    return correlation_tensors

# Example usage
tensors_base = {}
with safe_open("checkpoint/roberta-base_rte_vera_qv_80/model/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors_base[k] = f.get_tensor(k) # loads the full tensor given a key

tensors_large = {}
with safe_open("checkpoint/roberta-base_cola_vera_qv_20/model/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors_large[k] = f.get_tensor(k) # loads the full tensor given a key

correlation_tensors = calculate_tensor_correlations(tensors_base, tensors_large)

# Plotting the correlations
keys = list(correlation_tensors.keys())
correlations = list(correlation_tensors.values())

fig_name = 'bottom'

plt.figure(figsize=(12, 12))

# plt.bar(keys, correlations)
# plt.xlabel('Tensor Keys')
# plt.ylabel('Correlation')
# plt.title('Correlation between Tensors with Matching Keys')
# plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability

bars = plt.bar(keys, correlations, color='skyblue', width=0.6, label='Correlation')
plt.xlabel('Tensor Keys', fontsize=14)
plt.ylabel('Correlation', fontsize=14)
plt.title('Correlation between Tensors with Matching Keys : ' + fig_name, fontsize=16)
plt.xticks(rotation=90, fontsize=10)  # Rotate the x-axis labels for better readability
plt.yticks(fontsize=10)
plt.legend(fontsize=12) 

# Manually adjust the margins
plt.subplots_adjust(bottom=0.3, top=0.9)


# Save the plot as an image file
plt.savefig('correlation_plot_' + fig_name + '.png')

plt.show()

# Separate keys based on ending with 'lambda_b' or 'lambda_d'
keys_lambda_b = [key for key in correlation_tensors.keys() if key.endswith('lambda_b')]
keys_lambda_d = [key for key in correlation_tensors.keys() if key.endswith('lambda_d')]

# Plotting the correlations for keys ending with 'lambda_b'
correlations_lambda_b = [correlation_tensors[key] for key in keys_lambda_b]

plt.figure(figsize=(12, 12))
plt.bar(keys_lambda_b, correlations_lambda_b, color='skyblue', width=0.6, label='Correlation (lambda_b)')
plt.xlabel('Tensor Keys', fontsize=14)
plt.ylabel('Correlation', fontsize=14)
plt.title('Correlation between Tensors with Matching Keys : ' + fig_name + ' (lambda_b)', fontsize=16)
plt.xticks(rotation=90, fontsize=10)  # Rotate the x-axis labels for better readability
plt.yticks(fontsize=10)
plt.legend(fontsize=12)  # Add a legend

# Manually adjust the margins
plt.subplots_adjust(bottom=0.3, top=0.9)

# Save the plot as an image file
plt.savefig('correlation_plot_' + fig_name + '_lambda_b.png')

plt.show()

# Plotting the correlations for keys ending with 'lambda_d'
correlations_lambda_d = [correlation_tensors[key] for key in keys_lambda_d]

plt.figure(figsize=(12, 12))
plt.bar(keys_lambda_d, correlations_lambda_d, color='lightcoral', width=0.6, label='Correlation (lambda_d)')
plt.xlabel('Tensor Keys', fontsize=14)
plt.ylabel('Correlation', fontsize=14)
plt.title('Correlation between Tensors with Matching Keys : ' + fig_name + ' (lambda_d)', fontsize=16)
plt.xticks(rotation=90, fontsize=10)  # Rotate the x-axis labels for better readability
plt.yticks(fontsize=10)
plt.legend(fontsize=12)  # Add a legend

# Manually adjust the margins
plt.subplots_adjust(bottom=0.3, top=0.9)

# Save the plot as an image file
plt.savefig('correlation_plot_' + fig_name + '_lambda_d.png')

plt.show()

 