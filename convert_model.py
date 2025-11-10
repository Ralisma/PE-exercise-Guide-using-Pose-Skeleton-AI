import json
import numpy as np
import tensorflow as tf
import os

# Load the model.json
with open('my_model/model.json', 'r') as f:
    model_config = json.load(f)

# Build the model
model = tf.keras.Sequential()
layers_config = model_config['modelTopology']['config']['layers']

for layer_config in layers_config:
    layer_class = layer_config['class_name']
    config = layer_config['config']
    
    if layer_class == 'Dense':
        kwargs = {
            'units': config['units'],
            'activation': config.get('activation'),
            'use_bias': config.get('use_bias', True),
            'name': config['name']
        }
        if 'batch_input_shape' in config:
            kwargs['input_shape'] = config['batch_input_shape'][1:]
        model.add(tf.keras.layers.Dense(**kwargs))
    elif layer_class == 'Dropout':
        model.add(tf.keras.layers.Dropout(
            rate=config['rate'],
            name=config['name']
        ))

# Load weights
weights_manifest = model_config['weightsManifest'][0]
bin_path = os.path.join('my_model', weights_manifest['paths'][0])
with open(bin_path, 'rb') as f:
    weights_data = f.read()

all_weights = []
offset = 0
for weight_spec in weights_manifest['weights']:
    shape = weight_spec['shape']
    dtype = np.dtype(weight_spec['dtype'])
    num_elements = np.prod(shape)
    byte_size = num_elements * dtype.itemsize
    weight_bytes = weights_data[offset:offset + byte_size]
    weight_array = np.frombuffer(weight_bytes, dtype=dtype).reshape(shape)
    all_weights.append(weight_array)
    offset += byte_size
    print(f'Loaded {weight_spec["name"]} with shape {shape}')

# Assign weights to layers
model_layers = {layer.name: layer for layer in model.layers}
weight_idx = 0
for layer_config in layers_config:
    if layer_config['class_name'] == 'Dense':
        layer_name = layer_config['config']['name']
        if layer_name in model_layers:
            layer = model_layers[layer_name]
            if layer_config['config'].get('use_bias', True):
                layer.set_weights([all_weights[weight_idx], all_weights[weight_idx + 1]])
                weight_idx += 2
            else:
                layer.set_weights([all_weights[weight_idx]])
                weight_idx += 1

# Save the model
model.save('my_keras_model/model.h5')
print('Model saved to my_keras_model/model.h5')
