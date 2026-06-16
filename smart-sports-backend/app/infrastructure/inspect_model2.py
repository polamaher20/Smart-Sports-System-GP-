import h5py
import json
import pathlib

path = pathlib.Path(r"c:\Users\new castle\Desktop\last update\graduation-project\smart-sports-backend\app\infrastructure\ml_models\nutrition_model.h5")
with h5py.File(path, 'r') as f:
    raw = f.attrs.get('model_config')
    raw = raw.decode('utf-8') if isinstance(raw, bytes) else raw
    cfg = json.loads(raw)
    print('config_keys', list(cfg.get('config', {}).keys()))
    layers = cfg.get('config', {}).get('layers', [])
    print('layers', len(layers))
    for i, layer in enumerate(layers):
        print('--- layer', i, layer.get('class_name'))
        cfg_item = layer.get('config')
        print('  config_type', type(cfg_item))
        if isinstance(cfg_item, dict):
            print('  keys', list(cfg_item.keys()))
            print('  name', cfg_item.get('name'))
            if layer.get('class_name') == 'InputLayer':
                print('  input_shape', cfg_item.get('input_shape'))
                print('  batch_shape', cfg_item.get('batch_shape'))
                print('  dtype', cfg_item.get('dtype'))
            if layer.get('class_name') == 'Dense':
                print('  units', cfg_item.get('units'))
                print('  activation', cfg_item.get('activation'))
                print('  kernel_initializer', cfg_item.get('kernel_initializer'))
                print('  bias_initializer', cfg_item.get('bias_initializer'))
                print('  quantization_config', cfg_item.get('quantization_config'))
                print('  batch_input_shape', cfg_item.get('batch_input_shape'))
        else:
            print('  config value', cfg_item)
    print('build_input_shape', cfg.get('config', {}).get('build_input_shape'))
    print('input_layers', cfg.get('config', {}).get('input_layers'))
    print('output_layers', cfg.get('config', {}).get('output_layers'))
