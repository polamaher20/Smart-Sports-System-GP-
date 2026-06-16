import h5py
import json
import pathlib

path = pathlib.Path(r"c:\Users\new castle\Desktop\last update\graduation-project\smart-sports-backend\app\infrastructure\ml_models\nutrition_model.h5")
with h5py.File(path, 'r') as f:
    raw = f.attrs.get('model_config')
    print('raw type', type(raw))
    raw = raw.decode('utf-8') if isinstance(raw, bytes) else raw
    cfg = json.loads(raw)
    layers = cfg.get('config', {}).get('layers', [])
    print('layers', len(layers))
    if layers:
        l0 = layers[0]
        print('layer0', l0.get('class_name'))
        print('layer0 config keys', list(l0.get('config', {}).keys()))
        print('layer0 input_shape', l0.get('config', {}).get('input_shape'))
        print('layer0 batch_shape', l0.get('config', {}).get('batch_shape'))
        if len(layers) > 1:
            l1 = layers[1]
            print('layer1', l1.get('class_name'))
            print('layer1 config keys', [k for k in l1.get('config', {}).keys() if k in ('quantization_config', 'batch_input_shape')])
            print('layer1 quantization_config', l1.get('config', {}).get('quantization_config'))
    print('input_layers', cfg.get('config', {}).get('input_layers'))
