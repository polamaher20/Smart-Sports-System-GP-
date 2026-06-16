import h5py
import json

path = r"C:\Users\new castle\Desktop\last update\graduation-project\smart-sports-backend\app\infrastructure\ml_models\nutrition_model.h5"

with h5py.File(path, 'r+') as f:
    raw = f.attrs['model_config']
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8')
    config = json.loads(raw)

    def fix(node):
        if isinstance(node, dict):
            if isinstance(node.get('dtype'), dict) and node['dtype'].get('class_name') == 'DTypePolicy':
                node['dtype'] = node['dtype'].get('config', {}).get('name', 'float32')
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for item in node:
                fix(item)

    fix(config)
    f.attrs['model_config'] = json.dumps(config)

print("تم التعديل بنجاح")