import os
import yaml
import importlib
import re

def getModels() -> dict:
    models = {}
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    for file in os.listdir(configs_dir):
        if file.endswith(".yaml") or file.endswith(".yml"):
            path = os.path.join(configs_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                if "model_name" in cfg:
                    models[cfg["model_name"]] = path
    return models


def importModels(model_config_file_path) -> list:

    def load_class(full_class_path):
        module_name, class_name = full_class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    with open(model_config_file_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_items = []
    for key, val in cfg.items():
        if key.startswith("model_"):
            match = re.match(r"model_(\d+)", key)
            if not match:
                continue
            order = int(match.group(1))

            modality = val.get("modality")
            if isinstance(modality, str):
                modality = [modality]
            model_dict = {
                "modality": modality,
                "render_model_loader": {},
                "sensor_loader": {},
                "renderer": {}
            }

            # render_model_loader
            if "render_model_loader" in val and "simple_loader" not in val:
                model_dict["render_model_loader"] = {
                    "info": load_class(val["render_model_loader"]["info"]),
                    "object": load_class(val["render_model_loader"]["object"])
                }
            elif "simple_loader" in val and "render_model_loader" not in val:
                model_dict["simple_loader"] = load_class(val["simple_loader"])
            else:
                raise TypeError("Both 'render_model_loader' and 'simple_loader' found!")

            # sensor_loader
            if "sensor_loader" in val and "simple_loader" not in val:
                model_dict["sensor_loader"] = {
                    "info": load_class(val["sensor_loader"]["info"]),
                    "object": load_class(val["sensor_loader"]["object"])
                }

            # renderer
            if "renderer" in val and "simple_renderer" not in val:
                model_dict["renderer"] = {
                    "info": load_class(val["renderer"]["info"]),
                    "object": load_class(val["renderer"]["object"])
                }
            elif "simple_renderer" in val and "renderer" not in val:
                model_dict["simple_renderer"] = load_class(val["simple_renderer"])
            else:
                raise TypeError("Both 'renderer' and 'simple_renderer' found!")
            

            model_items.append((order, model_dict))

    model_items.sort(key=lambda x: x[0])
    return [m[1] for m in model_items]
