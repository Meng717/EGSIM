import sys
import os
import yaml

def set_env():
    # Add project root to sys.path
    package_names = ['configs', 'models', 'modules']
    current_file_path = os.path.abspath(__file__)
    project_root_path = os.path.dirname(current_file_path)
    if os.path.isdir(project_root_path) and project_root_path not in sys.path:
        sys.path.append(project_root_path)
        print(f"Added {project_root_path} to sys.path")
    for package_name in package_names:
        package_path = os.path.join(project_root_path, package_name)
        if os.path.isdir(package_path) and package_path not in sys.path:
            sys.path.append(package_path)
            print(f"Added {package_path} to sys.path")
    with open(os.path.join(project_root_path, 'configs', 'include_path.yaml')) as f:
        for model_path in yaml.safe_load(f)['include_path']:
            sys.path.append(model_path)
            print(f"Added {model_path} to sys.path")

set_env()

from modules.gui.egsim_gui import EGSIMWindow
from PySide6.QtWidgets import QApplication

app = QApplication()

window = EGSIMWindow()
window.show()

sys.exit(app.exec())