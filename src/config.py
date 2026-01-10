# config.py
import yaml
import os
import logging

class Config:
    """Configuration manager that loads from params.yaml"""

    def __init__(self, params_path='params.yaml'):
        self.params_path = params_path
        self.params = self.load_params()

    def load_params(self):
        """Load parameters from params.yaml"""
        if not os.path.exists(self.params_path):
            logging.warning(f"params.yaml not found at {self.params_path}. Using defaults.")
            return self.get_default_params()

        with open(self.params_path, 'r') as f:
            params = yaml.safe_load(f)
        return params

    def get_default_params(self):
        """Default parameters if params.yaml doesn't exist"""
        return {
            'train': {
                'batch_size': 16,
                'max_epochs': 5,
                'learning_rate': 0.00001,
                'freeze_backbone': True,
                'patience': 10,
                'k_folds': 5
            },
            'model': {
                'backbone': 'resnet18',
                'num_classes': 2,
                'class_names': ['sea', 'forest']
            },
            'data': {
                'data_path': './data/',
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1
            },
            'paths': {
                'model_dir': './models/',
                'plots_dir': './plots/',
                'metrics_dir': './metrics/'
            },
            'mlflow': {
                'tracking_uri': 'http://localhost:5000',
                'experiment_name': 'pytorch-classification'
            },
            'dvc': {
                'remote': 'myremote',
                'enabled': True
            }
        }

    @property
    def BATCH_SIZE(self):
        return self.params['train']['batch_size']

    @property
    def MAX_EPOCHS_NUM(self):
        return self.params['train']['max_epochs']

    @property
    def FREEZE_BACKBONE(self):
        return self.params['train']['freeze_backbone']

    @property
    def CLASS_NAMES(self):
        return self.params['model']['class_names']

    @property
    def BACKBONE(self):
        return self.params['model']['backbone']

    @property
    def MODEL_DIR(self):
        return self.params['paths']['model_dir']

    @property
    def PLOTS_DIR(self):
        return self.params['paths']['plots_dir']

    @property
    def METRICS_DIR(self):
        return self.params['paths'].get('metrics_dir', './metrics/')

    def update_param(self, section, key, value):
        """Update a parameter value"""
        if section in self.params and key in self.params[section]:
            self.params[section][key] = value
            return True
        return False

# Create global config instance
config = Config()

BATCH_SIZE = config.BATCH_SIZE
MAX_EPOCHS_NUM = config.MAX_EPOCHS_NUM
FREEZE_BACKBONE = config.FREEZE_BACKBONE
CLASS_NAMES = config.CLASS_NAMES
BACKBONE = config.BACKBONE
MODEL_DIR = config.MODEL_DIR
PLOTS_DIR = config.PLOTS_DIR
