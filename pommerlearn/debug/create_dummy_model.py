from pathlib import Path

import training.train_cnn

train_config = training.train_cnn.fill_default_config({})
training.train_cnn.export_initial_model(train_config, Path("./model"))
