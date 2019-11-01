from util import *
import model as module_model
from config import *


if __name__ == '__main__':
    config = Config()
    model = getattr(module_model, config.MODEL)(config)
    input_shape = (15,50)
    model_visual(model,input_shape)