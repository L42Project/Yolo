import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
import config as cfg
import common

params=tf.experimental.tensorrt.ConversionParams(
  #max_workspace_size_bytes=1 << 32,
  precision_mode='FP16')
  #maximum_cached_engines=16)

converter=tf.experimental.tensorrt.Converter(input_saved_model_dir=cfg.train_model, conversion_params=params)
#converter=trt.TrtGraphConverterV2(input_saved_model_dir=cfg.train_model)
converter.convert()

def my_input_fn():
  inp1=np.random.normal(size=(1, 128, 128, 3)).astype(np.float32)
  yield inp1

converter.build(input_fn=my_input_fn)
converter.save(cfg.fast_train_model)
