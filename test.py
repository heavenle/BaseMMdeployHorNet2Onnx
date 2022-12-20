import argparse
import logging
import os
import os.path as osp

from mmdeploy.apis import (extract_model, get_predefined_partition_cfg,
                           torch2onnx)
from mmdeploy.utils import (get_ir_config, get_partition_config,
                            get_root_logger, load_config)
