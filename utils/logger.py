"""
Utilitaire de logging, complète config.logger (voir config.py) avec un writer
TensorBoard optionnel.
"""
import os
from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir, phase_name):
        self.dir = os.path.join(log_dir, phase_name)
        os.makedirs(self.dir, exist_ok=True)
        self.writer = SummaryWriter(self.dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_prefix, scalar_dict, step):
        for k, v in scalar_dict.items():
            self.writer.add_scalar(f'{tag_prefix}/{k}', v, step)

    def log_image(self, tag, image_tensor, step):
        self.writer.add_image(tag, image_tensor, step)

    def close(self):
        self.writer.close()
