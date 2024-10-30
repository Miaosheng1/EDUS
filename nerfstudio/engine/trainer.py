# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""
from __future__ import annotations

from rich.console import Console
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.engine.optimizers import Optimizers, setup_optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline


CONSOLE = Console(width=120)


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.

    """

    pipeline: VanillaPipeline
    optimizers: Optimizers

    def __init__(self, config: cfg.Config, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision = self.config.trainer.mixed_precision
        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.mixed_precision)
        self.checkpoint_dir = config.get_checkpoint_dir()

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val",config_path = None):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datset into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device, test_mode=test_mode, world_size=self.world_size, local_rank=self.local_rank,config_path = config_path
        )
        self.optimizers = setup_optimizers(self.config, self.pipeline.get_param_groups())

        self._load_checkpoint()


    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.trainer.load_dir
        if load_dir is not None:
            load_step = self.config.trainer.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            self.pipeline.load_pipeline(loaded_state["pipeline"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.trainer.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")