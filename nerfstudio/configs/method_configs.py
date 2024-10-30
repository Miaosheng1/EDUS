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
Put all the method implementations in one location.
"""

from __future__ import annotations

from typing import Dict

import tyro

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import (
    Config,
    SchedulerConfig,
    TrainerConfig,
    ViewerConfig,
)
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManagerConfig,
)


from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    MultiStepSchedulerConfig
)

from nerfstudio.data.dataparsers.zeronpt_dataparser import ZeronptDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from nerfstudio.models.neuralpoint import EDUSConfig
from nerfstudio.models.nerfacto import GVSNerfModelConfig

from nerfstudio.pipelines.base_pipeline import (
    VanillaPipelineConfig,
)


method_configs: Dict[str, Config] = {}
descriptions = {
    "nerfacto": "Recommended real-time model tuned for real captures.",
    "neuralpnt": "Implementation of EDUS. This model will be continually updated.",
}

method_configs["edus"] = Config(
    method_name="NeuralPoint",
    trainer=TrainerConfig(
        steps_per_eval_batch=1000000, steps_per_save=1000, max_num_iterations=10000, mixed_precision=True,steps_per_eval_image=50000,
        steps_per_eval_all_images=1000000,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=ZeronptDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=EDUSConfig(eval_num_rays_per_chunk=1 << 15),
    ),
    optimizers={
        "voxel_encoder": {
            "optimizer": AdamOptimizerConfig(lr=0.5*1e-2, eps=1e-15),
            "scheduler": SchedulerConfig(lr_final=0.0001,max_steps=500000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=0.1*1e-2, eps=1e-15),
            "scheduler": SchedulerConfig(lr_final=0.0001,max_steps=500000),
        },
        "field_background": {
        "optimizer": AdamOptimizerConfig(lr=0.05*1e-2, eps=1e-15),
        "scheduler": SchedulerConfig(lr_final=0.0001,max_steps=500000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="tensorboard",
)

method_configs["nerfacto"] = Config(
    method_name="nerfacto",
    trainer=TrainerConfig(
        steps_per_eval_batch=100000, steps_per_save=2000, max_num_iterations=20000, mixed_precision=True,steps_per_eval_image=50000,
        steps_per_eval_all_images=100000,
    ),
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            camera_optimizer=CameraOptimizerConfig(
                mode="off", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=GVSNerfModelConfig(eval_num_rays_per_chunk=1 << 15,
                                max_res=1024,
                                log2_hashmap_size=19),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="tensorboard",
)


AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
