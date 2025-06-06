# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

# ruff: noqa: E402
import os
import sys
from pathlib import Path

import pytest
import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common

from physicsnemo.models.diffusion import StormCastUNet, UNet


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_forwards(device):
    """Test forward passes of UNet wrappers"""

    # Construct the UNet model
    res, inc, outc = 64, 2, 3
    model = UNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
    ).to(device)
    input_image = torch.ones([1, inc, res, res]).to(device)
    lr_image = torch.randn([1, outc, res, res]).to(device)
    output = model(x=input_image, img_lr=lr_image)
    assert output.shape == (1, outc, res, res)

    # Construct the StormCastUNet model
    model = StormCastUNet(
        img_resolution=res, img_in_channels=inc, img_out_channels=outc
    ).to(device)
    input_image = torch.ones([1, inc, res, res]).to(device)
    output = model(x=input_image)
    assert output.shape == (1, outc, res, res)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_optims(device):
    """Test optimizations of U-Net wrappers"""

    res, inc, outc = 64, 2, 3

    def setup_model():

        model = UNet(
            img_resolution=res,
            img_in_channels=inc,
            img_out_channels=outc,
            model_type="SongUNet",
        ).to(device)
        input_image = torch.ones([1, inc, res, res]).to(device)
        lr_image = torch.randn([1, outc, res, res]).to(device)

        return model, [input_image, lr_image]

    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))

    def setup_model():
        model = StormCastUNet(
            img_resolution=res, img_in_channels=inc, img_out_channels=outc
        ).to(device)
        input_image = torch.ones([1, inc, res, res]).to(device)

        return model, [input_image]

    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_checkpoint(device):
    """Test UNet wrapper checkpoint save/load"""
    # Construct UNet models
    res, inc, outc = 64, 2, 3
    model_1 = UNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
    ).to(device)
    model_2 = UNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
    ).to(device)

    input_image = torch.ones([1, inc, res, res]).to(device)
    lr_image = torch.randn([1, outc, res, res]).to(device)
    assert common.validate_checkpoint(model_1, model_2, (*[input_image, lr_image],))

    # Construct StormCastUNet models
    res, inc, outc = 64, 2, 3
    model_1 = StormCastUNet(
        img_resolution=res, img_in_channels=inc, img_out_channels=outc
    ).to(device)
    model_2 = StormCastUNet(
        img_resolution=res, img_in_channels=inc, img_out_channels=outc
    ).to(device)

    input_image = torch.ones([1, inc, res, res]).to(device)
    assert common.validate_checkpoint(model_1, model_2, (input_image,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_amp_mode_property(device):
    """Test UNet wrappers amp_mode property"""

    res, inc, outc = 32, 1, 1

    model = UNet(
        img_resolution=res,
        img_in_channels=inc,
        img_out_channels=outc,
        model_type="SongUNet",
    ).to(device)

    # Getter should reflect underlying model value (default False)
    assert model.amp_mode is False

    # Set to True and verify propagation
    model.amp_mode = True
    assert model.amp_mode is True
    if hasattr(model.model, "amp_mode"):
        assert model.model.amp_mode is True
    for sub in model.model.modules():
        if hasattr(sub, "amp_mode"):
            assert sub.amp_mode is True

    # Toggle back to False and verify again
    model.amp_mode = False
    assert model.amp_mode is False
    if hasattr(model.model, "amp_mode"):
        assert model.model.amp_mode is False
    for sub in model.model.modules():
        if hasattr(sub, "amp_mode"):
            assert sub.amp_mode is False


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet_backward_compat(device):
    """Test backward compatibility of UNet wrappers"""

    # Construct Load UNet from older version
    UNet.from_checkpoint(
        file_name=(
            str(
                Path(__file__).parents[1].resolve()
                / Path("data")
                / Path("diffusion_unet_0.1.0.mdlus")
            )
        )
    )
