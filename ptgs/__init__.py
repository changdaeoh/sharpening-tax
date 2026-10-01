# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""PTGS -- Posterior-Tempered Group Sampling.

``controller`` is the method, ``integration`` the trainer call sites; ``python -m ptgs`` demos it.
"""

from .controller import PTGS, pivot_at, temperature_from_p
from .integration import PTGSHooks

__all__ = ["PTGS", "PTGSHooks", "pivot_at", "temperature_from_p"]
