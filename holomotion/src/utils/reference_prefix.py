# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


from typing import Mapping


def resolve_reference_tensor_key(
    batch_tensors: Mapping[str, object],
    base_key: str,
    prefix: str = "ref_",
) -> str:
    tensor_key = base_key
    if prefix:
        prefixed_key = f"{prefix}{base_key}"
        if prefixed_key in batch_tensors:
            tensor_key = prefixed_key
        elif prefix == "ft_ref_":
            raise KeyError(
                f"Filtered tensor '{prefixed_key}' is not present in the "
                "current motion cache batch. Ensure online filtering is "
                "enabled and 'ft_ref_' is materialized in allowed_prefixes."
            )
        elif base_key not in batch_tensors:
            raise KeyError(
                f"Neither '{prefixed_key}' nor '{base_key}' is present in "
                "the current motion cache batch."
            )
    elif base_key not in batch_tensors:
        raise KeyError(
            f"Tensor '{base_key}' is not present in the current motion cache batch."
        )
    return tensor_key
