import torch
import numpy as np
from framework_activity_recognition.processing import normalize_color_input_zero_center_unit_range, \
    normalize_color_input_zero_center_unit_range_per_channel,unit_range_zero_center_to_unit_range_zero_min,\
        center_crop, random_crop, random_select, random_horizontal_flip

class normalizeColorInputZeroCenterUnitRange(object):
    def __init__(self, max_val = 255.0):

        self.max_val = max_val


    def __call__(self, input_tensor):
        result = normalize_color_input_zero_center_unit_range(input_tensor, max_val = self.max_val)

        return result

class normalizeColorInputZeroCenterUnitRangeChannelWise(object):

    def __call__(self, input_tensor):
        result = normalize_color_input_zero_center_unit_range_per_channel(input_tensor)

        return result

class unitRangeZeroCenterToUnitRangeZeroMin(object):

    def __call__(self, input_tensor):
        result = unit_range_zero_center_to_unit_range_zero_min(input_tensor)

        return result

class CenterCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, input_tensor):

        result = center_crop(input_tensor, self.height, self.width)

        return result


class RandomCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, input_tensor):

        result = random_crop(input_tensor, self.height, self.width)

        return result


class RandomSelect(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, input_tensor):

        result = random_select(input_tensor, self.n)

        return result


class RandomHorizontalFlip(object):
    def __init__(self):
        super().__init__()

    def __call__(self, input_tensor):

        result = random_horizontal_flip(input_tensor)

        return result


class ToTensor(object):
    def __call__(self, input_tensor):


        # Swap color channels axis because
        # numpy frames: Frames ID x Height x Width x Channels
        # torch frames: Channels x Frame ID x Height x Width

        result = input_tensor.transpose(3, 0, 1, 2)
        result = np.float32(result)

        return torch.from_numpy(result)

