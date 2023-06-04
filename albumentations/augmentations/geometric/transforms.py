import math
import random
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox

from ... import random_utils
from ...core.transforms_interface import (
    BoxInternalType,
    DualTransform,
    ImageColorType,
    KeypointInternalType,
    ScaleFloatType,
    to_tuple,
)
from ..functional import bbox_from_mask
from . import functional as F

__all__ = [
    "ShiftScaleRotate",
    "ElasticTransform",
    "Perspective",
    "VerticalFlip",
    "HorizontalFlip",
    "Flip",
    "Transpose",
    "OpticalDistortion",
    "GridDistortion",
    "PadIfNeeded",
]


class ShiftScaleRotate(DualTransform):
    """Randomly apply affine transforms: translate, scale and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        shift_limit_x ((float, float) or float): shift factor range for width. If it is set then this value
            instead of shift_limit will be used for shifting width.  If shift_limit_x is a single float value,
            the range will be (-shift_limit_x, shift_limit_x). Absolute values for lower and upper bounds should lie in
            the range [0, 1]. Default: None.
        shift_limit_y ((float, float) or float): shift factor range for height. If it is set then this value
            instead of shift_limit will be used for shifting height.  If shift_limit_y is a single float value,
            the range will be (-shift_limit_y, shift_limit_y). Absolute values for lower and upper bounds should lie
            in the range [0, 1]. Default: None.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        shift_limit_x=None,
        shift_limit_y=None,
        rotate_method="largest_box",
        always_apply=False,
        p=0.5,
    ):
        super(ShiftScaleRotate, self).__init__(always_apply, p)
        self.shift_limit_x = to_tuple(shift_limit_x if shift_limit_x is not None else shift_limit)
        self.shift_limit_y = to_tuple(shift_limit_y if shift_limit_y is not None else shift_limit)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method

        if self.rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")

    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def apply_to_keypoint(self, keypoint, angle=0, scale=0, dx=0, dy=0, rows=0, cols=0, **params):
        return F.keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols)

    def get_params(self):
        return {
            "angle": random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
            "scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
            "dx": random.uniform(self.shift_limit_x[0], self.shift_limit_x[1]),
            "dy": random.uniform(self.shift_limit_y[0], self.shift_limit_y[1]),
        }

    def apply_to_bbox(self, bbox, angle, scale, dx, dy, **params):
        return F.bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, self.rotate_method, **params)

    def get_transform_init_args(self):
        return {
            "shift_limit_x": self.shift_limit_x,
            "shift_limit_y": self.shift_limit_y,
            "scale_limit": to_tuple(self.scale_limit, bias=-1.0),
            "rotate_limit": self.rotate_limit,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "value": self.value,
            "mask_value": self.mask_value,
            "rotate_method": self.rotate_method,
        }


class ElasticTransform(DualTransform):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        alpha (float):
        sigma (float): Gaussian filter parameter.
        alpha_affine (float): The range will be (-alpha_affine, alpha_affine)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.
        same_dxdy (boolean): Whether to use same random generated shift for x and y.
                             Enabling this option gives ~2X speedup.

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        alpha=1,
        sigma=50,
        alpha_affine=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        approximate=False,
        same_dxdy=False,
        p=0.5,
    ):
        super(ElasticTransform, self).__init__(always_apply, p)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.approximate = approximate
        self.same_dxdy = same_dxdy

    def apply(self, img, random_state=None, interpolation=cv2.INTER_LINEAR, **params):
        return F.elastic_transform(
            img,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            interpolation,
            self.border_mode,
            self.value,
            np.random.RandomState(random_state),
            self.approximate,
            self.same_dxdy,
        )

    def apply_to_mask(self, img, random_state=None, **params):
        return F.elastic_transform(
            img,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_state),
            self.approximate,
            self.same_dxdy,
        )

    def apply_to_bbox(self, bbox, random_state=None, **params):
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = F.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.elastic_transform(
            mask,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_state),
            self.approximate,
        )
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def get_params(self):
        return {"random_state": random.randint(0, 10000)}

    def get_transform_init_args_names(self):
        return (
            "alpha",
            "sigma",
            "alpha_affine",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
            "approximate",
            "same_dxdy",
        )


class Perspective(DualTransform):
    """Perform a random four point perspective transform of the input.

    Args:
        scale (float or (float, float)): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners.
            If scale is a single float value, the range will be (0, scale). Default: (0.05, 0.1).
        keep_size (bool): Whether to resize image’s back to their original size after applying the perspective
            transform. If set to False, the resulting images may end up having different shapes
            and will always be a list, never an array. Default: True
        pad_mode (OpenCV flag): OpenCV border mode.
        pad_val (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0
        mask_pad_val (int, float, list of int, list of float): padding value for mask
            if border_mode is cv2.BORDER_CONSTANT. Default: 0
        fit_output (bool): If True, the image plane size and position will be adjusted to still capture
            the whole image after perspective transformation. (Followed by image resizing if keep_size is set to True.)
            Otherwise, parts of the transformed image may be outside of the image plane.
            This setting should not be set to True when using large scale values as it could lead to very large images.
            Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale=(0.05, 0.1),
        keep_size=True,
        pad_mode=cv2.BORDER_CONSTANT,
        pad_val=0,
        mask_pad_val=0,
        fit_output=False,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = to_tuple(scale, 0)
        self.keep_size = keep_size
        self.pad_mode = pad_mode
        self.pad_val = pad_val
        self.mask_pad_val = mask_pad_val
        self.fit_output = fit_output
        self.interpolation = interpolation

    def apply(self, img, matrix=None, max_height=None, max_width=None, **params):
        return F.perspective(
            img, matrix, max_width, max_height, self.pad_val, self.pad_mode, self.keep_size, params["interpolation"]
        )

    def apply_to_bbox(self, bbox, matrix=None, max_height=None, max_width=None, **params):
        return F.perspective_bbox(bbox, params["rows"], params["cols"], matrix, max_width, max_height, self.keep_size)

    def apply_to_keypoint(self, keypoint, matrix=None, max_height=None, max_width=None, **params):
        return F.perspective_keypoint(
            keypoint, params["rows"], params["cols"], matrix, max_width, max_height, self.keep_size
        )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]

        scale = random_utils.uniform(*self.scale)
        points = random_utils.normal(0, scale, [4, 2])
        points = np.mod(np.abs(points), 1)

        # top left -- no changes needed, just use jitter
        # top right
        points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
        # bottom right
        points[2] = 1.0 - points[2]  # w = 1.0 - jitt
        # bottom left
        points[3, 1] = 1.0 - points[3, 1]  # h = 1.0 - jitter

        points[:, 0] *= w
        points[:, 1] *= h

        # Obtain a consistent order of the points and unpack them individually.
        # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
        # here, because the reordered points is used further below.
        points = self._order_points(points)
        tl, tr, br, bl = points

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        min_width = None
        max_width = None
        while min_width is None or min_width < 2:
            width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            max_width = int(max(width_top, width_bottom))
            min_width = int(min(width_top, width_bottom))
            if min_width < 2:
                step_size = (2 - min_width) / 2
                tl[0] -= step_size
                tr[0] += step_size
                bl[0] -= step_size
                br[0] += step_size

        # compute the height of the new image, which will be the maximum distance between the top-right
        # and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        min_height = None
        max_height = None
        while min_height is None or min_height < 2:
            height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = int(max(height_right, height_left))
            min_height = int(min(height_right, height_left))
            if min_height < 2:
                step_size = (2 - min_height) / 2
                tl[1] -= step_size
                tr[1] -= step_size
                bl[1] += step_size
                br[1] += step_size

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        dst = np.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]], dtype=np.float32)

        # compute the perspective transform matrix and then apply it
        m = cv2.getPerspectiveTransform(points, dst)

        if self.fit_output:
            m, max_width, max_height = self._expand_transform(m, (h, w))

        return {"matrix": m, "max_height": max_height, "max_width": max_width, "interpolation": self.interpolation}

    @classmethod
    def _expand_transform(cls, matrix, shape):
        height, width = shape
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2, max_height
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        dst = cv2.perspectiveTransform(np.array([rect]), matrix)[0]

        # get min x, y over transformed 4 points
        # then modify target points by subtracting these minima  => shift to (0, 0)
        dst -= dst.min(axis=0, keepdims=True)
        dst = np.around(dst, decimals=0)

        matrix_expanded = cv2.getPerspectiveTransform(rect, dst)
        max_width, max_height = dst.max(axis=0)
        return matrix_expanded, int(max_width), int(max_height)

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        pts = np.array(sorted(pts, key=lambda x: x[0]))
        left = pts[:2]  # points with smallest x coordinate - left points
        right = pts[2:]  # points with greatest x coordinate - right points

        if left[0][1] < left[1][1]:
            tl, bl = left
        else:
            bl, tl = left

        if right[0][1] < right[1][1]:
            tr, br = right
        else:
            br, tr = right

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def get_transform_init_args_names(self):
        return "scale", "keep_size", "pad_mode", "pad_val", "mask_pad_val", "fit_output", "interpolation"


class PadIfNeeded(DualTransform):
    """Pad side of the image / max if side is less than desired number.

    Args:
        min_height (int): minimal result image height.
        min_width (int): minimal result image width.
        pad_height_divisor (int): if not None, ensures image height is dividable by value of this argument.
        pad_width_divisor (int): if not None, ensures image width is dividable by value of this argument.
        position (Union[str, PositionType]): Position of the image. should be PositionType.CENTER or
            PositionType.TOP_LEFT or PositionType.TOP_RIGHT or PositionType.BOTTOM_LEFT or PositionType.BOTTOM_RIGHT.
            or PositionType.RANDOM. Default: PositionType.CENTER.
        border_mode (OpenCV flag): OpenCV border mode.
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value for mask if border_mode is cv2.BORDER_CONSTANT.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        image, mask, bbox, keypoints

    Image types:
        uint8, float32
    """

    class PositionType(Enum):
        CENTER = "center"
        TOP_LEFT = "top_left"
        TOP_RIGHT = "top_right"
        BOTTOM_LEFT = "bottom_left"
        BOTTOM_RIGHT = "bottom_right"
        RANDOM = "random"

    def __init__(
        self,
        min_height: Optional[int] = 1024,
        min_width: Optional[int] = 1024,
        pad_height_divisor: Optional[int] = None,
        pad_width_divisor: Optional[int] = None,
        position: Union[PositionType, str] = PositionType.CENTER,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[ImageColorType] = None,
        mask_value: Optional[ImageColorType] = None,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        if (min_height is None) == (pad_height_divisor is None):
            raise ValueError("Only one of 'min_height' and 'pad_height_divisor' parameters must be set")

        if (min_width is None) == (pad_width_divisor is None):
            raise ValueError("Only one of 'min_width' and 'pad_width_divisor' parameters must be set")

        super(PadIfNeeded, self).__init__(always_apply, p)
        self.min_height = min_height
        self.min_width = min_width
        self.pad_width_divisor = pad_width_divisor
        self.pad_height_divisor = pad_height_divisor
        self.position = PadIfNeeded.PositionType(position)
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params["rows"]
        cols = params["cols"]

        if self.min_height is not None:
            if rows < self.min_height:
                h_pad_top = int((self.min_height - rows) / 2.0)
                h_pad_bottom = self.min_height - rows - h_pad_top
            else:
                h_pad_top = 0
                h_pad_bottom = 0
        else:
            pad_remained = rows % self.pad_height_divisor
            pad_rows = self.pad_height_divisor - pad_remained if pad_remained > 0 else 0

            h_pad_top = pad_rows // 2
            h_pad_bottom = pad_rows - h_pad_top

        if self.min_width is not None:
            if cols < self.min_width:
                w_pad_left = int((self.min_width - cols) / 2.0)
                w_pad_right = self.min_width - cols - w_pad_left
            else:
                w_pad_left = 0
                w_pad_right = 0
        else:
            pad_remainder = cols % self.pad_width_divisor
            pad_cols = self.pad_width_divisor - pad_remainder if pad_remainder > 0 else 0

            w_pad_left = pad_cols // 2
            w_pad_right = pad_cols - w_pad_left

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = self.__update_position_params(
            h_top=h_pad_top, h_bottom=h_pad_bottom, w_left=w_pad_left, w_right=w_pad_right
        )

        params.update(
            {
                "pad_top": h_pad_top,
                "pad_bottom": h_pad_bottom,
                "pad_left": w_pad_left,
                "pad_right": w_pad_right,
            }
        )
        return params

    def apply(
        self, img: np.ndarray, pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0, **params
    ) -> np.ndarray:
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.value,
        )

    def apply_to_mask(
        self, img: np.ndarray, pad_top: int = 0, pad_bottom: int = 0, pad_left: int = 0, pad_right: int = 0, **params
    ) -> np.ndarray:
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=self.mask_value,
        )

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        rows: int = 0,
        cols: int = 0,
        **params
    ) -> BoxInternalType:
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)[:4]
        bbox = x_min + pad_left, y_min + pad_top, x_max + pad_left, y_max + pad_top
        return normalize_bbox(bbox, rows + pad_top + pad_bottom, cols + pad_left + pad_right)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        **params
    ) -> KeypointInternalType:
        x, y, angle, scale = keypoint[:4]
        return x + pad_left, y + pad_top, angle, scale

    def get_transform_init_args_names(self):
        return (
            "min_height",
            "min_width",
            "pad_height_divisor",
            "pad_width_divisor",
            "border_mode",
            "value",
            "mask_value",
        )

    def __update_position_params(
        self, h_top: int, h_bottom: int, w_left: int, w_right: int
    ) -> Tuple[int, int, int, int]:
        if self.position == PadIfNeeded.PositionType.TOP_LEFT:
            h_bottom += h_top
            w_right += w_left
            h_top = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.TOP_RIGHT:
            h_bottom += h_top
            w_left += w_right
            h_top = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_LEFT:
            h_top += h_bottom
            w_right += w_left
            h_bottom = 0
            w_left = 0

        elif self.position == PadIfNeeded.PositionType.BOTTOM_RIGHT:
            h_top += h_bottom
            w_left += w_right
            h_bottom = 0
            w_right = 0

        elif self.position == PadIfNeeded.PositionType.RANDOM:
            h_pad = h_top + h_bottom
            w_pad = w_left + w_right
            h_top = random.randint(0, h_pad)
            h_bottom = h_pad - h_top
            w_left = random.randint(0, w_pad)
            w_right = w_pad - w_left

        return h_top, h_bottom, w_left, w_right


class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.vflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_vflip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_vflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return F.hflip_cv2(img)

        return F.hflip(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_hflip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_hflip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Flip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, d: int = 0, **params) -> np.ndarray:
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        return F.random_flip(img, d)

    def get_params(self):
        # Random int in the range [-1, 1]
        return {"d": random.randint(-1, 1)}

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_flip(bbox, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_flip(keypoint, **params)

    def get_transform_init_args_names(self):
        return ()


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return F.transpose(img)

    def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
        return F.bbox_transpose(bbox, 0, **params)

    def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
        return F.keypoint_transpose(keypoint)

    def get_transform_init_args_names(self):
        return ()


class OpticalDistortion(DualTransform):
    """
    Args:
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.05, 0.05).
        shift_limit (float, (float, float))): If shift_limit is a single float, the range
            will be (-shift_limit, shift_limit). Default: (-0.05, 0.05).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        distort_limit: ScaleFloatType = 0.05,
        shift_limit: ScaleFloatType = 0.05,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[ImageColorType] = None,
        mask_value: Optional[ImageColorType] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(OpticalDistortion, self).__init__(always_apply, p)
        self.shift_limit = to_tuple(shift_limit)
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(
        self, img: np.ndarray, k: int = 0, dx: int = 0, dy: int = 0, interpolation: int = cv2.INTER_LINEAR, **params
    ) -> np.ndarray:
        return F.optical_distortion(img, k, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img: np.ndarray, k: int = 0, dx: int = 0, dy: int = 0, **params) -> np.ndarray:
        return F.optical_distortion(img, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def apply_to_bbox(self, bbox: BoxInternalType, k: int = 0, dx: int = 0, dy: int = 0, **params) -> BoxInternalType:
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = F.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.optical_distortion(mask, k, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def get_params(self):
        return {
            "k": random.uniform(self.distort_limit[0], self.distort_limit[1]),
            "dx": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
            "dy": round(random.uniform(self.shift_limit[0], self.shift_limit[1])),
        }

    def get_transform_init_args_names(self):
        return (
            "distort_limit",
            "shift_limit",
            "interpolation",
            "border_mode",
            "value",
            "mask_value",
        )


class GridDistortion(DualTransform):
    """
    Args:
        num_steps (int): count of grid cells on each side.
        distort_limit (float, (float, float)): If distort_limit is a single float, the range
            will be (-distort_limit, distort_limit). Default: (-0.03, 0.03).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        normalized (bool): if true, distortion will be normalized to do not go outside the image. Default: False
            See for more information: https://github.com/albumentations-team/albumentations/pull/722

    Targets:
        image, mask

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        num_steps: int = 5,
        distort_limit: ScaleFloatType = 0.3,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[ImageColorType] = None,
        mask_value: Optional[ImageColorType] = None,
        normalized: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(GridDistortion, self).__init__(always_apply, p)
        self.num_steps = num_steps
        self.distort_limit = to_tuple(distort_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.normalized = normalized

    def apply(
        self, img: np.ndarray, stepsx: Tuple = (), stepsy: Tuple = (), interpolation: int = cv2.INTER_LINEAR, **params
    ) -> np.ndarray:
        return F.grid_distortion(img, self.num_steps, stepsx, stepsy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img: np.ndarray, stepsx: Tuple = (), stepsy: Tuple = (), **params) -> np.ndarray:
        return F.grid_distortion(
            img, self.num_steps, stepsx, stepsy, cv2.INTER_NEAREST, self.border_mode, self.mask_value
        )

    def apply_to_bbox(self, bbox: BoxInternalType, stepsx: Tuple = (), stepsy: Tuple = (), **params) -> BoxInternalType:
        rows, cols = params["rows"], params["cols"]
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = F.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.grid_distortion(
            mask, self.num_steps, stepsx, stepsy, cv2.INTER_NEAREST, self.border_mode, self.mask_value
        )
        bbox_returned = bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned

    def _normalize(self, h, w, xsteps, ysteps):

        # compensate for smaller last steps in source image.
        x_step = w // self.num_steps
        last_x_step = min(w, ((self.num_steps + 1) * x_step)) - (self.num_steps * x_step)
        xsteps[-1] *= last_x_step / x_step

        y_step = h // self.num_steps
        last_y_step = min(h, ((self.num_steps + 1) * y_step)) - (self.num_steps * y_step)
        ysteps[-1] *= last_y_step / y_step

        # now normalize such that distortion never leaves image bounds.
        tx = w / math.floor(w / self.num_steps)
        ty = h / math.floor(h / self.num_steps)
        xsteps = np.array(xsteps) * (tx / np.sum(xsteps))
        ysteps = np.array(ysteps) * (ty / np.sum(ysteps))

        return {"stepsx": xsteps, "stepsy": ysteps}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]

        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for _ in range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for _ in range(self.num_steps + 1)]

        if self.normalized:
            return self._normalize(h, w, stepsx, stepsy)

        return {"stepsx": stepsx, "stepsy": stepsy}

    def get_transform_init_args_names(self):
        return "num_steps", "distort_limit", "interpolation", "border_mode", "value", "mask_value", "normalized"
