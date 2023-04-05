from pathlib import Path
from typing import Any, Optional

import napari
import numpy as np
import torch
from magicgui.widgets import Container, FileEdit, create_widget
from napari.layers import Image
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
from skimage import color, util

# from segment_anything.utils.transforms import ResizeLongestSide


class SAMWidget(Container):
    _sam: Sam
    _predictor: SamPredictor

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        self._model_type = "default"
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._weights_file = FileEdit(filter="*.pth", label="Model weights")
        self._weights_file.changed.connect(self._load_checkpoint)
        self.append(self._weights_file)

        self._im_layer_widget = create_widget(annotation=Image, label="Image:")
        self._im_layer_widget.changed.connect(self._load_image)
        self.append(self._im_layer_widget)

        self._mask_layer = self._viewer.add_labels(
            data=np.zeros((256, 256), dtype=int)
        )

        self._pts_layer = self._viewer.add_points(name="SAM points")
        self._pts_layer.events.data.connect(self._on_run)
        # self._boxes_layer = self._viewer.add_shapes(name="SAM boxes")

        self._logits: Optional[torch.TensorType] = None

    @property
    def model_type(self) -> str:
        return self._model_type

    @model_type.setter
    def model_type(self, value: str) -> None:
        if value not in sam_model_registry:
            raise ValueError(
                f"Model type {value} not found. Expected {list(sam_model_registry.keys())}"
            )
        # NOTE: weights and model are not updated until checkpoint is loaded
        self._model_type = value

    def _load_checkpoint(self, ckpt_path: Path) -> None:
        self._sam = sam_model_registry[self._model_type](ckpt_path)
        self._sam.to(self._device)
        self._predictor = SamPredictor(self._sam)
        self._load_image(self._im_layer_widget.value)

    def _load_image(self, im_layer: Optional[Image]) -> None:
        if im_layer is None or not hasattr(self, "_sam"):
            return

        image = im_layer.data
        if image.ndim == 2:
            image = color.gray2rgb(image)

        elif image.ndim == 3 and image.shape[-1] == 4:
            image = color.rgba2rgb(image)

        image = util.img_as_ubyte(image)

        self._mask_layer.data = np.zeros(image.shape[:2], dtype=int)
        self._predictor.set_image(image)

    def _on_run(self, _: Optional[Any] = None) -> None:
        points = self._pts_layer.data
        if len(points) == 0 or self._im_layer_widget.value is None:
            return

        colors = self._pts_layer.face_color
        labels = np.all(colors == colors[0], axis=1)
        mask, _, self._logits = self._predictor.predict(
            point_coords=np.flip(points, axis=-1),
            point_labels=labels,
            mask_input=self._logits,
            multimask_output=False,
        )
        self._mask_layer.data = mask
