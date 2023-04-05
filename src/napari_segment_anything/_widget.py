from typing import Any, Optional

import napari
import numpy as np
import torch
from magicgui.widgets import ComboBox, Container, create_widget
from napari.layers import Image, Points
from qtpy.QtCore import Qt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
from skimage import color, util

from napari_segment_anything.utils import get_weights_path

# from segment_anything.utils.transforms import ResizeLongestSide


class SAMWidget(Container):
    _sam: Sam
    _predictor: SamPredictor

    def __init__(self, viewer: napari.Viewer, model_type: str = "default"):
        super().__init__()
        self._viewer = viewer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model_type_widget = ComboBox(
            value=model_type,
            choices=list(sam_model_registry.keys()),
            label="Model:",
        )
        self._model_type_widget.changed.connect(self._load_model)
        self.append(self._model_type_widget)

        self._im_layer_widget = create_widget(annotation=Image, label="Image:")
        self._im_layer_widget.changed.connect(self._load_image)
        self.append(self._im_layer_widget)

        self._mask_layer = self._viewer.add_labels(
            data=np.zeros((256, 256), dtype=int)
        )

        self._pts_layer = self._viewer.add_points(name="SAM points")
        self._pts_layer.current_face_color = "blue"
        self._pts_layer.events.data.connect(self._on_run)
        self._pts_layer.mouse_drag_callbacks.append(
            self._mouse_button_modifier
        )
        # self._boxes_layer = self._viewer.add_shapes(name="SAM boxes")

        self._logits: Optional[torch.TensorType] = None

        self._model_type_widget.changed.emit(model_type)

    def _load_model(self, model_type: str) -> None:
        self._sam = sam_model_registry[model_type](
            get_weights_path(model_type)
        )
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

        if np.issubdtype(image.dtype, np.floating):
            image = image - image.min()
            image = image / image.max()

        image = util.img_as_ubyte(image)

        self._mask_layer.data = np.zeros(image.shape[:2], dtype=int)
        self._predictor.set_image(image)

    def _mouse_button_modifier(self, _: Points, event) -> None:
        self._pts_layer.selected_data = []
        if event.button == Qt.LeftButton:
            self._pts_layer.current_face_color = "blue"
        else:
            self._pts_layer.current_face_color = "red"

    def _on_run(self, _: Optional[Any] = None) -> None:
        points = self._pts_layer.data
        if len(points) == 0 or self._im_layer_widget.value is None:
            return

        colors = self._pts_layer.face_color
        blue = [0, 0, 1, 1]
        labels = np.all(colors == blue, axis=1)

        mask, _, self._logits = self._predictor.predict(
            point_coords=np.flip(points, axis=-1),
            point_labels=labels,
            mask_input=self._logits,
            multimask_output=False,
        )
        self._mask_layer.data = mask
