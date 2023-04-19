from typing import Any, Generator, Optional

import napari
import numpy as np
import torch
from magicgui.widgets import ComboBox, Container, PushButton, create_widget
from napari.layers import Image, Points, Shapes
from napari.layers.shapes._shapes_constants import Mode
from qtpy.QtCore import Qt
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
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
        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

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

        self._confirm_mask_btn = PushButton(
            text="Confirm Annot.",
            enabled=False,
            tooltip="Press C to confirm annotation.",
        )
        self._confirm_mask_btn.changed.connect(self._on_confirm_mask)
        self.append(self._confirm_mask_btn)

        self._cancel_annot_btn = PushButton(
            text="Cancel Annot.",
            enabled=False,
            tooltip="Press X to cancel annotation.",
        )
        self._cancel_annot_btn.changed.connect(self._cancel_annot)
        self.append(self._cancel_annot_btn)

        self._auto_segm_btn = PushButton(text="Auto. Segm.")
        self._auto_segm_btn.changed.connect(self._on_auto_run)
        self.append(self._auto_segm_btn)

        self._labels_layer = self._viewer.add_labels(
            data=np.zeros((256, 256), dtype=int),
            name="SAM labels",
        )

        self._mask_layer = self._viewer.add_labels(
            data=np.zeros((256, 256), dtype=int),
            name="SAM mask",
            color={1: "cyan"},
        )
        self._mask_layer.contour = 2

        self._pts_layer = self._viewer.add_points(name="SAM points")
        self._pts_layer.current_face_color = "blue"
        self._pts_layer.events.data.connect(self._on_interactive_run)
        self._pts_layer.mouse_drag_callbacks.append(
            self._mouse_button_modifier
        )
        self._boxes_layer = self._viewer.add_shapes(
            name="SAM box",
            face_color="transparent",
            edge_color="green",
            edge_width=2,
        )
        self._boxes_layer.mouse_drag_callbacks.append(self._on_shape_drag)

        self._image: Optional[np.ndarray] = None
        self._logits: Optional[torch.TensorType] = None

        self._model_type_widget.changed.emit(model_type)
        self._viewer.bind_key("C", self._on_confirm_mask)
        self._viewer.bind_key("X", self._cancel_annot)

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

        if im_layer.ndim != 2:
            raise ValueError(
                f"Only 2D images supported. Got {im_layer.ndim}-dim image."
            )

        image = im_layer.data
        if not im_layer.rgb:
            image = color.gray2rgb(image)

        elif image.shape[-1] == 4:
            # images with alpha
            image = color.rgba2rgb(image)

        if np.issubdtype(image.dtype, np.floating):
            image = image - image.min()
            image = image / image.max()

        self._image = util.img_as_ubyte(image)

        self._mask_layer.data = np.zeros(self._image.shape[:2], dtype=int)
        self._labels_layer.data = np.zeros(self._image.shape[:2], dtype=int)
        self._predictor.set_image(self._image)

    def _mouse_button_modifier(self, _: Points, event) -> None:
        self._pts_layer.selected_data = []
        if event.button == Qt.LeftButton:
            self._pts_layer.current_face_color = "blue"
        else:
            self._pts_layer.current_face_color = "red"

    def _on_interactive_run(self, _: Optional[Any] = None) -> None:
        points = self._pts_layer.data
        boxes = self._boxes_layer.data

        if self._im_layer_widget.value is None or (
            len(points) == 0 and len(boxes) == 0
        ):
            return

        if len(boxes) > 0:
            box = boxes[-1]
            box = np.stack([box.min(axis=0), box.max(axis=0)], axis=0)
            box = np.flip(box, -1).reshape(-1)[None, ...]
        else:
            box = None

        if len(points) > 0:
            points = np.flip(points, axis=-1)
            colors = self._pts_layer.face_color
            blue = [0, 0, 1, 1]
            labels = np.all(colors == blue, axis=1)
        else:
            points = None
            labels = None

        mask, _, self._logits = self._predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=self._logits,
            multimask_output=False,
        )
        self._mask_layer.data = mask[0]
        self._confirm_mask_btn.enabled = True
        self._cancel_annot_btn.enabled = True

    def _on_shape_drag(self, _: Shapes, event) -> Generator:
        if self._boxes_layer.mode != Mode.ADD_RECTANGLE:
            return
        # on mouse click
        yield
        # on move
        while event.type == "mouse_move":
            yield
        # on mouse release
        self._on_interactive_run()

    def _on_auto_run(self) -> None:
        if self._image is None:
            return
        mask_gen = SamAutomaticMaskGenerator(self._sam)
        preds = mask_gen.generate(self._image)

        labels = self._labels_layer.data

        for i, pred_dict in enumerate(preds):
            labels[pred_dict["segmentation"]] = i + 1

        self._labels_layer.data = labels

    def _on_confirm_mask(self, _: Optional[Any] = None) -> None:
        if self._image is None:
            return

        labels = self._labels_layer.data
        mask = self._mask_layer.data
        labels[np.nonzero(mask)] = labels.max() + 1
        self._labels_layer.data = labels
        self._cancel_annot()

    def _cancel_annot(self, _: Optional[Any] = None) -> None:
        # boxes must be reset first because of how of points data update signal
        self._boxes_layer.data = []
        self._pts_layer.data = []
        self._mask_layer.data = np.zeros_like(self._mask_layer.data)

        self._confirm_mask_btn.enabled = False
        self._cancel_annot_btn.enabled = False
