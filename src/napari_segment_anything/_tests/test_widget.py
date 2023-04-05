from typing import Callable

import napari
import numpy as np
import pytest
from skimage.data import astronaut

from napari_segment_anything import SAMWidget


@pytest.mark.parametrize("im_dtype", [np.uint8, np.float32])
def test_click(
    make_napari_viewer: Callable[[], napari.Viewer],
    im_dtype: np.dtype,
) -> None:
    viewer = make_napari_viewer()
    # viewer = napari.Viewer()
    image = astronaut().astype(im_dtype)

    widget = SAMWidget(viewer, model_type="vit_b")

    viewer.window.add_dock_widget(widget)
    viewer.add_image(image)

    assert widget._predictor is not None
    assert widget._im_layer_widget.value is not None

    widget._pts_layer.data = [[42, 233]]  # point on hair

    assert np.any(widget._mask_layer.data > 0)

    widget._pts_layer.data = [[42, 233], [125, 225]]  # adding point to face

    assert np.any(widget._mask_layer.data > 0)

    # napari.run()
