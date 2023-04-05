from typing import Callable

import napari
import numpy as np
from skimage.data import astronaut

from napari_segment_anything import SAMWidget


def test_click(make_napari_viewer: Callable[[], napari.Viewer]) -> None:
    viewer = make_napari_viewer()
    # viewer = napari.Viewer()
    image = astronaut()

    widget = SAMWidget(viewer)

    viewer.window.add_dock_widget(widget)
    viewer.add_image(image)

    widget._weights_file.value = "weights/sam_vit_h_4b8939.pth"

    assert widget._predictor is not None
    assert widget._im_layer_widget.value is not None

    widget._pts_layer.data = [[42, 233]]  # point on hair

    assert np.any(widget._mask_layer.data > 0)

    widget._pts_layer.data = [[42, 233], [125, 225]]  # adding point to face

    assert np.any(widget._mask_layer.data > 0)

    # napari.run()
