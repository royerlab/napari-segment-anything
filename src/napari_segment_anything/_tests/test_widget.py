from typing import Callable

import napari
import numpy as np
import pytest
from numpy.typing import ArrayLike
from skimage.data import astronaut, camera

from napari_segment_anything import SAMWidget


@pytest.fixture
def sample_image() -> ArrayLike:
    return astronaut()


@pytest.mark.parametrize("im_dtype", [np.uint8, np.float32])
def test_click(
    make_napari_viewer: Callable[[], napari.Viewer],
    sample_image: ArrayLike,
    im_dtype: np.dtype,
) -> None:
    viewer = make_napari_viewer()
    # viewer = napari.Viewer()
    sample_image = sample_image.astype(im_dtype)

    widget = SAMWidget(viewer, model_type="vit_b")

    viewer.window.add_dock_widget(widget)
    viewer.add_image(sample_image)

    assert widget._predictor is not None
    assert widget._im_layer_widget.value is not None
    assert not widget._confirm_mask_btn.enabled
    assert not widget._cancel_annot_btn.enabled

    # first interaction
    widget._pts_layer.data = [[42, 233]]  # point on hair

    assert np.any(widget._mask_layer.data > 0)
    assert np.all(widget._labels_layer.data == 0)
    assert widget._confirm_mask_btn.enabled
    assert widget._cancel_annot_btn.enabled

    # confirm mask
    widget._confirm_mask_btn.clicked()
    assert np.all(widget._mask_layer.data == 0)
    assert np.any(widget._labels_layer.data > 0)
    assert not widget._confirm_mask_btn.enabled
    assert not widget._cancel_annot_btn.enabled

    # new interaction
    widget._pts_layer.data = [[42, 233], [125, 225]]  # adding point to face
    assert np.any(widget._mask_layer.data > 0)

    # confirm mask
    widget._confirm_mask_btn.clicked()
    assert np.all(widget._mask_layer.data == 0)
    assert len(np.unique(widget._labels_layer.data)) == 3

    # test cancel interaction
    widget._pts_layer.data = [[42, 233]]
    assert np.any(widget._mask_layer.data > 0)

    widget._cancel_annot_btn.clicked()
    assert np.any(widget._mask_layer.data == 0)
    assert len(np.unique(widget._labels_layer.data)) == 3  # still the same
    assert not widget._confirm_mask_btn.enabled
    assert not widget._cancel_annot_btn.enabled


@pytest.mark.parametrize("image", [astronaut(), camera()])
def test_automatic_segmentation(
    make_napari_viewer: Callable[[], napari.Viewer],
    image: ArrayLike,
) -> None:
    viewer = make_napari_viewer()
    widget = SAMWidget(viewer, model_type="vit_b")

    viewer.window.add_dock_widget(widget)
    viewer.add_image(image)

    widget._auto_segm_btn.clicked()
    assert np.any(widget._labels_layer.data > 0)
