# napari-segment-anything

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/napari-segment-anything.svg?color=green)](https://github.com/jookuma/napari-segment-anything/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-segment-anything.svg?color=green)](https://pypi.org/project/napari-segment-anything)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-segment-anything.svg?color=green)](https://python.org)
[![tests](https://github.com/jookuma/napari-segment-anything/workflows/tests/badge.svg)](https://github.com/jookuma/napari-segment-anything/actions)
[![codecov](https://codecov.io/gh/jookuma/napari-segment-anything/branch/main/graph/badge.svg)](https://codecov.io/gh/jookuma/napari-segment-anything)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-segment-anything)](https://napari-hub.org/plugins/napari-segment-anything)

Napari plugin of [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)

Download the network weights [here](https://github.com/facebookresearch/segment-anything#model-checkpoints)


https://user-images.githubusercontent.com/21022743/230456433-2fa7bc40-a735-4d73-8d87-ecf776bbe2be.mp4


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

You can install `napari-segment-anything` via [pip]:

```bash
pip install napari-segment-anything
```

We recommend installing the latest development version because this package is being developed:

```bash
pip install git+https://github.com/jookuma/napari-segment-anything.git
```

**IMPORTANT**: `napari` won't work if you don't have `pyqt5` or `pyside2` installed.

## Instructions

### Opening napari-segment-anything

This software is napari plugin, so once you have napari installed you can open it using the command line:

```bash
napari <your image path> -w napari-segment-anything 'Segment Anything'
```

I noticed that sometimes napari fails to load the plugin widget from the command line, go to step 2 from below to load it.

If you prefer the user interface you need to:

1) Drag and drop your image into napari to load it;
2) Go to the "Plugins" file menu in the upper right corner and select the "Segment Anything" plugin.
3) Follow the instructions below for usage.

**IMPORTANT**: If you get an error make sure you have `pyqt5` or `pyside2` installed.

### Usage

- Interactions are done on the "SAM points" and "SAM box" layers using the existing functionalities of napari. Only rectangle shapes trigger the network prediction.
- For points supervision, left clicks are positive cues (object) and right clicks are negative (background).
- Press the "Confirm Annot." button (or the "C" key) to propagate the current segmentation mask to the label image.
- Use the napari labels layer features to delete or edit already confirmed labels.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"napari-segment-anything" is a free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/jookuma/napari-segment-anything/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
