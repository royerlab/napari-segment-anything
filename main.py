import napari

def main():
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.window.add_plugin_dock_widget('napari-segment-anything')

if __name__ == "__main__":
    main()