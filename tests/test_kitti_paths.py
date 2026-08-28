import pytest

from src.data.kitti import kitti_groundtruth_name


def test_selected_validation_rgb_maps_to_groundtruth() -> None:
    image_name = (
        "2011_09_26_drive_0002_sync_image_0000000069_image_02.png"
    )
    expected = (
        "2011_09_26_drive_0002_sync_groundtruth_depth_0000000069_image_02.png"
    )

    assert kitti_groundtruth_name(image_name) == expected


def test_unexpected_filename_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected KITTI"):
        kitti_groundtruth_name("0000000069.png")

