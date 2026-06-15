import importlib
import unittest

import numpy as np


class OptionalVisualizationDependencyTests(unittest.TestCase):
    def test_env_utils_imports_with_or_without_open3d(self):
        env_utils = importlib.import_module("rocobench.envs.env_utils")
        self.assertTrue(hasattr(env_utils, "o3d"))

    def test_visualization_helpers_behave_correctly_for_current_environment(self):
        env_utils = importlib.import_module("rocobench.envs.env_utils")
        point_cloud = env_utils.PointCloud(
            xyz_pts=np.zeros((1, 3), dtype=np.float32),
            rgb_pts=np.zeros((1, 3), dtype=np.uint8),
            segmentation_pts={},
        )

        if env_utils.o3d is None:
            with self.assertRaises(ImportError) as exc_info:
                point_cloud.to_open3d()
            self.assertIn("pip install open3d", str(exc_info.exception))
        else:
            open3d_cloud = point_cloud.to_open3d()
            self.assertEqual(len(open3d_cloud.points), 1)


if __name__ == "__main__":
    unittest.main()
