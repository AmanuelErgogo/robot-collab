import unittest

import numpy as np

from metaworld_integration.config import (
    get_policy_class_name,
    resolve_task_spec,
    task_one_hot,
)
from metaworld_integration.parser import MetaWorldActionParser


class MetaWorldIntegrationTests(unittest.TestCase):
    def test_resolve_task_spec_expands_groups_and_preserves_order(self):
        tasks = resolve_task_spec("medium,reach-v3,medium")
        self.assertEqual(tasks[:3], ["basketball-v3", "bin-picking-v3", "box-close-v3"])
        self.assertIn("reach-v3", tasks)
        self.assertEqual(tasks.count("reach-v3"), 1)

    def test_task_one_hot_marks_exactly_one_entry(self):
        one_hot = task_one_hot("assembly-v3")
        self.assertEqual(len(one_hot), 50)
        self.assertEqual(sum(one_hot), 1)
        self.assertEqual(one_hot[0], 1)

    def test_policy_class_name_matches_upstream_convention(self):
        self.assertEqual(get_policy_class_name("pick-place-wall-v3"), "SawyerPickPlaceWallV3Policy")
        self.assertEqual(get_policy_class_name("reach-v3"), "SawyerReachV3Policy")

    def test_action_parser_accepts_valid_response(self):
        parser = MetaWorldActionParser()
        success, reason, action = parser.parse("EXECUTE\nACTION [0.1, -0.2, 0.3, -1.0]")
        self.assertTrue(success)
        self.assertEqual(reason, "")
        self.assertTrue(np.allclose(action, np.array([0.1, -0.2, 0.3, -1.0], dtype=np.float32)))

    def test_action_parser_rejects_bad_shape(self):
        parser = MetaWorldActionParser()
        success, reason, action = parser.parse("EXECUTE\nACTION [0.1, 0.2, 0.3]")
        self.assertFalse(success)
        self.assertIn("Expected exactly 4 action values", reason)
        self.assertIsNone(action)

    def test_action_parser_clips_out_of_range_values(self):
        parser = MetaWorldActionParser()
        success, reason, action = parser.parse("EXECUTE\nACTION [2.0, -3.0, 0.3, 0.0]")
        self.assertTrue(success)
        self.assertIn("clipped", reason)
        self.assertTrue(np.allclose(action, np.array([1.0, -1.0, 0.3, 0.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
