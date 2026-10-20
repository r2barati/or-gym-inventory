"""Regression tests for NewsvendorEnv inventory constraints."""

import unittest

import numpy as np

from newsvendor import NewsvendorEnv


class NewsvendorInventoryCapTests(unittest.TestCase):
    def test_zero_lead_time_caps_immediate_sales(self):
        env = NewsvendorEnv(lead_time=0, max_inventory=3, max_order_quantity=10)
        env.reset(seed=42)
        # Deterministic seeded Poisson demand comfortably exceeds the cap.
        env.mu = 100.0
        env.state[4] = env.mu

        _, _, _, _, info = env.step(np.array([10.0], dtype=np.float32))

        self.assertGreater(info["demand"], 3)
        self.assertAlmostEqual(info["purchase_cost"], 3 * env.cost)
        self.assertAlmostEqual(info["revenue"], 3 * env.price)

    def test_zero_inventory_cap_prevents_immediate_sales(self):
        env = NewsvendorEnv(lead_time=0, max_inventory=0, max_order_quantity=10)
        env.reset(seed=42)
        env.mu = 100.0
        env.state[4] = env.mu

        _, _, _, _, info = env.step(np.array([10.0], dtype=np.float32))

        self.assertGreater(info["demand"], 0)
        self.assertEqual(info["revenue"], 0)
        self.assertEqual(info["purchase_cost"], 0)


    def test_positive_lead_time_delays_sales_until_arrival(self):
        env = NewsvendorEnv(lead_time=1, max_inventory=3, max_order_quantity=10)
        env.reset(seed=42)
        env.mu = 100.0
        env.state[4] = env.mu

        _, _, _, _, first = env.step(np.array([10.0], dtype=np.float32))
        self.assertGreater(first["demand"], 3)
        self.assertEqual(first["revenue"], 0)
        self.assertAlmostEqual(first["purchase_cost"], 3 * env.cost)

        _, _, _, _, second = env.step(np.array([0.0], dtype=np.float32))
        self.assertGreater(second["demand"], 3)
        self.assertAlmostEqual(second["revenue"], 3 * env.price)
        self.assertEqual(second["purchase_cost"], 0)


if __name__ == "__main__":
    unittest.main()
