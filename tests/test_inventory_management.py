"""Inventory conservation regressions for multi-echelon replenishment."""
import unittest

import numpy as np

from inventory_management import InvManagementBacklogEnv, InvManagementLostSalesEnv


class InventoryShipmentTests(unittest.TestCase):
    def make_env(self, env_type, initial_stock):
        n = len(initial_stock)
        env = env_type(env_config={
            "periods": 2,
            "I0": initial_stock,
            "r": [1.0] * (n + 1),
            "k": [0.0] * (n + 1),
            "h": [0.0] * n,
            "c": [50] * n,
            "L": [1] * n,
            "dist": 5,
            "user_D": [0, 0],
            "alpha": 1.0,
        })
        env.reset(seed=42)
        return env

    def test_supplier_stock_is_debited_for_downstream_order(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = self.make_env(env_type, [0, 10])
                _, _, _, _, info = env.step(np.array([4, 0], dtype=np.int64))
                np.testing.assert_array_equal(info["ending_inventory"], [0, 6])
                np.testing.assert_array_equal(env.R[0], [4, 0])

                # The four shipped units arrive at the retailer one period later.
                _, _, _, _, info = env.step(np.array([0, 0], dtype=np.int64))
                np.testing.assert_array_equal(info["ending_inventory"], [4, 6])

    def test_each_supplier_loses_only_its_own_shipments(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = self.make_env(env_type, [0, 10, 20])
                _, _, _, _, info = env.step(np.array([4, 5, 0], dtype=np.int64))
                np.testing.assert_array_equal(env.R[0], [4, 5, 0])
                np.testing.assert_array_equal(info["ending_inventory"], [0, 6, 15])
                self.assertEqual(sum(info["ending_inventory"]), 21)

                _, _, _, _, info = env.step(np.array([0, 0, 0], dtype=np.int64))
                np.testing.assert_array_equal(info["ending_inventory"], [4, 11, 15])
                self.assertEqual(sum(info["ending_inventory"]), 30)

    def test_supplier_cannot_ship_more_than_available_stock(self):
        for env_type in (InvManagementBacklogEnv, InvManagementLostSalesEnv):
            with self.subTest(env_type=env_type.__name__):
                env = self.make_env(env_type, [0, 2, 7])
                _, _, _, _, info = env.step(np.array([4, 5, 0], dtype=np.int64))
                np.testing.assert_array_equal(env.R[0], [2, 5, 0])
                np.testing.assert_array_equal(info["ending_inventory"], [0, 0, 2])
                self.assertTrue(np.all(info["ending_inventory"] >= 0))
                self.assertEqual(info["unfulfilled"][1], 2)


if __name__ == "__main__":
    unittest.main()
