import unittest

from railrl.misc.ml_util import StatFollowingIntSchedule


class TestLossFollowingIntSchedule(unittest.TestCase):

    def test_value_changes_average_1(self):
        schedule = StatFollowingIntSchedule(
            0,
            (-1, 1),
            1,
        )
        values = []
        for stat in [0, 0, 2, 2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, 1, 2, 1]
        self.assertEqual(values, expected)

    def test_value_changes_average_3(self):
        schedule = StatFollowingIntSchedule(
            0,
            (-1, 1),
            3,
        )
        values = []
        for stat in [0, 0, 2, 2, -2, -2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, 0, 1, 1, 1, 0]
        self.assertEqual(values, expected)

    def test_value_changes_average_1_inverse(self):
        schedule = StatFollowingIntSchedule(
            0,
            (-1, 1),
            1,
            invert=True,
        )
        values = []
        for stat in [0, 0, 2, 2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, -1, -2, -1]
        self.assertEqual(values, expected)

    def test_value_changes_average_3_inverse(self):
        schedule = StatFollowingIntSchedule(
            0,
            (-1, 1),
            3,
            invert=True,
        )
        values = []
        for stat in [0, 0, 2, 2, -2, -2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, 0, -1, -1, -1, 0]
        self.assertEqual(values, expected)

    def test_value_clipped(self):
        schedule = StatFollowingIntSchedule(
            0,
            (-1, 1),
            1,
            value_bounds=(-2, 2),
        )
        values = []
        for stat in [2, 2, 2, 2, -2, -2, -2, -2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [1, 2, 2, 2, 1, 0, -1, -2, -2]
        self.assertEqual(values, expected)


if __name__ == '__main__':
    unittest.main()