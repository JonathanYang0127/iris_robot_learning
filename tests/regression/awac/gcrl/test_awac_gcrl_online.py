"""Test AWAC GCRL online. Requires the Sawyer environment from this multiworld
branch: https://github.com/vitchyr/multiworld/tree/ashvin-awac
"""
import os
import sys

from experiments.references.awac.gcrl.pusher1 import main

from rlkit.core import logger
from rlkit.testing import csv_util

def test_awac_gcrl_online():
    cmd = "python experiments/references/awac/gcrl/pusher1.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()
    reference_csv = "tests/regression/awac/gcrl/id0/progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "eval/Average Returns", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_awac_gcrl_online()
