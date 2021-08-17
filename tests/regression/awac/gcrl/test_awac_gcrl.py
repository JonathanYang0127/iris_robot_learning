import os
import sys

from experiments.references.awac.gcrl.pusher1 import main

from rlkit.core import logger
from rlkit.testing import csv_check

def test_awac_gcrl_online():
    cmd = "python experiments/references/awac/gcrl/pusher1.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()
    reference_csv = "tests/regression/awac/gcrl/id0/progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    output = csv_check.get_exp(output_csv)
    reference = csv_check.get_exp(reference_csv)
    csv_check.check_equal(reference, output, ["epoch", "eval/Average Returns", "trainer/Advantage Score Max"], )

if __name__ == "__main__":
    test_awac_gcrl_online()
