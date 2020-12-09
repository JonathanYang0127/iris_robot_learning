import os
import importlib

from rlkit.envs.pearl_envs.ant_dir import AntDirEnv
from rlkit.envs.pearl_envs.half_cheetah_dir import HalfCheetahDirEnv
from rlkit.envs.pearl_envs.half_cheetah_vel import HalfCheetahVelEnv
from rlkit.envs.pearl_envs.humanoid_dir import HumanoidDirEnv
from rlkit.envs.pearl_envs.point_robot import PointEnv, SparsePointEnv

ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn

def _register_env(name, fn):
    """Registers a env by name for instantiation in rlkit."""
    if name in ENVS:
        raise ValueError("Cannot register duplicate env {}".format(name))
    if not callable(fn):
        raise TypeError("env {} must be callable".format(name))
    ENVS[name] = fn


def register_pearl_envs():
    _register_env('sparse-point-robot', SparsePointEnv)
    _register_env('ant-dir', AntDirEnv)
    _register_env('cheetah-dir', HalfCheetahDirEnv)
    _register_env('cheetah-vel', HalfCheetahVelEnv)
    _register_env('humanoid-dir', HumanoidDirEnv)
    _register_env('point-robot', PointEnv)
    # _register_env('walker-rand-params', Walker2DRandParamsEnv)

# automatically import any envs in the envs/ directory
# for file in os.listdir(os.path.dirname(__file__)):
#     if file.endswith('.py') and not file.startswith('_'):
#         module = file[:file.find('.py')]
#         importlib.import_module('rlkit.envs.pearl_envs.' + module)
