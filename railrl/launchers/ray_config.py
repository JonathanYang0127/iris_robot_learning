DIR_AND_MOUNT_POINT_MAPPINGS = [
    dict(
        local_dir='/home/steven/.mujoco',
        remote_dir='/home/ubuntu/.mujoco',
        mount_point='/root/.mujoco',
    ),
    dict(
        local_dir='/home/steven/res/railrl-private',
        remote_dir='/home/ubuntu/res/railrl-private',
        mount_point='/home/steven/res/railrl-private',
    ),
    dict(
        local_dir='/home/steven/res/multiworld',
        remote_dir='/home/ubuntu/res/multiworld',
        mount_point='/home/steven/res/multiworld',
    ),
    dict(
        local_dir='/tmp/local_exp.pkl',
        remote_dir='/home/ubuntu/local_exp.pkl',
        mount_point='/tmp/local_exp.pkl',
    ),
]
# This can basically be anything. Used for launching on instances. The
# local launch parameters (exo_func, exp_variant, etc) are saved at this location
# on the local machine and then transfered to the remote machine.
EXPERIMENT_INFO_PKL_FILEPATH = '/tmp/local_exp.pkl'
# Again, can be anything. The Ray autoscaler yaml file is saved to this location
# before launching.
LAUNCH_FILEPATH = '/tmp/autoscaler_launch.yaml'

LOCAL_LOG_DIR = '/home/steven/logs'
RUN_DOODAD_EXPERIMENT_SCRIPT_PATH = (
    '/home/steven/res/railrl-private/scripts/run_experiment_from_doodad.py'
)

AWS_CONFIG_NO_GPU=dict(
    REGION='us-west-2',
    INSTANCE_TYPE = 'g2.2xlarge',
    SPOT_PRICE = 0.05,
    REGION_TO_AWS_IMAGE_ID = {
        'us-west-2': 'ami-0d5bb58171e5325a8'
    },
    REGION_TO_AWS_AVAIL_ZONE = {
        'us-west-2': 'us-west-2a,us-west-2b'
    },

)

AWS_CONFIG_GPU = dict(
    REGION='us-west-2',
    INSTANCE_TYPE = 'g3.4xlarge',
    SPOT_PRICE = 0.6,
    REGION_TO_AWS_IMAGE_ID = {
        'us-west-2': 'ami-0d5bb58171e5325a8'
    },
    REGION_TO_AWS_AVAIL_ZONE = {
        'us-west-2': 'us-west-2a,us-west-2b'
    },
)

GCP_CONFIG_GPU = dict(
    REGION='us-west1',
    INSTANCE_TYPE='n1-highmem-8',
    SOURCE_IMAGE='https://www.googleapis.com/compute/v1/projects/railrl-private-gcp/global/images/railrl-private-ray',
    PROJECT_ID='railrl-private-gcp',
    gpu_kwargs=dict(
        num_gpu=1,
    ),
    REGION_TO_GCP_AVAIL_ZONE = {
        'us-west1': "us-west1-b",
    },

)


AWS_CONFIG = {
    True: AWS_CONFIG_GPU,
    False: AWS_CONFIG_NO_GPU,
}
GCP_CONFIG = {
    True: GCP_CONFIG_GPU,
    False: GCP_CONFIG_GPU,
}

DOODAD_DOCKER_IMAGE = 'vitchyr/railrl-torch4cuda9'
# If not set, default will be chosen by doodad
# AWS_S3_PATH = 's3://bucket/directory
GPU_DOODAD_DOCKER_IMAGE = 'stevenlin598/ray_railrl'
DOCKER_IMAGE = {
    True: GPU_DOODAD_DOCKER_IMAGE,
    False: GPU_DOODAD_DOCKER_IMAGE
}
LOG_BUCKET = 's3://steven.railrl/ray'
