"""
I recommend using sshfs to mount directories.
See this link for sshfs:
https://www.digitalocean.com/community/tutorials/how-to-use-sshfs-to-mount-remote-file-systems-over-ssh

For exmaple, I (Vitchyr) run these commands

sudo mkdir -p /mnt/gauss1/ashvin-all-data
sudo sshfs -o allow_other vitchyr@gauss1.banatao.berkeley.edu:/home/ashvin/data/ /mnt/gauss1/ashvin-all-data
"""
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 15})
plt.style.use("ggplot")

output_dir = '/home/vitchyr/git/railrl/data/papers/nips2018/script-output-2/'
ashvin_base_dir = '/mnt/gauss1/ashvin-all-data/'
vitchyr_base_dir = '/home/vitchyr/git/railrl/data/'

our_method_name = 'GRILL'


def format_func(value, tick_number):
    return(str(int(value // 1000)) + 'K')


assert output_dir[-1] == '/', 'Please add trailing slash'
assert ashvin_base_dir[-1] == '/', 'Please add trailing slash'
assert vitchyr_base_dir[-1] == '/', 'Please add trailing slash'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
