from urdf_parser_py.urdf import URDF 
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
# import intera_interface as ii
import baxter_interface as bi
import rospy
import numpy as np 
robot = URDF.from_parameter_server(key='robot_description')
from pykdl_utils.kdl_kinematics import KDLKinematics
base_link = 'base'
# end_link = 'right_hand'
end_link = 'right_gripper'
kdl_kin = KDLKinematics(robot, base_link, end_link)
rospy.init_node('saw')
# arm = ii.Limb('right')
arm = bi.Limb('right')
q = arm.joint_angles()
q = [q['right_s0'], q['right_s1'], q['right_e0'], q['right_e1'], q['right_w0'], q['right_w1'], q['right_w2']]
# q = [q['right_j0'], q['right_j1'], q['right_j2'], q['right_j3'], q['right_j4'], q['right_j5'], q['right_j6']]
pose = kdl_kin.forward(q, end_link='right_gripper')
pose = np.squeeze(np.asarray(pose))
pose = [pose[0][3], pose[1][3], pose[2][3]]
print pose

j = kdl_kin.jacobian(q)
import ipdb; ipdb.set_trace()