import json
from swarmae_api import SwarmaeApiClass
from swarmae.SwarmAEClient import SwarmAEClient
import argparse
# import time
# from PIL import Image
# import argparse
# import numpy as np
# import math
# import cv2, time
# import matplotlib.pyplot as plt
# from skimage import color
# from skimage.morphology import skeletonize
# from scipy.spatial import distance
import math
import numpy as np
import matplotlib.pyplot as plt
import time

import lib.utils as utils
import lib.swarm_api as api
import lib.frenet_optimal_planner as fop
import lib.controller as control
import lib.data_struct as struct
import lib.global_planner as gp
import lib.image_process as imp

def plot_helper(vehicle_pose, ref_path, vehicle_geometry,vehicle_pose_2, ref_path_2, vehicle_geometry_2, actual_x_list, actual_y_list, actual_x_list_2, actual_y_list_2):
    area = 15
    #area = 1500
    vehicle_shape = vehicle_geometry.get_vehicle_shape(vehicle_pose)
    vehicle_shape_2 = vehicle_geometry_2.get_vehicle_shape(vehicle_pose_2)

    plt.cla()
    plt.plot(ref_path.interp_x_, ref_path.interp_y_, color='black', label='reference path')
    plt.plot(vehicle_shape[:,0], vehicle_shape[:,1], color='orange', label='ego_vehicle')
    plt.plot(actual_x_list, actual_y_list, color='red', label='traveld path')

    plt.plot(ref_path_2.interp_x_, ref_path_2.interp_y_, color='black', label='reference path 2')
    plt.plot(vehicle_shape_2[:,0], vehicle_shape_2[:,1], color='pink', label='ego_vehicle 2')
    plt.plot(actual_x_list_2, actual_y_list_2, color='blue', label='traveld path 2')

    plt.axis('equal')
    plt.xlim([vehicle_pose.x_ - area, vehicle_pose.x_ + area])
    plt.ylim([vehicle_pose.y_ - area, vehicle_pose.y_ + area])
    plt.legend()
    plt.grid(True)
    plt.pause(0.01)

def apply_control(self, steer, throttle, brake, hand_brake):
    # 油门为正是前进，油门为负是倒车
    throttle = min(throttle,  1)
    throttle = max(throttle, -1)

    # 无论前进还是倒车，刹车都为正
    brake = min(brake, 1)
    brake = max(brake, 0)

    # 转向的取值范围是 [-1, 1]
    steer = min(steer,  1)
    steer = max(steer, -1)

    # 调用接口控制车辆
    # 挡位设为 0 即可，不用调整
    _, status_code = self.apply_vehicle_control(throttle, steer, brake, hand_brake, 0)

    return status_code    
def sample(ip, port):
    swarmeapi = SwarmaeApiClass()

    swarmeapi.game.stage_start('reconnaissance_start')
    swarmeapi.game.stage_complete('reconnaissance_end')
    swarmeapi.game.stage_start('vau_reconnaissance_start')
    swarmeapi.game.stage_complete('vau_reconnaissance_end')
    #获取区域信息
    area, b, c = swarmeapi.game.get_task_info()
    task_info = json.loads(area)
    subject3 = task_info.get('subject_5')
    start = subject3.get('start_line')
    end = subject3.get('end_line')
    task_areas, no_fly_arena_list = swarmeapi.get_no_fly_arena()
    print(no_fly_arena_list)

    img, scale, world_offset, frame, code = swarmeapi.game.get_road_network()

    img_skl = imp.skeleton_refinement(img)

    sw_node_1 = swarmeapi.sw_node[0]
    sw_node_2 = swarmeapi.sw_node[1]
    sw_node_3 = swarmeapi.sw_node[2]#无人机
    sw_node_4 = swarmeapi.sw_node[3]#无人机
    start_x_world, start_y_world, _, _ = sw_node_1.get_location()
    start_x = int(scale * (start_x_world - world_offset[0]))
    start_y = int(scale * (start_y_world - world_offset[1]))

    # 设定途径点和终点坐标
    waypoints = [
        (int(430 - world_offset[0]), int(-553 - world_offset[1])),
        (int(1100 - world_offset[0]), int(-361 - world_offset[1]))
    ]
    end_x = int(1250 - world_offset[0])
    end_y = int(-83 - world_offset[1])

    # 将所有目标点整合，包括终点
    all_points = [(start_x, start_y)] + waypoints + [(end_x, end_y)]

    # 定义路径数组
    path = []

    # 循环通过所有的途经点和终点，进行路径规划
    for i in range(len(all_points) - 1):
        # 获取当前段的起点和终点
        start = all_points[i]
        end = all_points[i + 1]

        # 获取当前段的骨架点坐标
        start_x_skl, start_y_skl, end_x_skl, end_y_skl = imp.skeleton_point(img_skl, start[0], start[1], end[0], end[1])

        # 初始化当前段的A*规划器
        a_star_planner = gp.AStar(start_x_skl, start_y_skl, end_x_skl, end_y_skl, img_skl)

        # 执行规划获取当前段路径
        segment_path = a_star_planner.update_planning()

        # 如果不是第一个段，去掉重复连接的起点路径
        if i > 0:
            segment_path = segment_path[1:]

        # 将当前段路径添加到总路径中
        path.extend(segment_path)

    # 绘制整条路径
    gp.plot_map_and_path(img_skl, path)

    # 将路径转换为真实世界坐标
    path_array = np.array(path)
    waypoint_x = path_array[:, 0] + world_offset[0]
    waypoint_y = path_array[:, 1] + world_offset[1]

    print("finish")
    #新的代码**************************************************************************
    waypoint_x = np.mat(waypoint_x).T
    waypoint_y = np.mat(waypoint_y).T
    print("x",waypoint_x)
    print("y",waypoint_y)

    # 利用全局路径生成局部规划的参考路径
    ref_path = fop.ReferencePath(waypoint_x, waypoint_y, resolution=2)

    #print("ref_path = ",ref_path)

    # 初始化控制器
    stanley = control.Stanley(k=1)
    pid = control.PID(Kp=0.1, Ki=0, Kd=0.0)
    wheel_base = 2

    # 创建任务目标 -> 行驶到最后一个参考点
    goal_x = ref_path.interp_x_[-1, 0]
    goal_y = ref_path.interp_y_[-1, 0]

    # 初始化车辆状态信息
    # 获取位置和姿态信息，替代 get_transform 函数
    x, y, _, _ = sw_node_1.get_location()  # 获取位置
    yaw, _, _, _, _ = sw_node_1.get_attitude()  # 获取姿态
    vehicle_pose = struct.Transform(x, y, yaw*math.pi/180)  # 将姿态角度转换为弧度
    vx, vy, vz, v, ax, ay, az, a, g, p, q, r, _,  = sw_node_1.get_velocity()
    vehicle_imu = struct.ImuData(vx, vy, vz, v, ax, ay, az, a, g, p, q, r)
    # vehicle_imu = sw_node_1.get_imu_data()    
    vehicle_state = struct.State(vehicle_pose, abs(vehicle_imu.v), vehicle_imu.a)
    vehicle_geometry = fop.VehicleGeometry(l=3, w=2)

    # 第二辆车的处理，直接使用获取位置和姿态信息的接口
    x2, y2, _, _ = sw_node_2.get_location()  # 获取位置
    yaw2, _, _, _, _ = sw_node_2.get_attitude()  # 获取姿态
    vehicle_pose_2 = struct.Transform(x2, y2, yaw2*math.pi/180)
    vx, vy, vz, v, ax, ay, az, a, g, p, q, r, _,  = sw_node_2.get_velocity()
    vehicle_imu_2 = struct.ImuData(vx, vy, vz, v, ax, ay, az, a, g, p, q, r)
    # vehicle_imu_2 = sw_node_2.get_imu_data()
    vehicle_state_2 = struct.State(vehicle_pose_2, abs(vehicle_imu_2.v), vehicle_imu_2.a)
    vehicle_geometry_2 = fop.VehicleGeometry(l=3, w=2)

    
    target_dx = 560
    target_dy = -630
    sw_node_3.control_kinetic_simply_global(target_dx, target_dy, 100, 100, frame_timestamp=int(round(time.time() * 1000)))
    sw_node_4.control_kinetic_simply_global(target_dx, target_dy, 90, 100, frame_timestamp=int(round(time.time() * 1000)))
    # 创建记录数据的空数组
    actual_x_list = []
    actual_y_list = []
    actual_x_list_2 = []
    actual_y_list_2 = []
    drone_x, drone_y, _, _ = sw_node_3.get_location()
    # 初始化采样时间
    prev_time = time.time()
    dt = 0.01
    flag1 = 0
    flag_2 = 0 
    # 样例中并不涉及局部规划，需要参考 Tutorial 4-5 自行编写代码
    # TODO

    while True:
        # #探测障碍物
        # ob = sw_node_1.detect_fun()
        # #打击模块
        # Health_flag = False
        # for i in ob.Health:
        #     if i > 0 :
        #         Health_flag = True
                
        # if len(ob.Type) != 0 and Health_flag :
        #     #如果识别到障碍物，开火！！！！！！
        #     for i in range(len(ob.Type)):
        #         sw_node_1.set_weapon(ob, i) 


        # 如果到达了终点，就将车辆刹停，并跳出循环
        if utils.distance(vehicle_pose.x_, vehicle_pose.y_, goal_x, goal_y) < 3 :
            # 先踩 10 秒钟刹车
            apply_control(sw_node_1,steer=0,throttle=0,brake=0.8, hand_brake=False)
            apply_control(sw_node_2,steer=0,throttle=0,brake=0.8, hand_brake=False)
            time.sleep(10)
            
            # 然后拉手刹制动
            apply_control(sw_node_1,steer=0, throttle=0, brake=0, hand_brake=True)
            apply_control(sw_node_2,steer=0, throttle=0, brake=0, hand_brake=True)
            
            # 随后跳出循环
            # break\
            t,code,res = sw_node_1.detect_situation()
            print(res)
            # #获取武器类
            # 弹药类型 0-子弹（95发）（伤害20）  1-40炮（15发）（伤害1600） 2-反坦克炮（2发）（伤害3000）
            slot,weapon,t,code = sw_node_1.get_weapon()
            print(weapon.get_weapon_ammo())
            time.sleep(5)
            weapon.set_weapon_status(2, 2, 1250, -30, 0.3)
            time.sleep(3)
            weapon.set_weapon_status(10, 1, 1250, -30, 0.6)
            flag1 = 1
        
        # 记录数据
        actual_x_list.append(vehicle_pose.x_)
        actual_y_list.append(vehicle_pose.y_)

        # sw_node_3.control_kinetic_simply_global(vehicle_pose.x_,vehicle_pose.y_, 100, 100, frame_timestamp=int(round(time.time() * 1000)))
        actual_x_list_2.append(vehicle_pose_2.x_)
        actual_y_list_2.append(vehicle_pose_2.y_)

        # 实时绘图
        ref_path_2 = ref_path #纯跟车可以公用路径
        plot_helper(vehicle_pose, ref_path, vehicle_geometry, vehicle_pose_2, ref_path_2, vehicle_geometry_2 ,actual_x_list, actual_y_list, actual_x_list_2, actual_y_list_2)

        # 寻找控制器参考点（一些工程经验，虽然 Stanley 的原理上是要找前轴投影点，但往往找一个更远的投影点可以在
        # 车辆高速行驶时获得更好的效果，此外，stanley 控制中的 delta_y 和 delta_theta 的参与比例都可以调整，为
        # 了调参方便，大家可以参考 Tutorial04 和 Tutorial05 将所有参数封装到 ini 文件中进行处理）
        front_x = vehicle_pose.x_ + 10 * math.cos(vehicle_pose.theta_)
        front_y = vehicle_pose.y_ + 10 * math.sin(vehicle_pose.theta_)
        front_x_2 = vehicle_pose_2.x_ + 10 * math.cos(vehicle_pose_2.theta_)
        front_y_2 = vehicle_pose_2.y_ + 10 * math.sin(vehicle_pose_2.theta_)

        dx = np.ravel(ref_path.interp_x_) - front_x
        dy = np.ravel(ref_path.interp_y_) - front_y
        dx_2 = np.ravel(ref_path.interp_x_) - front_x_2
        dy_2 = np.ravel(ref_path.interp_y_) - front_y_2
        dist = np.abs(dx) + np.abs(dy)
        dist_2 = np.abs(dx_2) + np.abs(dy_2)

        min_dist_idx = np.argmin(dist) #预瞄点
        min_dist_idx_2 = np.argmin(dist_2)
        
        target_x = ref_path.interp_x_[min_dist_idx, 0]
        target_y = ref_path.interp_y_[min_dist_idx, 0]
        target_x_2 = ref_path.interp_x_[min_dist_idx_2, 0]
        target_y_2 = ref_path.interp_y_[min_dist_idx_2, 0]
        target_theta = ref_path.interp_theta_[min_dist_idx, 0]
        target_theta_2 = ref_path.interp_theta_[min_dist_idx_2, 0]
        target_rou = ref_path.calculate_kappa()[min(min_dist_idx+25,852), 0]
        target_rou_2 = ref_path.calculate_kappa()[min(min_dist_idx_2+25,852), 0]
        target_pose = struct.Transform(target_x, target_y, target_theta)
        target_pose_2 = struct.Transform(target_x_2, target_y_2, target_theta_2)
        #第二辆车的路径跟踪 如果是跟车就不需要路径跟踪
        # target_x_2 = target_x 
        # target_y_2 = target_y
        # target_theta_2 = target_theta
        # target_pose_2 = target_pose

        # 这里仅仅是演示控制逻辑，因此没有考虑局部轨迹规划，目标车速为常数 10m/s
        target_v = 20
        s0 = 20 #希望保持的车距
        
        if target_v > 0.5 :
            target_v_2 = target_v + 0.2 * (math.sqrt((vehicle_pose.x_ - vehicle_pose_2.x_) * (vehicle_pose.x_ - vehicle_pose_2.x_) + (vehicle_pose.y_ - vehicle_pose_2.y_) * (vehicle_pose.y_ - vehicle_pose_2.y_)) - s0)
            if target_v_2 < 0 :
                target_v_2 = 1
        # if target_v < 0.5 :
        #     target_v_2 = 5 
        if target_v < 0 : #倒车
            target_v_2 = target_v
        

        # 计算控制量
        delta = stanley.update_control(target_pose, vehicle_state)
        accel = pid.update_control(target_v, vehicle_state.v_, dt)
        # if abs(delta) > 30:
        #     accel -= 0.9

        delta_2 = stanley.update_control(target_pose_2, vehicle_state_2)
        accel_2 = pid.update_control(target_v_2, vehicle_state_2.v_, dt)
        

        # 将转向归一化到 [-pi/6, pi/6]
        steer = delta / (math.pi / 6)
        steer_2 = delta_2 / (math.pi / 6)
        # if abs(steer) > 0.35:
        if target_rou < 25:
            accel = accel - 0.8
            accel_2 -= 0.3
        else:
            accel = accel + 0.1
        if target_rou_2 < 20:
            accel_2 -= 0.1 
        # 将加速度转换为油门、刹车
        if accel > 0:
            throttle = accel
            brake = 0
        else:
            throttle = 0
            brake = abs(accel) * 0.5

        if accel_2 > 0:
            throttle_2 = accel_2
            brake_2 = 0
        else:
            throttle_2 = 0
            brake_2 = abs(accel_2) * 0.5
        
        # 控制车辆运动
        if(flag1 != 1):
            apply_control(sw_node_1,steer, throttle, brake, hand_brake=False)
            apply_control(sw_node_2,steer_2, throttle_2, brake_2, hand_brake=False)

        # 更新采样时间
        dt = time.time() - prev_time
        prev_time = time.time()
        
        # 更新车辆状态信息
        x, y, _, _ = sw_node_1.get_location()  # 获取位置
        yaw, _, _, _, _ = sw_node_1.get_attitude()  # 获取姿态
        vehicle_pose = struct.Transform(x, y, yaw*math.pi/180)
        x2, y2, _, _ = sw_node_2.get_location()  # 获取位置
        yaw2, _, _, _, _ = sw_node_2.get_attitude()  # 获取姿态
        vehicle_pose_2 = struct.Transform(x2, y2, yaw2*math.pi/180)
        # vehicle_pose = vehicle.get_transform()
        vx, vy, vz, v, ax, ay, az, a, g, p, q, r, _,  = sw_node_1.get_velocity()
        vehicle_imu = struct.ImuData(vx, vy, vz, v, ax, ay, az, a, g, p, q, r)
        # vehicle_imu  = sw_node_1.get_imu_data()
        vehicle_state = struct.State(vehicle_pose, abs(vehicle_imu.v), vehicle_imu.a)

        # vehicle_pose_2 = vehicle_2.get_transform()
        vx, vy, vz, v, ax, ay, az, a, g, p, q, r, _,  = sw_node_2.get_velocity()
        vehicle_imu_2 = struct.ImuData(vx, vy, vz, v, ax, ay, az, a, g, p, q, r)
        # vehicle_imu_2  = sw_node_2.get_imu_data()
        # vehicle_imu_2  = vehicle_2.get_imu_data()
        vehicle_state_2 = struct.State(vehicle_pose_2, abs(vehicle_imu_2.v), vehicle_imu_2.a)
    
        # 打印 Debug 信息
        print("--- Debug ---------------------")
        print("throttle: ", throttle)
        print("brake: ", brake)
        print("steer:", steer)
        print("current speed", vehicle_state.v_)

        # 多车协同项目以及空地协同项目的同学需要在这个循环中继续计算无人机或者
        # 其他车的局部规划结果及控制量
        # TODO
        sw_node_3.control_kinetic_simply_global(target_dx, target_dy, 100, 100,
                                                frame_timestamp=int(round(time.time() * 1000)))
        sw_node_4.control_kinetic_simply_global(target_dx, target_dy, 90, 100,
                                                frame_timestamp=int(round(time.time() * 1000)))
        if utils.distance(drone_x, drone_y, 560, -630) < 3:
            target_dx = 560
            target_dy = -500
        if utils.distance(drone_x, drone_y, 560, -500) < 3:
            target_dx = 917
            target_dy = -140
        elif utils.distance(drone_x, drone_y, 917, -140) < 3:
            target_dx = 1250
            target_dy = -83
        elif utils.distance(drone_x, drone_y, 1250, -83) < 2:
            flag_2 = 1
        # drone_pose = drone.get_transform()
        # drone_x, drone_y = drone_pose.x_, drone_pose.y
        drone_x, drone_y, _, _ = sw_node_3.get_location()

        if flag1 == 1 and flag_2 == 1:
            break
    
    print("Done!")
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-i', '--address',
        default='127.0.0.1')
    argparser.add_argument(
        '-p', '--port',
        type=int,
        default=2000)
    argparser.add_argument(
        '-n', '--number',
        type=int,
        default=10)
    argparser.add_argument(
        '-s', '--subject',
        type=int,
        default=1
    )
    args = argparser.parse_args()
    ip = args.address.strip()
    port = args.port
    try:
        sample(ip, port)
    except KeyboardInterrupt:
        pass
    finally:
        print('done.')