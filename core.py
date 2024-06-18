import argparse
import time
from swarmae.SwarmAEActor import SwarmAEActor
from swarmae.SwarmAEGame import SwarmAEGame
from swarmae.SwarmAEClient import SwarmAEClient
from swarmae.SwarmAENode import SwarmAENode
import math
from scipy.optimize import root  # 解方程
import json
import cv2
import threading
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import task_vis_map
'''全局变量'''


# 侦察可打击目标信息列表   [{"Name":  ,"Position": { "x": , "y":  , "z": }, "Type_Name":  , "HealthPoint": }, {},...,{}]
strike_info = []


'''工具函数'''


def path_vis(im, list):  # 输入参数是彩色图像和字典
    if list:
        im_vis = im.copy()
        for dict in list:
            Y = dict['y']
            X = dict['x']
            im_vis.putpixel((X, Y), (255, 0, 0, 255))
        im_vis.show()

def vis_fig(road_source, save_flag, show_flag, name):
    print(type(road_source))
    plt.title(name)
    if show_flag :
        road_source.show()
    if save_flag:
        # 构建正确的文件路径，使用os.path.join确保跨平台兼容性
        save_path = os.path.join("visualization", f"{name}.png")
        road_source.save(save_path, format='PNG')


def vis_task_info(task_info, save_flag, show_flag):
    # 创建一个新的figure和axes
    # print(task_info['no_fly_arena'])
    # print(len(task_info['no_fly_arena']))
    fig, ax = plt.subplots()
    no_fly_rectangle1 = [
        (task_info['no_fly_arena'][0][0]['x'], task_info['no_fly_arena'][0][0]['y']),
        (task_info['no_fly_arena'][0][1]['x'], task_info['no_fly_arena'][0][1]['y']),
        (task_info['no_fly_arena'][0][2]['x'], task_info['no_fly_arena'][0][2]['y']),
        (task_info['no_fly_arena'][0][3]['x'], task_info['no_fly_arena'][0][3]['y'])
    ]
    no_fly_rectangle2 = [
        (task_info['no_fly_arena'][1][0]['x'], task_info['no_fly_arena'][1][0]['y']),
        (task_info['no_fly_arena'][1][1]['x'], task_info['no_fly_arena'][1][1]['y']),
        (task_info['no_fly_arena'][1][2]['x'], task_info['no_fly_arena'][1][2]['y']),
        (task_info['no_fly_arena'][1][3]['x'], task_info['no_fly_arena'][1][3]['y']),
        (task_info['no_fly_arena'][1][4]['x'], task_info['no_fly_arena'][1][4]['y']),
        (task_info['no_fly_arena'][1][5]['x'], task_info['no_fly_arena'][1][5]['y']),
        (task_info['no_fly_arena'][1][6]['x'], task_info['no_fly_arena'][1][6]['y']),
        (task_info['no_fly_arena'][1][7]['x'], task_info['no_fly_arena'][1][7]['y'])
    ]
    no_fly_rectangle3 = [
        (task_info['no_fly_arena'][2][0]['x'], task_info['no_fly_arena'][2][0]['y']),
        (task_info['no_fly_arena'][2][1]['x'], task_info['no_fly_arena'][2][1]['y']),
        (task_info['no_fly_arena'][2][2]['x'], task_info['no_fly_arena'][2][2]['y']),
        (task_info['no_fly_arena'][2][3]['x'], task_info['no_fly_arena'][2][3]['y']),
        (task_info['no_fly_arena'][2][4]['x'], task_info['no_fly_arena'][2][4]['y']),
        (task_info['no_fly_arena'][2][5]['x'], task_info['no_fly_arena'][2][5]['y']),
        (task_info['no_fly_arena'][2][6]['x'], task_info['no_fly_arena'][2][6]['y']),
        (task_info['no_fly_arena'][2][7]['x'], task_info['no_fly_arena'][2][7]['y'])
    ]
    arena_rectangle = [
        (task_info['arena'][0]['x'], task_info['arena'][0]['y']), (task_info['arena'][1]['x'], task_info['arena'][1]['y']),
        (task_info['arena'][2]['x'], task_info['arena'][2]['y']), (task_info['arena'][3]['x'], task_info['arena'][3]['y'])
    ]
    # print(arena_rectangle)
    emc_rectangle = [
        (task_info['emc_arena'][0]['x'], task_info['emc_arena'][0]['y']),
        (task_info['emc_arena'][1]['x'], task_info['emc_arena'][1]['y']),
        (task_info['emc_arena'][2]['x'], task_info['emc_arena'][2]['y']),
        (task_info['emc_arena'][3]['x'], task_info['emc_arena'][3]['y'])
    ]
    # print(emc_rectangle)
    mine_rectangle = [
        (task_info['mine_arena'][0]['x'], task_info['mine_arena'][0]['y']),
        (task_info['mine_arena'][1]['x'], task_info['mine_arena'][1]['y']),
        (task_info['mine_arena'][2]['x'], task_info['mine_arena'][2]['y']),
        (task_info['mine_arena'][3]['x'], task_info['mine_arena'][3]['y'])
    ]
    # print(mine_rectangle)

    # 绘制每个矩形，并设置不同的颜色和透明度
    ax.add_patch(patches.Polygon(arena_rectangle, closed=True, fill=True, linewidth=1, edgecolor='black',
                                  facecolor='black', alpha=0.5))
    ax.add_patch(patches.Polygon(emc_rectangle, closed=True, fill=True, linewidth=1, edgecolor='black',
                                  facecolor='b', alpha=0.5))
    ax.add_patch(patches.Polygon(mine_rectangle, closed=True, fill=True, linewidth=1, edgecolor='black',
                                  facecolor='r', alpha=0.5))
    ax.add_patch(patches.Polygon(no_fly_rectangle1, closed=True, fill=True, linewidth=1, edgecolor='black',
                                  facecolor='purple', alpha=0.5))
    ax.add_patch(patches.Polygon(no_fly_rectangle2, closed=True, fill=True, linewidth=1, edgecolor='black',
                                  facecolor='purple', alpha=0.5))
    ax.add_patch(patches.Polygon(no_fly_rectangle3, closed=True, fill=True, linewidth=1, edgecolor='black',
                                  facecolor='purple', alpha=0.5))
    # print("emc-蓝色，mine-红色，nofly-紫色")

    line_vertices_x = []
    line_vertices_y = []
    # 绘制线段
    for i in range(0, 6):
        subject_name = 'subject_{}'.format(i+1)
        line_vertices_x.append([task_info[subject_name]['start_line'][0]['x'], task_info[subject_name]['start_line'][1]['x']])
        line_vertices_y.append([task_info[subject_name]['start_line'][0]['y'], task_info[subject_name]['start_line'][1]['y']])
        line_vertices_x.append([task_info[subject_name]['end_line'][0]['x'], task_info[subject_name]['end_line'][1]['x']])
        line_vertices_y.append([task_info[subject_name]['end_line'][0]['y'], task_info[subject_name]['end_line'][1]['y']])

    for i in range(12):
        if (i % 2) == 0:
            subject_name = 'subject_{}'.format(i/2 + 1)
            plt.plot(line_vertices_x[i], line_vertices_y[i])
            plt.text(line_vertices_x[i][0] + random.uniform(10,20), line_vertices_y[i][0] + random.uniform(10,20), subject_name+'_S')
        else:
            subject_name = 'subject_{}'.format(i/2 + 1)
            plt.plot(line_vertices_x[i], line_vertices_y[i])
            plt.text(line_vertices_x[i][0] + random.uniform(10,20), line_vertices_y[i][0] + random.uniform(10,20), subject_name+'_E')

    # # 设置坐标轴的范围以确保所有矩形可见
    # ax.set_xlim(-2000, 3000)  # 注意调整x轴的范围以包含最后一个矩形
    # ax.set_ylim(-1500, 500)  # 同样调整y轴的范围

    # 显示图形
    plt.title('all_arena')
    if save_flag :
        save_path = os.path.join("visualization", "all_arena.png")
        plt.savefig(save_path, format='PNG')
    if show_flag :
        plt.show()


def p2p_dist(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


'''控制算法'''


def calculate_alpha(current_pose, lookahead_point):
    dx = lookahead_point['x'] - current_pose[0]
    dy = lookahead_point['y'] - current_pose[1]
    alpha = math.atan2(dy, dx) - current_pose[2]
    return alpha


def calculate_cmd(alpha):
    global last_steer
    steering_angle = STEER_CONTROL_KP * alpha
    # if last_steer != math.inf and math.fabs(steering_angle - last_steer) > STEER_MAX_ACC:
    #     if alpha > 0.0:
    #         steering_angle = last_steer + STEER_MAX_ACC
    #     else:
    #         steering_angle = last_steer - STEER_MAX_ACC
    throttle = 1.0
    print('alpha:', alpha, 'steer:', steering_angle, 'throttle:', throttle)
    return throttle, steering_angle


def pure_pursuit_control(path, node, current_pose):
    closest_point_index = find_closest_point_index(path, current_pose)
    lookahead_point_index = find_lookahead_point_index(path, closest_point_index, current_pose)
    lookahead_point = path[lookahead_point_index]
    alpha = calculate_alpha(current_pose, lookahead_point)
    throttle, steering_angle = calculate_cmd(alpha)
    global last_steer
    last_steer = steering_angle
    node.apply_vehicle_control(throttle, steering_angle, 0.0, False, 0)


def find_closest_point_index(path, current_pose):
    distances = [math.sqrt((point['x'] - current_pose[0]) ** 2 + (point['y'] - current_pose[1]) ** 2) for point in
                 path]
    return distances.index(min(distances))


def find_lookahead_point_index(path, closest_point_index, current_pose):
    for i in range(closest_point_index + 1, len(path)):
        if math.sqrt(
                (path[i]['x'] - current_pose[0]) ** 2 + (path[i]['y'] - current_pose[1]) ** 2) > LOOKAHEAD_DISTANCE:
            return i
    return len(path) - 1


def pp_path_track(path, node):
    end_pt = [path[-1]['x'], path[-1]['y']]
    current_pose = [node.get_location()[0], node.get_location()[1], math.radians(node.get_attitude()[0])]
    while p2p_dist(end_pt, current_pose) > PATH_TRACK_THRESHOLD:
        if should_replan:
            break
        tik = time.time()
        pure_pursuit_control(path, node, current_pose)
        tok = time.time()
        pp_dt = tok - tik
        print("pp_dt:", pp_dt)
        current_pose = [node.get_location()[0], node.get_location()[1], node.get_attitude()[0]]
        time.sleep(1.0)
    brakedown(node)


'''ue类'''


class SwarmaeClass:

    def __init__(self):
        self.born_pos_word = []  # 初始化 born_pos_word 属性为空列表
        self.ae_client = None  # 初始化 ae_client 属性为 None 

    # 创建Client类    10.62.148.65      127.0.0.1
    def create_client(self, ue_ip="127.0.0.1", ue_port=2000):
        print("【初始化】创建Client: ", ue_ip, ue_port)
        self.ae_client = SwarmAEClient(ue_ip=ue_ip, ue_port=ue_port)
        return self.ae_client

    # 创建Node类
    def create_node(self, ae_client, type, id):
        frame_timestamp, sw_node, sw_code = ae_client.register_node(
            node_type=type, node_name="节点" + str(id), node_no=id, frame_timestamp=int(round(time.time() * 1000))
        )
        if sw_code == 200:
            node_name, node_no, team, node_type, _, _ = sw_node.get_node_info()
            self.born_pos_word.append(sw_node.get_location()[:3])
            print("【初始化】register_node [",id,"]: ", node_name, node_type)
        else:
            print("【初始化】register_node [", id, "]:", frame_timestamp, sw_code)
        return sw_code, sw_node


''' ——————————第1阶段无人机路径规划需要用到的函数—————————— '''


def reduce_arena(_arena):  # 输入:元素是字典的list 输出:元素是字典的list 缩小区域的搜索面积
    new_per = 0.9
    reduce = 1 - new_per
    a = abs(_arena[0]['x'] - _arena[1]['x'])
    b = abs(_arena[0]['y'] - _arena[3]['y'])
    d = reduce * a * b / 2 / (a + b)
    d = round(d, 1)  # 取整
    return [{"x": _arena[0]['x'] + d, "y": _arena[0]['y'] + d}, {"x": _arena[1]['x'] - d, "y": _arena[1]['y'] + d},
            {"x": _arena[2]['x'] - d, "y": _arena[2]['y'] - d}, {"x": _arena[3]['x'] + d, "y": _arena[3]['y'] - d}]


def arena_cut(task_arena):  # 将整个矩形工作区域由下至上均分成3部分，注意只有当json中的顶点顺序为左下、右下、右上、左上时才成立
    height = round((task_arena[3]["y"] - task_arena[0]["y"]) / 3, 1)
    sub_arena_0 = [  # 最下面
        {"x": task_arena[0]["x"], "y": task_arena[0]["y"]},
        {"x": task_arena[1]["x"], "y": task_arena[1]["y"]},
        {"x": task_arena[1]["x"], "y": task_arena[1]["y"] + height},
        {"x": task_arena[0]["x"], "y": task_arena[1]["y"] + height}
    ]
    sub_arena_1 = [  # 中间
        {"x": task_arena[0]["x"], "y": task_arena[0]["y"] + height},
        {"x": task_arena[1]["x"], "y": task_arena[1]["y"] + height},
        {"x": task_arena[1]["x"], "y": task_arena[1]["y"] + 2 * height},
        {"x": task_arena[0]["x"], "y": task_arena[1]["y"] + 2 * height}
    ]
    sub_arena_2 = [  # 最上面
        {"x": task_arena[0]["x"], "y": task_arena[0]["y"] + 2 * height},
        {"x": task_arena[1]["x"], "y": task_arena[1]["y"] + 2 * height},
        {"x": task_arena[2]["x"], "y": task_arena[2]["y"]},
        {"x": task_arena[3]["x"], "y": task_arena[3]["y"]}
    ]
    cut_arena = [sub_arena_0, sub_arena_1, sub_arena_2]
    return cut_arena  #返回了一个二维列表


def check_coordinate_in_arena(x, y, arenas):  # 判断航迹点是否在禁飞区和电磁干扰区之内，在的话返回区域id，不在的话返回-1
    bound = 10
    for arena_id, arena in enumerate(arenas):
        x_cords = [point["x"] for point in arena]
        y_cords = [point["y"] for point in arena]
        x_min, x_max = min(x_cords), max(x_cords)
        y_min, y_max = min(y_cords), max(y_cords)

        if x_min - bound <= x <= x_max + bound and y_min - bound <= y <= y_max + bound:
            return arena_id
    return -1


def no_fly_bound(_id, _no_fly_arena):
    arena = _no_fly_arena[_id]
    x_cords = [point["x"] for point in arena]
    y_cords = [point["y"] for point in arena]
    x_min, x_max = min(x_cords), max(x_cords)
    y_min, y_max = min(y_cords), max(y_cords)
    return [x_min, x_max, y_min, y_max]


def generate_flight_path(start_points, task_arena, no_fly_arena):  # 航迹点生成函数
    reduced_task_arena = reduce_arena(task_arena)
    cut_task_arena = arena_cut(reduced_task_arena)  # 将任务区域三等分
    # cut_task_arena = arena_cut(task_arena)  # 将任务区域三等分
    print(cut_task_arena)
    flight_paths = []
    # 步长需要保持一致
    para_step = 200  # 水平移动步长 原来200
    vert_step = 20  # 垂直移动步长
    h = 200  # 固定飞机的高度
    r = 240  # 扫描半径 原来100
    # 遍历三块区域进行路径规划
    for idx, arena in enumerate(cut_task_arena):
        route = []
        # 先添加起点
        current_x = arena[0]["x"] + r
        current_y = arena[0]["y"] + r
        route.append({"x": round(current_x, 1), "y": round(current_y, 1), "z": h})
        direction = "up"

        while True:
            # 向上垂直运动，y每次加vert_step，直到运动到区域边界100米内
            if direction == "up":
                while current_y < (arena[3]["y"] - r):
                    current_y_temp = current_y + vert_step
                    t = check_coordinate_in_arena(current_x, current_y_temp, no_fly_arena)
                    if t == -1:  # 新移动的点不在禁飞区内
                        current_y = current_y_temp
                        route.append({"x": round(current_x, 1), "y": round(current_y, 1), "z": h})
                    else:  # 新移动的点在禁飞区之内
                        break
            # 向下垂直运动，y每次减para_step，直到运动到区域边界
            else:
                while current_y > (arena[0]["y"] + r):
                    current_y_temp = current_y - vert_step
                    t = check_coordinate_in_arena(current_x, current_y_temp, no_fly_arena)
                    if t == -1:  # 新移动的点不在禁飞区内
                        current_y = current_y_temp
                        route.append({"x": round(current_x, 1), "y": round(current_y, 1), "z": h})
                    else:  # 新移动的点在禁飞区之内
                        break

            if current_x + para_step >= arena[1]["x"] - r:
                break
            # 水平运动到边界后向上平移down_step，当超过工作区域的上界时退出当前区域的循环
            current_x_temp = current_x + para_step
            t = check_coordinate_in_arena(current_x_temp, current_y, no_fly_arena)
            if t == -1:  # 向右平行移动时没有进入到禁飞区内
                current_x = current_x_temp
            else:  # 向右平行移动时进入到禁飞区内
                square_no_fly = no_fly_bound(t, no_fly_arena)
                if direction == "up":  # 下面空间大，在左下角避障
                    current_x = square_no_fly[0] - 30
                    current_y = square_no_fly[2] - 30
                else:  # 上面空间大，在左上角避障
                    current_x = square_no_fly[0] - 30
                    current_y = square_no_fly[3] + 30
            route.append({"x": round(current_x, 1), "y": round(current_y, 1), "z": h})
            direction = "up" if direction == "down" else "down"

        flight_paths.append({"node_id": idx + 1, "routes": route})

    return flight_paths


def drone_plan():  # 无人机航迹规划函数
    info = list(game.get_task_info())  # 将元组转换成列表
    whole_arena_data = json.loads(info[0])  # 把区域信息提取出来，并解析为有效的字典
    task_arena = whole_arena_data["arena"]  # 缩小到原来的85%？
    no_fly_arena = whole_arena_data["no_fly_arena"]
    emc_arena = whole_arena_data["emc_arena"]
    no_fly_arena.append(emc_arena)
    print(no_fly_arena)
    start_points = [
        (sw_node8.get_location()[:3]),
        (sw_node9.get_location()[:3]),
        (sw_node10.get_location()[:3])
    ]
    flight_paths = generate_flight_path(start_points, task_arena, no_fly_arena)
    json_output = json.dumps(flight_paths)
    return json_output


'''——————————————————————————————————————————————————'''

''' ——————————第2阶段无人机空中侦察需要用到的函数—————————— '''


def is_arrived(_drone_x, _drone_y, _dest_x, _dest_y):  # 判断是否进入到指定点的范围之内，进入到±5之内就认为已到达，跳出循环发送下一个航迹点
    range = 20
    if (_dest_x - range) < _drone_x < (_dest_x + range) and (_dest_y - range) < _drone_y < (_dest_y + range):
        # if _drone_x == _dest_x and _drone_y == _dest_y:
        return True
    else:
        return False


def strike_info_is_repeated(target_dict):  # 判断之前存过的目标信息是否和当前侦察到的重合
    if not strike_info:  # 当目标信息为空的时候，返回false表示不重复
        return False
    else:
        for target in strike_info:  # detect_res可能会包含之前已经探测到并存放到strike_info中的元素，因此要判断strike_info是否已经有了当前侦测到的目标
            if target_dict["Name"] == target["Name"]:
                return True
        return False


# def drone_stopped(_drone_node):  # 如果无人机速度为0那就执行下一个
#
#     drone_veh, _, _, _, _ = _drone_node.get_velocity()
#     if drone_veh == 0:
#         return True
#     else:
#         return False

def follow_route_points(_drone, _vau_route_points, _drone_start_point):  # 1. 控制飞机飞向目标点  2. 途中不断查询是否有侦察到的目标
    # 控制飞机朝目标点飞去
    print(_drone_start_point)
    while _drone.get_location()[2] < 150:
        _drone.control_kinetic_simply_global(_drone_start_point[0], _drone_start_point[1], 200, 36, timestamp)
    for _vau_route_point in _vau_route_points:
        x = _vau_route_point["x"]
        y = _vau_route_point["y"]
        z = _vau_route_point["z"]
        vel = 36
        while not is_arrived(_drone.get_location()[0], _drone.get_location()[1], x, y):  # 返回1到达，返回0没有到达
            _drone.control_kinetic_simply_global(x, y, z, vel, timestamp)
        # 每走一步长都调用一次侦察信息接口，但是要保证步长选择合理，避免多次冗余的调用（每隔几个步长调用）
        _res = _drone.detect_situation()
        whole_info = list(_res)
        detect_res = json.loads(whole_info[2])
        # 当侦测到信息后，分门别类放到对应的全局变量里
        if detect_res:
            for each_res in detect_res:  # detect_res返回的是100m内的目标，是一个大列表，会包含很多个目标，因此要遍历每个列表中的字典看看是不是strike类别的
                if each_res["Type"] == "strike":
                    # detect_res可能会包含之前已经探测到并存放到strike_info中的元素，因此要判断strike_info是否已经有了当前侦测到的目标
                    if not strike_info_is_repeated(each_res):
                        each_strike_info = [
                            {
                                "Name": each_res["Name"],
                                "Position": {  # 返回的位置是以cm为单位的，要除以100，注意原数据中xyz是大写的
                                    "x": round(each_res["Position"]["X"] / 100, 1),
                                    "y": round(each_res["Position"]["Y"] / 100, 1),
                                    "z": round(each_res["Position"]["Z"] / 100, 1)
                                },
                                "Type_Name": each_res["Type_Name"],
                                "HealthPoint": each_res["HealthPoint"]
                            }
                        ]
                        global strike_info
                        strike_info += each_strike_info
                        print(strike_info)


def reduced_target_point(route):  # route是一维列表，是一个无人机的所有轨迹点（字典格式）组成的列表，用于减少途径点，从而减少运行时间
    reduced_point = [route[0]]
    for j in range(1, len(route) - 1, 1):  # 遍历一次除起终点的所有点，筛选出需要的点
        # 判断方法：当前点与前一点和后一点是否共线；通过向量的行列式判断
        delta1_x = route[j]['x'] - route[j - 1]['x']
        delta1_y = route[j]['y'] - route[j - 1]['y']
        delta2_x = route[j + 1]['x'] - route[j]['x']
        delta2_y = route[j + 1]['y'] - route[j]['y']
        if abs(delta1_y * delta2_x - delta2_y * delta1_x) > 0.0001:  # 共线行列式绝对值等于0，怕有计算误差，这里判断条件是行列式的绝对值大于一个很小的数
            reduced_point.append(route[j])
    reduced_point.append(route[-1])
    return reduced_point


def vanguard(_routes_points):
    vau_routes_points = []
    drones = [sw_node8, sw_node9, sw_node10]
    threads = []
    # print(_routes_points)
    # reduced_routes = []
    # for i in range(3):
    #     reduced_routes.append(reduced_target_point(_routes_points[i]['route']))
    # print(reduced_routes)
    # 把带有id的路径信息提炼出来，vau_routes_points里面是一个三维的列表，每一维装的是单架无人机的路径坐标点的列表，从前到后分别对应node8、9、10
    for _route_points in _routes_points:
        vau_routes_points.append(_route_points["routes"])
    # 对每架飞机的路径创建一个线程，并行操作让无人机同时开始运动
    for idx, drone in enumerate(drones):
        # follow_route_points要执行无人机运动、不断查询目标、最终返回目标信息
        thread = threading.Thread(target=follow_route_points,
                                  args=(drone, vau_routes_points[idx], drone_start_points[idx]))
        thread.start()
        threads.append(thread)
    # 等待各线程结束
    for thread in threads:
        thread.join()


'''——————————————————————————————————————————————————'''

''' ——————————第3阶段无人车路径规划需要用到的函数—————————— '''


# 定时任务函数，用于定期检查是否需要重新规划
def periodic_replan(node, interval, map_rgb):
    global should_replan
    while True:
        time.sleep(interval)  # 等待指定时间
        if replan4detect(node, map_rgb):
            print('【3阶段】检测到障碍物')
            should_replan = True
            break

def replan4detect(node, map_rgb):
    have_obs = False
    current_detect_info = list(node.detect_situation())
    detect_obs_info = json.loads(current_detect_info[2])

    # 检测全部障碍物，版本1
    # detect_obses = []
    # for each_obs in detect_obs_info:
    #     if each_obs['Type'] == 'Other':
    #         detect_obses.append([round(each_obs['Position']['X'] / 100, 0),
    #                              round(each_obs['Position']['Y'] / 100, 0)])
    # if detect_obses:
    #     have_obs = True
    #     brakedown(node)
    #     for each_obs in detect_obses:
    #         obs_location_pixel = ue_to_pixel(each_obs)
    #         map_rgb = edit_map(map_rgb, obs_location_pixel)  # 调整地图，添加障碍物的位置为不可通行区域

    # 检测第一个就返回，版本2
    for each_obs in detect_obs_info:
        if each_obs['Type'] == 'Other':
            have_obs = True
            brakedown(node)
            obs_location_pixel = ue_to_pixel(each_obs)
            map_rgb = edit_map(map_rgb, obs_location_pixel)  # 调整地图，添加障碍物的位置为不可通行区域
            return have_obs

    return have_obs


def pixel_to_ue(_point_pixel):
    _ue_x = _point_pixel[0] / SCALE + OFFSET_X
    _ue_y = _point_pixel[1] / SCALE + OFFSET_Y
    return [_ue_x, _ue_y]


def ue_to_pixel(_point_ue):
    _pixel_x = int(SCALE * (_point_ue[0] - OFFSET_X))
    _pixel_y = int(SCALE * (_point_ue[1] - OFFSET_Y))
    return [_pixel_x, _pixel_y]


def cross_the_end(_node, __end_line):  # 判断指定节点是否已经通过终点线
    x_now, _, _, _ = _node.get_location()
    if x_now > __end_line[0]["x"]:
        return True
    else:
        return False


# def obstacle_info_is_repeated(target_dict):  # 判断之前存过的目标信息是否和当前侦察到的重合
#     if not obstacle_info:  # 当目标信息为空的时候，返回false表示不重复
#         return False
#     else:
#         for target in obstacle_info:  # detect_res可能会包含之前已经探测到并存放到obstacle_info中的元素，因此要判断obstacle_info是否已经有了当前侦测到的目标
#             if target_dict["Name"] == target["Name"]:
#                 return True
#         return False

'''——————A* begin——————'''

open_list = []
G_value = [[]]
H_value = [[]]
Parent = [[]]


class Position:
    x = 0
    y = 0

    def __init__(self, X, Y):
        self.x, self.y = X, Y


# 存储地图每点信息Message


def getHeu(i, j, point2):  # 计算H值
    dx = abs(i - point2.x)
    dy = abs(j - point2.y)
    hscore = dx + dy
    return hscore


def farfrom_obs(i, j, map):  # 想法是尽量在路中间走，先排除掉路边缘的地方
    dist = 8
    dist_min = 6
    dist2 = 4
    if (map[i + dist_min][j] and map[i - dist_min][j] and map[i][j - dist_min] and map[i][j + dist_min]
            and map[i + dist2][j + dist2] and map[i + dist2][j - dist2] and
            map[i - dist2][j + dist2] and map[i - dist2][j - dist2]):
        # if (board[i + dist][j].occupied + board[i - dist][j].occupied + board[i][j - dist].occupied + board[i][j + dist].occupied == 0):
        #     # and board[i + 30][j].occupied + board[i - 30][j].occupied + board[i][j - 30].occupied + board[i][j + 30].occupied == 2
        #     # and abs(board[i + 30][j].occupied + board[i - 30][j].occupied - board[i][j - 30].occupied - board[i][j + 30].occupied) == 2):
        #     return 0
        # else:
        return 1
    else:
        return 0


# 输入map，起始点像素坐标，终止点像素坐标，flag
def Path_Search(map, start_pos1, target_position1, flag):  # flag = 1,过线就行
    col = start_pos1[0]
    row = start_pos1[1]
    start_pos = Position(row, col)
    target_position = Position(target_position1[1], target_position1[0])
    rows = len(map)
    cols = len(map[0])

    IsUnk = [[1 for _ in range(cols)] for _ in range(rows)]
    Parent[row][col] = None  # 重要
    IsUnk[start_pos.x][start_pos.y] = 0

    open_list.clear()
    open_list.append(start_pos)
    # 以下，x为行，y为列
    while open_list:
        # 取出第一个（F最小，判定最优）位置
        current_position = open_list[0]
        # print("当前点:", current_position.y, current_position.x)
        x_pos = current_position.x
        y_pos = current_position.y
        open_list.remove(current_position)
        # close_list.append(current_position)
        # 到达
        if ((current_position.x == target_position.x and current_position.y == target_position.y) or
                (flag == 1 and current_position.y > target_position.y) or
                (flag == 2 and current_position.x <= target_position.x)):  # 当前点就是终点
            print("A*已生成最优路径")
            # 内存储Position
            tmp = []
            while current_position:
                tmp.append(current_position)
                current_position = Parent[current_position.x][current_position.y]  # 回溯父节点
            tmp.reverse()

            dict_tmp = []

            for i in tmp:
                # print(str(i.__dict__))
                i_dict = {"x": i.y, "y": i.x}  # 图像坐标系与数组行数列数
                dict_tmp.append(i_dict)
            return dict_tmp

        # 将下一步可到达的位置加入open_list，并检查记录的最短路径G是否需要更新，记录最短路径经过的上一个点
        for i in range(x_pos - 1, x_pos + 2):
            for j in range(y_pos - 1, y_pos + 2):
                if 0 <= i < rows and 0 <= j < cols and map[i][j]:  # 不超出边界,可通行
                    if (i - x_pos) * (j - y_pos) == 0 and (i - x_pos) + (j - y_pos) != 0:  # 不能斜着走；不是自己本身
                        if farfrom_obs(i, j, map) or getHeu(i, j, target_position) <= 6 or getHeu(i, j, start_pos) <= 6:
                            # 希望不要贴着边缘走
                            new_G = G_value[x_pos][y_pos] + 1
                            # 维护当前已知最短G
                            if IsUnk[i][j]:  # 如果未被检查
                                IsUnk[i][j] = 0
                                G_value[i][j] = new_G
                                H_value[i][j] = getHeu(i, j, target_position)
                                open_list.append(Position(i, j))  # 将周围待检查的点放入openlist
                                Parent[i][j] = Position(x_pos, y_pos)  # 将他们的父节点设为当前点
                            if G_value[i][j] > new_G:  # 如果未遍历且G值大于从当前点到该点的代价值，则需更新
                                Parent[i][j] = Position(x_pos, y_pos)
                                G_value[i][j] = new_G
        # open_list.sort(key=searchKey(board))
        # 对open_list里的内容按F的大小排序，从小到大
        open_list.sort(
            key=lambda elem: (G_value[elem.x][elem.y] + H_value[elem.x][elem.y]))  # 将各个待检查的节点都放入openlist中并进行排序


# 作搜索前处理，将原始彩色图初始化为0（有障碍）,1（无障碍）的地图
def Process_before(image):
    im_gray = image.convert("L")
    # print("A*地图初始化", time.time())
    mapp = np.array(im_gray)
    mapp = mapp.astype(bool)
    rows = len(mapp)
    cols = len(mapp[0])
    global G_value
    global H_value
    global Parent
    G_value = [[0 for _ in range(cols)] for _ in range(rows)]
    H_value = [[0 for _ in range(cols)] for _ in range(rows)]
    Parent = [[None for _ in range(cols)] for _ in range(rows)]
    # print("A*地图初始化完成", time.time())
    return mapp


def dict_slice(pos_dict):  # 输出Astar路线坐标切片（每隔5个输出1个）
    if not pos_dict:
        print("未找到路径,不能切片")
    else:
        dict5_tmp = pos_dict[::30]
        if len(pos_dict) % 30 != 1:
            dict5_tmp.append(pos_dict[-1])
        return dict5_tmp


# 输入参数：彩色图像，障碍像素坐标，返回彩色图像
def dict_obs_edit(im_RGB, pos):  # 作为obs_edit_image的子函数调用
    pixels_RGB = im_RGB.load()
    im_gray = im_RGB.convert("L")
    y = pos[1]
    x = pos[0]
    up = down = left = right = 0
    pixels = im_gray.load()
    edit_dict = []
    while pixels[x, y - up] == 255:
        up += 1
        if up > 40:
            break
    while pixels[x, y + down] == 255:
        down += 1
        if down > 40:
            break
    while pixels[x - left, y] == 255:
        left += 1
        if left > 40:
            break
    while pixels[x + right, y] == 255:
        right += 1
        if right > 40:
            break
    if up + down > left + right:  # 左右短，路是上下方向的，封住左右路的宽度为left+right-2+1
        for i in range(x - left, x + right):
            pixels_RGB[i, y] = (0, 0, 0)
            edit_dict1 = {"x": i, "y": y}
            edit_dict.append(edit_dict1)
    else:
        for i in range(y - up, y + down):
            pixels_RGB[x, i] = (0, 0, 0)
    return edit_dict


def editobs_1(im_RGB, edit_dict):  # 作为obs_edit_image的子函数调用
    for i in edit_dict:
        im_RGB.putpixel((i["x"], i["y"]), (0, 0, 0))
    return im_RGB


def obs_edit_image(im_RGB, pos):  # 修改彩色图片
    dict = dict_obs_edit(im_RGB, pos)
    im_edit = editobs_1(im_RGB, dict)
    return im_edit


def obs_edit_map(im_RGB):  # 修改map
    im_gray = (im_RGB.convert("L"))
    mapp = np.array(im_gray)
    mapp = mapp.astype(bool)
    return mapp


'''——————A* end————————'''


def find_path(_map_rgb, _start_point, _end_point, _flag):  # 传进去的是像素坐标，输出的也是像素坐标, flag是1过线就行，0是到终点
    pixel_map = Process_before(_map_rgb)
    pos_dict = Path_Search(pixel_map, _start_point, _end_point, _flag)
    pos_dict_30 = dict_slice(pos_dict)
    # print(pos_dict_30)
    # path_vis(_map_rgb, pos_dict)
    return pos_dict_30


def tga_to_array(tga_image):
    image = tga_image.convert("L")  # 将图像转换为灰度图像

    width, height = image.size
    pixel_values = list(image.getdata())

    # 将一维像素值列表转换为二维数组
    array = [pixel_values[i:i + width] for i in range(0, width * height, width)]

    return array


def find_element(matrix, _point):  # 上下右都扩大50搜索，寻找可通行的点
    left_edge = _point[0]
    right_edge = _point[0] + 50
    top_edge = _point[1] + 50
    bottom_edge = _point[1] - 50
    for _y in range(top_edge, bottom_edge - 1, -1):
        for _x in range(left_edge, right_edge + 1):
            if matrix[_y][_x] == 255:
                return [_x, _y]  # 返回找到的行和列
    return None  # 如果未找到匹配元素，返回 None


def edit_map(_map, _obstacle_location):  # 根据障碍位置修改地图中的不可通行区域， 输入参数：彩色图像，障碍像素坐标，返回彩色图像
    pixels_RGB = _map.load()
    im_gray = _map.convert("L")
    obs_y = _obstacle_location[1]
    obs_x = _obstacle_location[0]
    up_dir = down_dir = left_dir = right_dir = 0
    pixels = im_gray.load()
    while pixels[obs_x, obs_y - up_dir] == 255:
        up_dir += 1
        if up_dir > 40:
            break
    while pixels[obs_x, obs_y + down_dir] == 255:
        down_dir += 1
        if down_dir > 40:
            break
    while pixels[obs_x - left_dir, obs_y] == 255:
        left_dir += 1
        if left_dir > 40:
            break
    while pixels[obs_x + right_dir, obs_y] == 255:
        right_dir += 1
        if right_dir > 40:
            break
    if up_dir + down_dir > left_dir + right_dir:  # 左右短，路是上下方向的，封住左右路的宽度为left+right-2+1
        for i in range(obs_x - left_dir, obs_x + right_dir):
            pixels_RGB[i, obs_y] = (0, 0, 0)
    else:
        for i in range(obs_y - up_dir, obs_y + down_dir):
            pixels_RGB[obs_x, i] = (0, 0, 0)
    return _map


# 直行：x-方向，对于yaw为负时，yaw+360
# 转弯：x-方向，对于yaw为负时，yaw+360，这样可以保证向左转度数减少，保证转弯参数的通用性
# 只要满足右转度数变大，那么k_yaw就为正
def veh_go_to(_node, _path_point_x, _path_point_y, _path_point_z, _last):  # 控制车辆从_path_point_x运动到_path_point_y
    x0 = _path_point_x[0]  # 起始点横坐标
    y0 = _path_point_x[1]  # 起始点纵坐标
    x1 = _path_point_y[0]  # 未来第一个点横坐标
    y1 = _path_point_y[1]  # 未来第一个点纵坐标
    x2 = _path_point_z[0]  # 未来第二个点横坐标
    y2 = _path_point_z[1]  # 未来第二个点纵坐标
    # print(_path_point_x, _path_point_y, _path_point_z)
    cv = 7  # 临界值，当x或者y的变化量大于cv才判断为转弯
    cvd = 40  # 航向角临界值，航向角基于标准角度的偏移值小于cvd,判定为正方向
    cvv = 3  # 为了缩小倒车转弯的判定范围
    node_model = _node.get_node_info()[4]
    velocity = _node.get_velocity()[3]
    if node_model == 'vehicle.imv.rb':  # 01 02 03
        if velocity <= 5:
            target_v = 5
            r = 4.5
        elif 5 < velocity <= 6:
            target_v = 6
            r = 5.5
        elif 6 < velocity <= 7:
            target_v = 7
            r = 5
        elif 7 < velocity <= 8:
            target_v = 8
            r = 5.5
        elif 8 < velocity <= 9:
            target_v = 9
            r = 6
        elif 9 < velocity <= 10:
            target_v = 10
            r = 6.5
        elif 10 < velocity <= 11:
            target_v = 11
            r = 7
        elif 11 < velocity <= 12:
            target_v = 12
            r = 9
        elif 12 < velocity <= 13:
            target_v = 13
            r = 11
        elif 13 < velocity <= 14:
            target_v = 14
            r = 10
        elif 14 < velocity <= 15:
            target_v = 15
            r = 12
        elif 15 < velocity <= 16:
            target_v = 16
            r = 13
        else:  # >16
            target_v = 17
            r = 14
        r_reverse = 19
    elif node_model == 'vehicle.auto.electron':  # 04 05
        r = 13.7
        r_reverse = 18.5
    else:  # 06 07 'vehicle.auto.minedestruct'
        r = 16.2
        r_reverse = 23
    yaw = _node.get_attitude()[0]  # 获取朝向

    def judge_straight():
        if abs(yaw - 0) <= cvd:
            if abs(y0 - y1) <= cv and abs(y1 - y2) <= cv and x0 <= x1 <= x2:
                return 'straight', 'drive', 'x+', 'z'
            elif abs(y0 - y1) <= cv and abs(y1 - y2) <= cv and x2 <= x1 <= x0:
                return 'straight', 'reverse', 'x+', 'z'
        elif abs(yaw - 180) <= cvd or abs(yaw + 180) <= cvd:
            if abs(y0 - y1) <= cv and abs(y1 - y2) <= cv and x0 <= x1 <= x2:
                return 'straight', 'reverse', 'x-', 'z'
            elif abs(y0 - y1) <= cv and abs(y1 - y2) <= cv and x2 <= x1 <= x0:
                return 'straight', 'drive', 'x-', 'z'
        elif abs(yaw - 90) <= cvd:
            if abs(x0 - x1) <= cv and abs(x1 - x2) <= cv and y0 <= y1 <= y2:
                return 'straight', 'drive', 'y+', 'z'
            elif abs(x0 - x1) <= cv and abs(x1 - x2) <= cv and y2 <= y1 <= y0:
                return 'straight', 'reverse', 'y+', 'z'
        elif abs(yaw + 90) <= cvd:
            if abs(x0 - x1) <= cv and abs(x1 - x2) <= cv and y0 <= y1 <= y2:
                return 'straight', 'reverse', 'y-', 'z'
            elif abs(x0 - x1) <= cv and abs(x1 - x2) <= cv and y2 <= y1 <= y0:
                return 'straight', 'drive', 'y-', 'z'
        return 'z', 'z', 'z', 'z'

    def judge_turn():
        if abs(yaw - 0) <= cvd:
            if abs(y0 - y1) >= cv and x0 <= x1 + cvv:
                if y1 > y0:
                    return 'turn_latter', 'drive', 'x+', 'y+'
                else:  # y1 < y0
                    return 'turn_latter', 'drive', 'x+', 'y-'
            elif abs(y0 - y2) >= cv and x0 <= x2 + cvv:
                if abs(x1 - x2) > r:
                    return 'straight', 'drive', 'x+', 'x+'
                else:
                    if y2 > y0:
                        return 'turn_former', 'drive', 'x+', 'y+'
                    else:  # y2 < y0
                        return 'turn_former', 'drive', 'x+', 'y-'

            elif abs(y0 - y1) >= cv and x0 > x1 + cvv:
                if y1 > y0:
                    return 'turn_latter', 'reverse', 'x+', 'y+'
                else:  # y1 < y0
                    return 'turn_latter', 'reverse', 'x+', 'y-'
            elif abs(y0 - y2) >= cv and x0 > x2 + cvv:
                if abs(x1 - x2) > r_reverse:  # 此时不该转弯
                    return 'straight', 'reverse', 'x+', 'x+'
                else:
                    if y2 > y0:
                        return 'turn_former', 'reverse', 'x+', 'y+'
                    else:  # y2 < y0
                        return 'turn_former', 'reverse', 'x+', 'y-'
        elif abs(yaw - 180) <= cvd or abs(yaw + 180) <= cvd:
            if abs(y0 - y1) >= cv and x0 >= x1 - cvv:
                if y1 > y0:
                    return 'turn_latter', 'drive', 'x-', 'y+'
                else:  # y1 < y0
                    return 'turn_latter', 'drive', 'x-', 'y-'
            elif abs(y0 - y2) >= cv and x0 >= x2 - cvv:
                if abs(x1 - x2) > r:
                    return 'straight', 'drive', 'x-', 'x-'
                else:
                    if y2 > y0:
                        return 'turn_former', 'drive', 'x-', 'y+'
                    else:  # y2 < y0
                        return 'turn_former', 'drive', 'x-', 'y-'

            elif abs(y0 - y1) >= cv and x0 < x1 - cvv:
                if y1 > y0:
                    return 'turn_latter', 'reverse', 'x-', 'y+'
                else:  # y1 < y0
                    return 'turn_latter', 'reverse', 'x-', 'y-'
            elif abs(y0 - y2) >= cv and x0 < x2 - cvv:
                if abs(x1 - x2) > r_reverse:  # 此时不该转弯
                    return 'straight', 'reverse', 'x-', 'x-'
                else:
                    if y2 > y0:
                        return 'turn_former', 'reverse', 'x-', 'y+'
                    else:  # y2 < y0
                        return 'turn_former', 'reverse', 'x-', 'y-'
        elif abs(yaw - 90) <= cvd:
            if abs(x0 - x1) >= cv and y0 <= y1 + cvv:
                if x1 > x0:
                    return 'turn_latter', 'drive', 'y+', 'x+'
                else:  # x1 < x0
                    return 'turn_latter', 'drive', 'y+', 'x-'
            elif abs(x0 - x2) >= cv and y0 <= y2 + cvv:
                if abs(y1 - y2) >= r:  # 此时不该转弯
                    return 'straight', 'drive', 'y+', 'y+'
                else:
                    if x2 > x0:
                        return 'turn_former', 'drive', 'y+', 'x+'
                    else:  # x2 < x0
                        return 'turn_former', 'drive', 'y+', 'x-'
            elif abs(x0 - x1) >= cv and y0 > y1 + cvv:
                if x1 > x0:
                    return 'turn_latter', 'reverse', 'y+', 'x+'
                else:  # x1 < x0
                    return 'turn_latter', 'reverse', 'y+', 'x-'
            elif abs(x0 - x2) >= cv and y0 > y2 + cvv:
                if abs(y1 - y2) > r_reverse:  # 此时不该转弯
                    return 'straight', 'reverse', 'y+', 'y+'
                else:
                    if x2 > x0:
                        return 'turn_former', 'reverse', 'y+', 'x+'
                    else:  # x2 < x0
                        return 'turn_former', 'reverse', 'y+', 'x-'
        elif abs(yaw + 90) <= cvd:
            if abs(x0 - x1) >= cv and y0 >= y1 - cvv:
                if x1 > x0:
                    return 'turn_latter', 'drive', 'y-', 'x+'
                else:  # x1 < x0
                    return 'turn_latter', 'drive', 'y-', 'x-'
            elif abs(x0 - x2) >= cv and y0 >= y2 - cvv:
                if abs(y1 - y2) > r:  # 此时不该转弯
                    return 'straight', 'drive', 'y-', 'y-'
                else:
                    if x2 > x0:
                        return 'turn_former', 'drive', 'y-', 'x+'
                    else:  # x2 < x0
                        return 'turn_former', 'drive', 'y-', 'x-'
            elif abs(x0 - x1) >= cv and y0 < y1 - cvv:
                if x1 > x0:
                    return 'turn_latter', 'reverse', 'y-', 'x+'
                else:  # x1 < x0
                    return 'turn_latter', 'reverse', 'y-', 'x-'
            elif abs(x0 - x2) >= cv and y0 < y2 - cvv:
                if abs(y1 - y2) > r_reverse:  # 此时不该转弯
                    return 'straight', 'reverse', 'y-', 'y-'
                else:
                    if x2 > x0:
                        return 'turn_former', 'reverse', 'y-', 'x+'
                    else:  # x2 < x0
                        return 'turn_former', 'reverse', 'y-', 'x-'
        raise ValueError("每个判断条件都不满足")

    if _last[0] == 'start':
        if abs(yaw - 0) <= cvd:
            if x0 <= x2:
                state, mode, toward, target_toward = 'start_straight', 'drive', 'x+', 'z'
            else:  # x2 < x0
                state, mode, toward, target_toward = 'start_straight', 'reverse', 'x+', 'z'
        elif abs(yaw - 90) <= cvd:
            if y0 <= y2:
                state, mode, toward, target_toward = 'start_straight', 'drive', 'y+', 'z'
            else:  # y2 < y0
                state, mode, toward, target_toward = 'start_straight', 'reverse', 'y+', 'z'
        elif abs(yaw + 90) <= cvd:
            if y0 <= y2:
                state, mode, toward, target_toward = 'start_straight', 'reverse', 'y-', 'z'
            else:  # y2 < y0
                state, mode, toward, target_toward = 'start_straight', 'drive', 'y-', 'z'
        elif abs(yaw - 180) <= cvd or abs(yaw + 180) <= cvd:
            if x0 <= x2:
                state, mode, toward, target_toward = 'start_straight', 'reverse', 'x-', 'z'
            else:  # x2 < x0
                state, mode, toward, target_toward = 'start_straight', 'drive', 'x-', 'z'
        else:
            raise ValueError("start_straight错误")
    elif _last[0] == 'start_straight':
        return 'straight', _last[1], _last[2], _last[3]
    elif _last[0] == 'turn_former':
        return 'turn_latter', _last[1], _last[2], _last[3]
    elif _last[0] == 'turn_latter':
        state = 'straight'
        mode = 'drive'
        toward = _last[3]
        target_toward = _last[3]
    else:
        state, mode, toward, target_toward = judge_straight()
        if state == 'z':
            state, mode, toward, target_toward = judge_turn()
    if state == 'start_straight' or state == 'straight':
        if state == 'start_straight':
            target_x = x2
            target_y = y2
        else:  # 'straight'
            target_x = x1
            target_y = y1
        if toward == 'x+':
            target_yaw = 0
        elif toward == 'y+':
            target_yaw = 90
        elif toward == 'y-':
            target_yaw = -90
        else:  # x-
            target_yaw = 180

        if mode == 'drive':
            kp_yaw = 0.04
        else:  # reverse
            kp_yaw = -0.04

        if toward == 'x+':
            kp_width = 0.1
        elif toward == 'x-':
            kp_width = -0.1
        elif toward == 'y+':
            kp_width = -0.1
        else:  # y-
            kp_width = 0.1

        if mode == 'drive':
            throttle = 1
        else:
            throttle = -1

        while True:
            velocity = _node.get_velocity()[3]
            x = _node.get_location()[0]
            y = _node.get_location()[1]
            yaw = _node.get_attitude()[0]
            if toward == 'x-' and yaw <= 0:
                yaw = yaw + 360

            if mode == 'reverse' and node_model == 'vehicle.auto.minedestruct':
                e_v = 13.8 - velocity
                throttle = -1 * e_v

            if toward == 'x+' or toward == 'x-':  # 根据航向角判断横向控制应该是跟踪x还是跟踪y
                e_width = target_y - y
            else:  # y+ or y-
                e_width = target_x - x

            e_yaw = target_yaw - yaw
            u_steer = kp_yaw * e_yaw + kp_width * e_width  # 转向控制
            _node.apply_vehicle_control(throttle, u_steer, 0, False, 0)

            if (mode == 'drive' and toward == 'x+' and x >= target_x) or \
                    (mode == 'reverse' and toward == 'x+' and x <= target_x) or \
                    (mode == 'drive' and toward == 'x-' and x <= target_x) or \
                    (mode == 'reverse' and toward == 'x-' and x >= target_x) or \
                    (mode == 'drive' and toward == 'y+' and y >= target_y) or \
                    (mode == 'reverse' and toward == 'y+' and y <= target_y) or \
                    (mode == 'drive' and toward == 'y-' and y <= target_y) or \
                    (mode == 'reverse' and toward == 'y-' and y >= target_y):
                break
    elif state == 'turn_latter' or state == 'turn_former':
        if target_toward == 'x+':
            target_yaw = 0
        elif target_toward == 'y+':
            target_yaw = 90
        elif target_toward == 'y-':
            if toward == 'x-':
                target_yaw = 270
            else:
                target_yaw = -90
        elif target_toward == 'x-' and toward == 'y+':
            target_yaw = 180
        else:  # y- -> x-
            target_yaw = -180

        if mode == 'drive':
            k_v = 1
            # 直行一小段会用到的参数
            kp_yaw = 0.04
            if toward == 'x+':
                kp_width = 0.1
                straight_yaw = 0
            elif toward == 'x-':
                kp_width = -0.1
                straight_yaw = 180
            elif toward == 'y+':
                kp_width = -0.1
                straight_yaw = 90
            else:  # y-
                kp_width = 0.1
                straight_yaw = -90

            if state == 'turn_former':
                target_x = x2
                target_y = y2
            else:
                target_x = x1
                target_y = y1

            if node_model == 'vehicle.imv.rb':  # 01 02 03
                kp_yaw_turn = 0.043
            elif node_model == 'vehicle.auto.electron':  # 04 05
                kp_yaw_turn = 0.043
            else:  # 06 07 'vehicle.auto.minedestruct'
                kp_yaw_turn = 0.03

            while True:
                x = _node.get_location()[0]
                y = _node.get_location()[1]
                yaw = _node.get_attitude()[0]
                velocity = _node.get_velocity()[3]
                if node_model == 'vehicle.imv.rb':  # 01 02 03
                    e_v = target_v - velocity
                    throttle = e_v * k_v
                else:
                    throttle = 1
                if toward == 'x-' and yaw <= 0:
                    yaw = yaw + 360
                elif target_toward == 'x-' and toward == 'y+' and yaw <= 0:
                    yaw = yaw + 360
                elif target_toward == 'x-' and toward == 'y-' and yaw >= 0:
                    yaw = yaw - 360

                if target_toward == 'x+' or target_toward == 'x-':  # 转到x轴
                    if abs(target_y - y) <= r:
                        e_yaw = target_yaw - yaw
                        u_steer = kp_yaw_turn * e_yaw  # 转向控制
                    else:  # 先直行一段
                        e_width = x0 - x
                        e_yaw = straight_yaw - yaw
                        u_steer = kp_yaw * e_yaw + kp_width * e_width  # 转向控制
                else:  # y+ y-
                    if abs(target_x - x) <= r:
                        e_yaw = target_yaw - yaw
                        u_steer = kp_yaw_turn * e_yaw
                    else:  # 先直行一段
                        e_width = y0 - y
                        e_yaw = straight_yaw - yaw
                        u_steer = kp_yaw * e_yaw + kp_width * e_width  # 转向控制
                _node.apply_vehicle_control(throttle, u_steer, 0, False, 0)

                if (target_toward == 'x+' and x >= target_x) or \
                        (target_toward == 'x-' and x <= target_x) or \
                        (target_toward == 'y+' and y >= target_y) or \
                        (target_toward == 'y-' and y <= target_y):
                    break

        else:  # reverse

            if 128 <= y0 <= 138 and toward == 'x+':  #最右侧道路的倒车转弯的处理方法
                kp_width = 0.1
                kp_yaw = -0.04
                target_toward = 'y-'
                if state == 'turn_former':
                    target_x = x2 + 1
                else:
                    target_x = x1 + 1
                first_iteration = True
                while True:
                    x = _node.get_location()[0]
                    y = _node.get_location()[1]
                    yaw = _node.get_attitude()[0]
                    e_width = y0 - y
                    e_yaw = 0 - yaw
                    u_steer = kp_yaw * e_yaw + kp_width * e_width  # 转向控制
                    if x <= target_x:
                        _node.apply_vehicle_control(1, u_steer, 1, True, 0)
                        if first_iteration:
                            start_time = time.time()
                            first_iteration = False
                        if (time.time() - start_time) >= 1:
                            break

                if state == 'turn_former':
                    target_x = x2
                    target_y = y2
                else:
                    target_x = x1
                    target_y = y1
                if node_model == 'vehicle.imv.rb':  # 01 02 03
                    kp_yaw_turn = 0.043
                elif node_model == 'vehicle.auto.electron':  # 04 05
                    kp_yaw_turn = 0.043
                else:  # 06 07 'vehicle.auto.minedestruct'
                    kp_yaw_turn = 0.03
                throttle = 1
                while True:
                    x = _node.get_location()[0]
                    y = _node.get_location()[1]
                    yaw = _node.get_attitude()[0]
                    velocity = _node.get_velocity()[3]
                    e_yaw = -90 - yaw
                    u_steer = kp_yaw_turn * e_yaw
                    _node.apply_vehicle_control(throttle, u_steer, 0, False, 0)
                    if target_toward == 'y-' and y <= target_y:
                        break
                return state, mode, toward, target_toward
            if state == 'turn_former':
                target_x = x2
                target_y = 2 * y0 - y2
            else:
                target_x = x1
                target_y = 2 * y0 - y1
            throttle = -1
            # 直行一小段会用到的参数
            kp_yaw = -0.04
            if toward == 'x+':
                kp_width = 0.1
                straight_yaw = 0
            elif toward == 'x-':
                kp_width = -0.1
                straight_yaw = 180
            elif toward == 'y+':
                kp_width = -0.1
                straight_yaw = 90
            else:  # y-
                kp_width = 0.1
                straight_yaw = -90

            if node_model == 'vehicle.imv.rb':  # 01 02 03
                kp_yaw_turn = -0.003
            elif node_model == 'vehicle.auto.electron':  # 04 05
                kp_yaw_turn = -0.004
            else:  # 06 07 'vehicle.auto.minedestruct'
                kp_yaw_turn = -0.006

            first_iteration = True

            while True:
                velocity = _node.get_velocity()[3]
                x = _node.get_location()[0]
                y = _node.get_location()[1]
                yaw = _node.get_attitude()[0]
                if toward == 'x-' and yaw <= 0:
                    yaw = yaw + 360
                elif target_toward == 'x-' and toward == 'y+' and yaw <= 0:
                    yaw = yaw + 360
                elif target_toward == 'x-' and toward == 'y-' and yaw >= 0:
                    yaw = yaw - 360

                if node_model == 'vehicle.auto.minedestruct':
                    e_v = 13.8 - velocity
                    throttle = -1 * e_v

                if target_toward == 'x+' or target_toward == 'x-':  # 转到x轴
                    if abs(target_y - y) <= r_reverse:
                        e_yaw = target_yaw - yaw
                        u_steer = kp_yaw_turn * e_yaw  # 转向控制
                    else:  # 先直行一段
                        e_width = x0 - x
                        e_yaw = straight_yaw - yaw
                        u_steer = kp_yaw * e_yaw + kp_width * e_width  # 转向控制
                else:  # y+ y-
                    if abs(target_x - x) <= r_reverse:
                        e_yaw = target_yaw - yaw
                        u_steer = kp_yaw_turn * e_yaw
                    else:  # 先直行一段
                        e_width = y0 - y
                        e_yaw = straight_yaw - yaw
                        u_steer = kp_yaw * e_yaw + kp_width * e_width  # 转向控制

                if (target_toward == 'y+' and y <= target_y) or \
                        (target_toward == 'y-' and y >= target_y) or \
                        (target_toward == 'x+' and x <= target_x) or \
                        (target_toward == 'x-' and x >= target_x):
                    _node.apply_vehicle_control(1, u_steer, 1, True, 0)
                    if first_iteration:
                        start_time = time.time()
                        first_iteration = False
                    if (time.time() - start_time) >= 1:
                        break
                else:
                    _node.apply_vehicle_control(throttle, u_steer, 0, False, 0)
    return state, mode, toward, target_toward


def keep_straight(_vehicle, _y):
    kp_yaw = 0.04
    kp_width = 0.1
    yaw = _vehicle.get_attitude()[0]
    y = _vehicle.get_location()[1]
    e_yaw = 0 - yaw
    e_width = _y - y
    u_steer = kp_yaw * e_yaw + kp_width * e_width
    _vehicle.apply_vehicle_control(1, u_steer, 0, False, 0)


def tail_the_explorer(map_rgb, node, end_point_fine_pixel, _end_line_ue, yield_duration, follower_path):
    print(node.get_node_info()[1],"start tail")
    last_state = 'start'
    last_mode = 'z'
    last_toward = 'z'
    last_target_toward = 'z'
    last = [last_state, last_mode, last_toward, last_target_toward]
    for _index, _path_point in enumerate(follower_path):
        _path_point_pixel = [_path_point["x"], _path_point["y"]]
        _path_point_ue = pixel_to_ue(_path_point_pixel)

        if _index < len(follower_path) - 1:
            path_point_next_pixel = [follower_path[_index + 1]["x"], follower_path[_index + 1]["y"]]
        else:  # 最后一个点时
            path_point_next_pixel = [_path_point["x"], _path_point["y"]]
        _path_point_next_ue = pixel_to_ue(path_point_next_pixel)  # 当前要前往的坐标点的下一个坐标点，对于最后一个点要怎么处理??

        _cur_location_ue = [node.get_location()[0], node.get_location()[1]]  # 当前的位置坐标
        last = list(veh_go_to(node, _cur_location_ue, _path_point_ue, _path_point_next_ue, last))

        if cross_the_end(node, _end_line_ue):
            start_time = time.time()
            while time.time() - start_time < yield_duration:
                cur_y = node.get_location()[1]
                keep_straight(node, cur_y)
            brakedown(node)
            break


def follow_the_explorer(vehicle_nodes, map_rgb, end_point_fine_pixel, _end_line_ue):
    threads = []
    sw_followers = [vehicle_nodes[5], vehicle_nodes[0], vehicle_nodes[1], vehicle_nodes[2], vehicle_nodes[3], vehicle_nodes[4]]

    for i, follower in enumerate(sw_followers):
        start_point_ue = [follower.get_location()[0], follower.get_location()[1]]
        start_point_pixel = ue_to_pixel(start_point_ue)
        follower_path = find_path(map_rgb, start_point_pixel, end_point_fine_pixel, 1)  # start_point, end_point是列表格式
        thread = threading.Thread(target=tail_the_explorer, args=(map_rgb, follower, end_point_fine_pixel, _end_line_ue, 5 - i, follower_path))
        thread.start()
        threads.append(thread)
        time.sleep(10)

    for thread in threads:
        thread.join()


def plan_and_follow(_vehicle_node, _road_net_image, _start_line_ue, _end_line_ue):
    explorer = _vehicle_node[6]
    map_rgb = _road_net_image  # 取出路网rgb地图
    map_array = list(tga_to_array(map_rgb))
    start_point_ue = [explorer.get_location()[0], explorer.get_location()[1]]
    start_point_pixel = ue_to_pixel(start_point_ue)
    # 终点选在x方向超过终点线5m，y方向和出发时一样，让车辆尽可能沿着直线走少拐弯。但仅限于终点线是向x方向时
    end_point_coarse_ue = [_end_line_ue[0]["x"] + 5, explorer.get_location()[1]]
    end_point_coarse_pixel = ue_to_pixel(end_point_coarse_ue)
    end_point_fine_pixel = find_element(map_array, end_point_coarse_pixel)
    if end_point_fine_pixel is not None:
        print("【3阶段】找到explorer目的地。坐标：", end_point_fine_pixel)
    else:
        print("【3阶段】未找到explorer目的地")

    # fsm
    while not cross_the_end(explorer, _end_line_ue):  # 当头车没有到达指定的停止线时，进行路径探索
        explorer_path_cur = find_path(map_rgb, start_point_pixel, end_point_fine_pixel,
                                      1)  # start_point, end_point是列表格式
        # print(explorer_path_cur)  # 在这打印出的路径

        # 初始化状态
        last_state = 'start'
        last_mode = 'z'
        last_toward = 'z'
        last_target_toward = 'z'
        last = [last_state, last_mode, last_toward, last_target_toward]

        # 逐点跟踪
        for index, path_point in enumerate(explorer_path_cur):  # 规划出的路径是由字典构成的列表
            path_point_pixel = [path_point["x"], path_point["y"]]
            path_point_ue = pixel_to_ue(path_point_pixel)  # 当前要前往的坐标点
            if index < len(explorer_path_cur) - 1:
                path_point_next_pixel = [explorer_path_cur[index + 1]["x"], explorer_path_cur[index + 1]["y"]]
            else:  # 最后一个点时
                path_point_next_pixel = [path_point["x"], path_point["y"]]
            path_point_next_ue = pixel_to_ue(path_point_next_pixel)
            cur_location = [explorer.get_location()[0], explorer.get_location()[1]]  # 当前的位置坐标
            whole_info_veh = list(explorer.detect_situation())
            detect_res_veh = json.loads(whole_info_veh[2])

            have_obs_flag = False
            for each_detect_res_veh in detect_res_veh:
                if each_detect_res_veh["Type"] == "Other":
                    have_obs_flag = True
                    obs_location_ue = [
                        round(each_detect_res_veh["Position"]["X"] / 100, 0),
                        round(each_detect_res_veh["Position"]["Y"] / 100, 0)
                    ]  # 障碍物坐标读出来
                    print("【3阶段】explorer_replan")
                    break  # 就读取一个other

            if have_obs_flag:  # 100m内有障碍物
                while explorer.get_velocity()[0] or explorer.get_velocity()[1]:
                    explorer.apply_vehicle_control(0, 0, 0, True, 0)
                explorer.apply_vehicle_control(-1, 0, 0, False, 0)
                time.sleep(10)
                explorer.apply_vehicle_control(0, 0, 0, True, 0)
                obs_location_pixel = ue_to_pixel(obs_location_ue)
                map_rgb = edit_map(map_rgb, obs_location_pixel)  # 调整地图，添加障碍物的位置为不可通行区域
                stay_x, stay_y, _, _ = explorer.get_location()
                start_point_reset = [stay_x, stay_y]  # 将下次A*规划的起始位置设置成当前停车点
                start_point_pixel = ue_to_pixel(start_point_reset)
                break
            else:  # 100m内无障碍物
                last = list(veh_go_to(explorer, cur_location, path_point_ue, path_point_next_ue, last))  # 输入两个轨迹点时候的函数

    time.sleep(2)
    brakedown(explorer)

    follow_the_explorer(_vehicle_node[:6], map_rgb, end_point_fine_pixel, _end_line_ue)  # 后方车辆跟随头车轨迹行驶


'''——————————————————————————————————————————————————'''

''' ——————————第4阶段扫雷车路径规划需要用到的函数—————————— '''


def is_repeat_line(lst, target, threshold):
    for num in lst:
        if abs(num - target) <= threshold:
            return True
    return False


def get_roads_in_region(_img, _lb_point_pixel, _rt_point_pixel, threshold=128):
    # 将 TGA 图片对象转换为 numpy 数组
    img_array = np.array(_img, dtype=np.uint8)
    # 将 numpy 数组转换为 OpenCV 图像对象
    image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # 1. 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2. 二值化灰度图
    _, binary = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    # 3. 裁剪小矩形区域
    sub_image = binary[_lb_point_pixel[1]:_rt_point_pixel[1], _lb_point_pixel[0]:_rt_point_pixel[0]]
    # 4. 边缘检测获取道路边界
    edges = cv2.Canny(sub_image, 10, 200)
    # 5. 识别横向和纵向的路并记录起始点和终点坐标
    # 参数可以再调，尽可能让所有线段都出现在lines里
    lines = cv2.HoughLinesP(edges, rho=0.5, theta=np.pi / 360, threshold=50, minLineLength=100, maxLineGap=300)
    # print(lines)
    h_road_line = []
    v_road_line = []
    horizontal_roads = []
    vertical_roads = []
    h_length = int(abs(_lb_point_pixel[0] - _rt_point_pixel[0]))
    v_length = int(abs(_lb_point_pixel[1] - _rt_point_pixel[1]))
    # 6. 提取道路边界，lines里面有很多线段，因此只要y1y2值接近的就是水平道路，x1x2值接近的就是垂直道路
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) <= 6:
            y0 = int((y1 + y2) / 2)
            # 同一个水平道路可能被分成多个线段，因此append前要判断这条线段所在的直线是否已经添加过
            if is_repeat_line(h_road_line, y0, 5):
                continue
            h_road_line.append(y0)
        if abs(x1 - x2) <= 6:
            x0 = int((x1 + x2) / 2)
            # 同一个垂直道路可能被分成多个线段，因此append前要判断这条线段所在的直线是否已经添加过
            if is_repeat_line(v_road_line, x0, 5):
                continue
            v_road_line.append(x0)
    h_road_line.sort(reverse=True)
    v_road_line.sort(reverse=True)
    # h_road_line、v_road_line都是在小图片里的坐标，要返回在整个图像坐标系里的坐标还要加上左上角在整个图片坐标系中的坐标
    for i in range(0, len(h_road_line), 2):
        horizontal_road = {"top_y": h_road_line[i] + _lb_point_pixel[1],
                           "bottom_y": h_road_line[i + 1] + _lb_point_pixel[1],
                           "start_x": 0 + _lb_point_pixel[0],
                           "end_x": h_length + _lb_point_pixel[0],
                           "width": h_road_line[i] - h_road_line[i + 1]}
        horizontal_roads.append(horizontal_road)
    for j in range(0, len(v_road_line), 2):
        vertical_road = {"left_x": v_road_line[j + 1] + _lb_point_pixel[0],
                         "right_x": v_road_line[j] + _lb_point_pixel[0],
                         "start_y": 0 + _lb_point_pixel[1],
                         "end_y": v_length + _lb_point_pixel[1],
                         "width": v_road_line[j] - v_road_line[j + 1]}
        vertical_roads.append(vertical_road)
    # 返回的是整个图像坐标系下的像素坐标
    return horizontal_roads, vertical_roads


def brakedown(_node):
    _node.apply_vehicle_control(0, 0, 1, True, 0)
    _node.apply_vehicle_control(-1, 0, 0, False, 0)
    time.sleep(0.1)
    while _node.get_velocity()[0] or _node.get_velocity()[1]:
        _node.apply_vehicle_control(0, 0, 1, True, 0)


def _reverse_brake(_node):
    _node.apply_vehicle_control(0, 0, 1, True, 0)
    _node.apply_vehicle_control(1, 0, 0, False, 0)
    time.sleep(0.1)
    while _node.get_velocity()[0] or _node.get_velocity()[1]:
        _node.apply_vehicle_control(0, 0, 1, True, 0)


def veh_move(_veh, _path):  # 输入像素坐标点，让车按照规划的轨迹走
    last_state = 'start'
    last_mode = 'z'
    last_toward = 'z'
    last_target_toward = 'z'
    last = [last_state, last_mode, last_toward, last_target_toward]
    for index, _path_point in enumerate(_path):
        _path_point_pixel = [_path_point["x"], _path_point["y"]]
        _path_point_ue = pixel_to_ue(_path_point_pixel)
        if index < len(_path) - 1:
            path_point_next_pixel = [_path[index + 1]["x"], _path[index + 1]["y"]]
        else:  # 最后一个点时
            path_point_next_pixel = [_path_point["x"], _path_point["y"]]
        path_point_next_ue = pixel_to_ue(path_point_next_pixel)  # 当前要前往的坐标点的下一个坐标点，对于最后一个点要怎么处理??
        _cur_location_ue = [_veh.get_location()[0], _veh.get_location()[1]]  # 当前的位置坐标
        last = list(veh_go_to(_veh, _cur_location_ue, _path_point_ue, path_point_next_ue, last))
    _drive_brake(_veh)


def sweeper_go_to(__sweeper, _sweep_route, _flag):  # 给扫雷车定制的轨迹跟踪方案（文博）flag为0调用横向的，为1调用纵向的
    # x为横向 y为纵向 这里设置的临界值为5，即横向超过5m认为需要变道
    # preprocessing函数是为了将相邻的两个直角转弯点的后一个删去
    # 横向调用x开头的两个函数，纵向调用y开头的两个函数
    def x_preprocessing(data):
        filtered_data = []
        # 遍历原始数据列表
        for i in range(len(data)):
            # 如果是第一个点，或者当前点的x值与上一个点的x值不相等，将当前点添加到筛选后的数据列表中
            if i == 0 or data[i]['x'] != data[i - 1]['x']:
                filtered_data.append(data[i])
        return filtered_data

    def x_go_to(_node, _path_point_x, _path_point_y):
        node_model = _node.get_node_info()[4]
        if node_model != 'vehicle.auto.minedestruct':
            raise ValueError("扫雷选错车了")
        x1 = _path_point_x[0]  # 起始点横坐标
        y1 = _path_point_x[1]  # 起始点纵坐标
        x2 = _path_point_y[0]  # 终点横坐标
        y2 = _path_point_y[1]  # 终点纵坐标
        cv = 5
        if abs(y2 - y1) >= cv:
            state = 'change'
            if x2 <= x1:
                _drive_brake(_node)
            else:  # x2>x1
                _reverse_brake(_node)
        else:
            state = 'straight'
        if x2 - x1 >= 0:
            direction = 'x+'
            throttle = 1
        else:
            direction = 'x-'
            throttle = -1
        target_yaw = 0
        if state == 'straight':
            kp_width = 0.1
            if direction == 'x-':
                kp_yaw = -0.04
            else:
                kp_yaw = 0.04
        else:
            kp_width = 0.4
            if direction == 'x-':
                kp_yaw = -0.08
            else:
                kp_yaw = 0.08
        while True:
            x, y, z, sw_time = _node.get_location()
            yaw, pitch, roll, heading, frame_timestamp = _node.get_attitude()
            e_yaw = target_yaw - yaw
            e_width = y2 - y
            u_steer = kp_yaw * e_yaw + kp_width * e_width  # 转向控制
            _node.apply_vehicle_control(throttle, u_steer, 0, False, 0)
            if direction == 'x+':  # 到达终点附件结束循环
                if x >= x2:
                    break
            else:
                if x <= x2:
                    break

    def y_preprocessing(data):
        filtered_data = []
        # 遍历原始数据列表
        for i in range(len(data)):
            # 如果是第一个点，或者当前点的y值与上一个点的y值不相等，将当前点添加到筛选后的数据列表中
            if i == 0 or data[i]['y'] != data[i - 1]['y']:
                filtered_data.append(data[i])
        return filtered_data

    def y_go_to(_node, _path_point_x, _path_point_y):
        node_model = _node.get_node_info()[4]
        if node_model != 'vehicle.auto.minedestruct':
            raise ValueError("扫雷选错车了")
        x1 = _path_point_x[0]  # 起始点横坐标
        y1 = _path_point_x[1]  # 起始点纵坐标
        x2 = _path_point_y[0]  # 终点横坐标
        y2 = _path_point_y[1]  # 终点纵坐标
        cv = 5
        if abs(x2 - x1) >= cv:
            state = 'change'
            if y2 <= y1:
                _drive_brake(_node)
            else:  # x2>x1
                _reverse_brake(_node)
        else:
            state = 'straight'
        if y2 - y1 >= 0:
            direction = 'y+'
            throttle = 1
        else:
            direction = 'y-'
            throttle = -1
        target_yaw = 90
        if state == 'straight':
            kp_width = -0.1
            if direction == 'y-':
                kp_yaw = -0.04
            else:
                kp_yaw = 0.04
        else:
            kp_width = -0.4
            if direction == 'y-':
                kp_yaw = -0.08
            else:
                kp_yaw = 0.08
        print(state)
        while True:
            x, y, z, sw_time = _node.get_location()
            yaw, pitch, roll, heading, frame_timestamp = _node.get_attitude()
            e_yaw = target_yaw - yaw
            e_width = x2 - x
            u_steer = kp_yaw * e_yaw + kp_width * e_width  # 转向控制
            _node.apply_vehicle_control(throttle, u_steer, 0, False, 0)
            if direction == 'y+':  # 到达终点附件结束循环
                if y >= y2:
                    break
            else:
                if y <= y2:
                    break

    if _flag:
        sweeper_path_cur = y_preprocessing(_sweep_route)
        for index, path_point in enumerate(sweeper_path_cur):  # 规划出的路径是由字典构成的列表
            path_point_ue = [path_point["x"], path_point["y"]]  # 当前要前往的坐标点
            cur_location = [__sweeper.get_location()[0], __sweeper.get_location()[1]]  # 当前的位置坐标
            # print(cur_location, path_point_ue)
            y_go_to(__sweeper, cur_location, path_point_ue)
    else:
        sweeper_path_cur = x_preprocessing(_sweep_route)
        for index, path_point in enumerate(sweeper_path_cur):  # 规划出的路径是由字典构成的列表
            path_point_ue = [path_point["x"], path_point["y"]]  # 当前要前往的坐标点
            cur_location = [__sweeper.get_location()[0], __sweeper.get_location()[1]]  # 当前的位置坐标
            # print(cur_location, path_point_ue)
            x_go_to(__sweeper, cur_location, path_point_ue)


def sweeper_hor_plan(__sweeper, _hor_road, _ver_road, _to_hor_mine_path):  # 单个扫雷车横纵道路轨迹规划
    # 横向道路轨迹规划
    hor_lb_ue = pixel_to_ue([_hor_road["start_x"], _hor_road["bottom_y"]])
    hor_rt_ue = pixel_to_ue([_hor_road["end_x"], _hor_road["top_y"]])
    hor_start_point_ue = [hor_lb_ue[0], hor_lb_ue[1] + 3]
    veh_move(__sweeper, _to_hor_mine_path)  # 扫雷车移动到横向扫雷区起点

    hor_sweeper_current_x = hor_start_point_ue[0]
    hor_sweeper_current_y = hor_start_point_ue[1] + 1.5  # +1是为了避免道路倾斜车上马路牙
    hor_para_step = 20  # 水平移动步长
    hor_vert_step = 6  # 垂直移动步长
    hor_n = int(0.13 * _hor_road["width"]) + 1
    hor_sweep_route = []
    hor_direction = "right"
    while hor_n:
        # 向右水平运动，x每次加hor_para_step，直到运动到道路边界
        if hor_direction == "right":
            while (hor_sweeper_current_x + hor_para_step) < hor_rt_ue[0] - 55:
                hor_sweeper_current_x += hor_para_step
                hor_sweep_route.append({"x": round(hor_sweeper_current_x, 1), "y": round(hor_sweeper_current_y, 1)})
        # 向左水平运动，x每次减hor_para_step，直到运动到道路边界
        else:
            while (hor_sweeper_current_x - hor_para_step) > hor_lb_ue[0] + 55:
                hor_sweeper_current_x -= hor_para_step
                hor_sweep_route.append({"x": round(hor_sweeper_current_x, 1), "y": round(hor_sweeper_current_y, 1)})
        hor_sweeper_current_y += hor_vert_step
        hor_sweep_route.append({"x": round(hor_sweeper_current_x, 1), "y": round(hor_sweeper_current_y, 1)})
        hor_direction = "left" if hor_direction == "right" else "right"
        hor_n -= 1
    print(hor_sweep_route)
    sweeper_go_to(__sweeper, hor_sweep_route, 0)  # 规划完之后让车跟着轨迹走，0横向
    _drive_brake(__sweeper)


def sweeper_ver_plan(__sweeper, _hor_road, _ver_road, _to_ver_mine_path):  # 单个扫雷车横纵道路轨迹规划
    ver_lb_ue = pixel_to_ue([_ver_road["left_x"], _ver_road["start_y"]])
    ver_rt_ue = pixel_to_ue([_ver_road["right_x"], _ver_road["end_y"]])
    ver_start_point_ue = [ver_rt_ue[0] - 3, ver_lb_ue[1]]
    veh_move(__sweeper, _to_ver_mine_path[:-2])  # 扫雷车移动到纵向扫雷区起点，去除了最后一个点

    ver_sweeper_current_x = ver_start_point_ue[0] + 1  # +1是为了避免道路倾斜车上马路牙
    ver_sweeper_current_y = ver_start_point_ue[1]
    ver_para_step = 6  # 水平移动步长
    ver_vert_step = 20  # 垂直移动步长
    ver_n = int(0.13 * _hor_road["width"]) + 1
    ver_sweep_route = []
    ver_direction = "up"
    while ver_n:
        # 向上垂直运动，y每次加ver_vert_step，直到运动到道路边界
        if ver_direction == "up":
            while (ver_sweeper_current_y + ver_vert_step) < ver_rt_ue[1] - 55:
                ver_sweeper_current_y += ver_vert_step
                ver_sweep_route.append({"x": round(ver_sweeper_current_x, 1), "y": round(ver_sweeper_current_y, 1)})
        # 向下垂直运动，y每次减ver_vert_step，直到运动到道路边界
        else:
            while (ver_sweeper_current_y - ver_para_step) > ver_lb_ue[1] + 55:
                ver_sweeper_current_y -= ver_vert_step
                ver_sweep_route.append({"x": round(ver_sweeper_current_x, 1), "y": round(ver_sweeper_current_y, 1)})
        ver_sweeper_current_x -= ver_para_step
        ver_sweep_route.append({"x": round(ver_sweeper_current_x, 1), "y": round(ver_sweeper_current_y, 1)})
        ver_direction = "down" if ver_direction == "up" else "up"
        ver_n -= 1
    print(ver_sweep_route)
    sweeper_go_to(__sweeper, ver_sweep_route, 1)  # 规划完之后让车跟着轨迹走，1纵向
    _drive_brake(__sweeper)


def plan_sweep_route(_sweepers, _hor_roads, _ver_roads):  # 给两个扫雷车分别开一个线程，一个车管一横一纵两条路
    to_hor_mine_path = []
    hor_lb_ue_0 = pixel_to_ue([_hor_roads[0]["start_x"], _hor_roads[0]["bottom_y"]])
    hor_rt_ue_0 = pixel_to_ue([_hor_roads[0]["end_x"], _hor_roads[0]["top_y"]])
    hor_sweeper_cur_location_ue_0 = [_sweepers[0].get_location()[0], _sweepers[0].get_location()[1]]
    hor_sweeper_cur_location_pixel_0 = ue_to_pixel(hor_sweeper_cur_location_ue_0)
    hor_start_point_ue_0 = [hor_lb_ue_0[0] + 50, hor_lb_ue_0[1] + 3]
    hor_start_point_pixel_0 = ue_to_pixel(hor_start_point_ue_0)
    hor_cur_start_path_0 = find_path(map_img, hor_sweeper_cur_location_pixel_0, hor_start_point_pixel_0,
                                     0)  # map还得传进来？？？要改成像素坐标！！！
    to_hor_mine_path.append(hor_cur_start_path_0)

    hor_lb_ue_1 = pixel_to_ue([_hor_roads[1]["start_x"], _hor_roads[1]["bottom_y"]])
    hor_rt_ue_1 = pixel_to_ue([_hor_roads[1]["end_x"], _hor_roads[1]["top_y"]])
    hor_sweeper_cur_location_ue_1 = [_sweepers[1].get_location()[0], _sweepers[1].get_location()[1]]
    hor_sweeper_cur_location_pixel_1 = ue_to_pixel(hor_sweeper_cur_location_ue_1)
    hor_start_point_ue_1 = [hor_lb_ue_1[0] + 50, hor_lb_ue_1[1] + 3]
    hor_start_point_pixel_1 = ue_to_pixel(hor_start_point_ue_1)
    hor_cur_start_path_1 = find_path(map_img, hor_sweeper_cur_location_pixel_1, hor_start_point_pixel_1,
                                     0)  # map还得传进来？？？要改成像素坐标！！！
    to_hor_mine_path.append(hor_cur_start_path_1)

    threads1 = []
    for i, _sweeper in enumerate(_sweepers):
        if i != 0:
            time.sleep(10)  # 让各个车之间错开60秒
        thread = threading.Thread(target=sweeper_hor_plan,
                                  args=(_sweeper, _hor_roads[i], _ver_roads[i], to_hor_mine_path[i]))
        thread.start()
        threads1.append(thread)
    for thread in threads1:
        thread.join()

    to_ver_mine_path = []
    ver_lb_ue_0 = pixel_to_ue([_ver_roads[0]["left_x"], _ver_roads[0]["start_y"]])
    ver_rt_ue_0 = pixel_to_ue([_ver_roads[0]["right_x"], _ver_roads[0]["end_y"]])
    ver_sweeper_cur_location_ue_0 = [_sweepers[0].get_location()[0] - 20, _sweepers[0].get_location()[1] - 3]
    ver_sweeper_cur_location_pixel_0 = ue_to_pixel(ver_sweeper_cur_location_ue_0)
    ver_start_point_ue_0 = [ver_rt_ue_0[0] - 3, ver_lb_ue_0[1] + 50]
    ver_start_point_pixel_0 = ue_to_pixel(ver_start_point_ue_0)
    ver_cur_start_path_0 = find_path(map_img, ver_sweeper_cur_location_pixel_0, ver_start_point_pixel_0,
                                     0)  # map还得传进来？？？
    to_ver_mine_path.append(ver_cur_start_path_0)

    ver_lb_ue_1 = pixel_to_ue([_ver_roads[1]["left_x"], _ver_roads[1]["start_y"]])
    ver_rt_ue_1 = pixel_to_ue([_ver_roads[1]["right_x"], _ver_roads[1]["end_y"]])
    ver_sweeper_cur_location_ue_1 = [_sweepers[1].get_location()[0] - 20, _sweepers[1].get_location()[1] - 3]
    ver_sweeper_cur_location_pixel_1 = ue_to_pixel(ver_sweeper_cur_location_ue_1)
    ver_start_point_ue_1 = [ver_rt_ue_1[0] - 3, ver_lb_ue_1[1] + 50]
    ver_start_point_pixel_1 = ue_to_pixel(ver_start_point_ue_1)
    ver_cur_start_path_1 = find_path(map_img, ver_sweeper_cur_location_pixel_1, ver_start_point_pixel_1,
                                     0)  # map还得传进来？？？
    to_ver_mine_path.append(ver_cur_start_path_1)

    threads2 = []
    for i, _sweeper in enumerate(_sweepers):
        if i != 0:
            time.sleep(25)  # 让各个车之间错开60秒
        thread = threading.Thread(target=sweeper_ver_plan,
                                  args=(_sweeper, _hor_roads[i], _ver_roads[i], to_ver_mine_path[i]))
        thread.start()
        threads2.append(thread)
    for thread in threads2:
        thread.join()


'''——————————————————————————————————————————————————'''

''' ——————————————第5阶段电磁压制需要用到的函数———————————— '''


class EMC:
    def __init__(self, arena, work, theta, radius, target_area0, target_area1):
        self.arena = arena  # 电磁区域四个点坐标
        self.mid = []  # 中点坐标
        self.link = []  # 用区域索引相关点
        self.center = []  # 圆弧的圆心
        self.work = work  # 工作区域
        self.theta = theta  # 扇形夹角
        self.radius = []  # 圆弧的半径
        self.limits = []  # 生成随机的范围
        self.target_point = [{'area': target_area0}, {'area': target_area1}]  # 两辆车的信息
        self.add_mid()  # 添加中点
        self.add_link()  # 添加与区域相关的点的索引
        self.add_center()  # 添加圆心
        self.add_radius(radius)  # 添加半径
        self.find_limits()  # 添加范围
        self.generate()  # 生成两辆车的位置坐标
        self.find_angle()  # 计算扇形角度

    def add_mid(self):
        x0 = (self.arena[3]['x'] + self.arena[2]['x']) / 2
        y0 = (self.arena[3]['y'] + self.arena[2]['y']) / 2
        self.mid.append({'x': x0, 'y': y0})
        x1 = (self.arena[3]['x'] + self.arena[1]['x']) / 2
        y1 = (self.arena[3]['y'] + self.arena[1]['y']) / 2
        self.mid.append({'x': x1, 'y': y1})
        x2 = (self.arena[3]['x'] + self.arena[0]['x']) / 2
        y2 = (self.arena[3]['y'] + self.arena[0]['y']) / 2
        self.mid.append({'x': x2, 'y': y2})
        x3 = (self.arena[1]['x'] + self.arena[0]['x']) / 2
        y3 = (self.arena[1]['y'] + self.arena[0]['y']) / 2
        self.mid.append({'x': x3, 'y': y3})

    def add_link(self):
        self.link.append({'edge1': 2, 'edge2': 3, 'mid': 0, 'far': 0})  # 0区域的边缘点分别为2和3，2和3的中点为0，最远点为0
        self.link.append({'edge1': 2, 'edge2': 3, 'mid': 0, 'far': 1})
        self.link.append({'edge1': 2, 'edge2': 0, 'mid': 1, 'far': 1})
        self.link.append({'edge1': 0, 'edge2': 3, 'mid': 2, 'far': 1})
        self.link.append({'edge1': 0, 'edge2': 3, 'mid': 2, 'far': 2})
        self.link.append({'edge1': 1, 'edge2': 3, 'mid': 1, 'far': 2})
        self.link.append({'edge1': 0, 'edge2': 1, 'mid': 3, 'far': 2})
        self.link.append({'edge1': 0, 'edge2': 1, 'mid': 3, 'far': 3})

    def add_center(self):
        def find_point(point1, point2, point_mid, initial_x, initial_y):
            def equations(x):
                point = {'x': x[0], 'y': x[1]}
                f1 = EMC.distance(point, point_mid) - EMC.distance(point1, point2) / (
                        2 * math.tan(math.radians(self.theta)))  # 要转换成弧度制
                f2 = (x[0] - point_mid['x']) * (point1['x'] - point2['x']) + (x[1] - point_mid['y']) * (
                        point1['y'] - point2['y'])  # 向量点乘为0则垂直
                return [f1, f2]

            initial_guess = np.array([initial_x, initial_y])  # 解方程的初值，要求为数组形式
            result = root(equations, initial_guess)
            return [result.x[0], result.x[1]]

        for i in range(0, 3):
            [x0, y0] = find_point(self.arena[self.link[i]['edge1']], self.arena[self.link[i]['edge2']],
                                  self.mid[self.link[i]['mid']], self.mid[self.link[i]['mid']]['x'],
                                  self.mid[self.link[i]['mid']]['y'] + 100)  # 0 1 2区域中内侧弧线的圆心在中点的上方，所以y+100
            self.center.append(
                {'outside': {'x': self.arena[self.link[i]['far']]['x'], 'y': self.arena[self.link[i]['far']]['y']},
                 'inside': {'x': x0, 'y': y0}})
        for i in range(3, 5):
            [x0, y0] = find_point(self.arena[self.link[i]['edge1']], self.arena[self.link[i]['edge2']],
                                  self.mid[self.link[i]['mid']], self.mid[self.link[i]['mid']]['x'] - 100,
                                  self.mid[self.link[i]['mid']]['y'])  # 3 4区域中内侧弧线的圆心在中点的左方，所以x-100
            self.center.append(
                {'outside': {'x': self.arena[self.link[i]['far']]['x'], 'y': self.arena[self.link[i]['far']]['y']},
                 'inside': {'x': x0, 'y': y0}})
        for i in range(5, 8):
            [x0, y0] = find_point(self.arena[self.link[i]['edge1']], self.arena[self.link[i]['edge2']],
                                  self.mid[self.link[i]['mid']], self.mid[self.link[i]['mid']]['x'],
                                  self.mid[self.link[i]['mid']]['y'] - 100)  # 5 6 7区域中内侧弧线的圆心在中点的下方，所以y-100
            self.center.append(
                {'outside': {'x': self.arena[self.link[i]['far']]['x'], 'y': self.arena[self.link[i]['far']]['y']},
                 'inside': {'x': x0, 'y': y0}})

    def add_radius(self, radius):
        def cal_radius(point1, point2):
            d = EMC.distance(point1, point2) / (2 * math.sin(math.radians(self.theta)))  # 根据几何关系得出的方程
            return d

        for i in range(0, 8):
            self.radius.append({'outside': radius, 'inside': cal_radius(self.arena[self.link[i]['edge1']],
                                                                        self.arena[self.link[i]['edge2']])})

    def find_limits(self):
        def cal_limits(center, radius, a, b, c, initial_x, initial_y):  # 根据两直线相交求出约束点位，a b c为直线ax+by+c的参数
            def equations(x):
                point = {'x': x[0], 'y': x[1]}
                f1 = EMC.distance(point, center) - radius
                f2 = a * x[0] + b * x[1] + c
                return [f1, f2]

            initial_guess = np.array([initial_x, initial_y])
            result = root(equations, initial_guess)
            return {'x': result.x[0], 'y': result.x[1]}

        to = cal_limits(self.center[0]['outside'], self.radius[0]['outside'], 1, 0, -self.mid[0]['x'],
                        self.mid[0]['x'],
                        self.mid[0]['y'])  # top
        lt = cal_limits(self.center[2]['outside'], self.radius[2]['outside'], 1, 0, -self.arena[3]['x'],
                        self.arena[3]['x'],
                        self.arena[3]['y'])  # left top
        le = cal_limits(self.center[3]['outside'], self.radius[3]['outside'], 0, 1, -self.mid[2]['y'],
                        self.mid[2]['x'],
                        self.mid[2]['y'])  # left
        lb = cal_limits(self.center[5]['outside'], self.radius[5]['outside'], 1, 0, -self.arena[0]['x'],
                        self.arena[0]['x'],
                        self.arena[0]['y'])  # left bottom
        bo = cal_limits(self.center[7]['outside'], self.radius[7]['outside'], 1, 0, -self.mid[3]['x'],
                        self.mid[3]['x'],
                        self.mid[3]['y'])  # bottom
        self.limits.append(
            {'x_max': self.arena[2]['x'], 'y_max': to['y'], 'x_min': self.mid[0]['x'], 'y_min': self.arena[2]['y']})
        self.limits.append(
            {'x_max': self.mid[0]['x'], 'y_max': to['y'], 'x_min': self.arena[3]['x'], 'y_min': self.arena[2]['y']})
        self.limits.append(
            {'x_max': self.arena[3]['x'], 'y_max': lt['y'], 'x_min': le['x'], 'y_min': self.arena[3]['y']})
        self.limits.append(
            {'x_max': self.arena[3]['x'], 'y_max': self.arena[3]['y'], 'x_min': le['x'], 'y_min': self.mid[2]['y']})
        self.limits.append(
            {'x_max': self.arena[3]['x'], 'y_max': self.mid[2]['y'], 'x_min': le['x'], 'y_min': self.arena[0]['y']})
        self.limits.append(
            {'x_max': self.arena[3]['x'], 'y_max': self.arena[0]['y'], 'x_min': le['x'], 'y_min': lb['y']})
        self.limits.append(
            {'x_max': self.mid[3]['x'], 'y_max': self.arena[0]['y'], 'x_min': self.arena[0]['x'], 'y_min': bo['y']})
        self.limits.append(
            {'x_max': self.arena[1]['x'], 'y_max': self.arena[0]['y'], 'x_min': self.mid[3]['x'], 'y_min': bo['y']})

    def generate(self):
        def test(area):
            result1 = 0
            result2 = 0
            point = {'x': x, 'y': y}
            if self.work['start_line'][0]['x'] < x < self.work['end_line'][0]['x'] and \
                    self.work['start_line'][0]['y'] < y < self.work['start_line'][1]['y']:
                result1 = 1  # 满足工作区域的条件，为1
            for j in range(0, 8):
                if area == j:
                    if EMC.distance(point, self.center[j]['outside']) < self.radius[j]['outside'] and EMC.distance(
                            point, self.center[j]['inside']) > \
                            self.radius[j]['inside']:
                        result2 = 1  # 满足弧线的约束，为1
            return result1 * result2  # 同时满足两个条件，result都为1，相乘才为1，有一个不满足就为0

        for i in range(0, 2):
            while 1:
                x = random.uniform(self.limits[self.target_point[i]['area']]['x_min'],
                                   self.limits[self.target_point[i]['area']]['x_max'])  # 产生随机数
                y = random.uniform(self.limits[self.target_point[i]['area']]['y_min'],
                                   self.limits[self.target_point[i]['area']]['y_max'])
                if test(self.target_point[i]['area']):
                    self.target_point[i]['x'] = x
                    self.target_point[i]['y'] = y
                    break

    def find_angle(self):
        for i in range(0, 2):
            angle1 = math.atan(
                (self.arena[self.link[self.target_point[i]['area']]['edge1']]['y'] - self.target_point[i]['y']) / (
                        self.arena[self.link[self.target_point[i]['area']]['edge1']]['x'] - self.target_point[i]['x']))
            angle2 = math.atan(
                (self.arena[self.link[self.target_point[i]['area']]['edge2']]['y'] - self.target_point[i]['y']) / (
                        self.arena[self.link[self.target_point[i]['area']]['edge2']]['x'] - self.target_point[i]['x']))
            angle1_deg = math.degrees(angle1)  # 转化为角度制
            angle2_deg = math.degrees(angle2)
            if self.target_point[i]['area'] in [0, 1]:
                alpha1_deg = angle1_deg - 180 if angle1_deg > 0 else angle1_deg
                alpha2_deg = angle2_deg - 180 if angle2_deg > 0 else angle2_deg
            elif self.target_point[i]['area'] in [6, 7]:
                alpha1_deg = 180 + angle1_deg if angle1_deg < 0 else angle1_deg
                alpha2_deg = 180 + angle2_deg if angle2_deg < 0 else angle2_deg
            else:
                alpha1_deg = angle1_deg
                alpha2_deg = angle2_deg
            alpha = -(alpha1_deg + alpha2_deg) / 2
            differ = (self.theta - abs(alpha1_deg - alpha2_deg)) / 2
            alpha_max = alpha + differ
            alpha_min = alpha - differ
            self.target_point[i]['angle'] = alpha
            self.target_point[i]['angle_range'] = [alpha_min, alpha_max]

    @staticmethod
    def distance(point1, point2):
        return ((point1['x'] - point2['x']) ** 2 + (point1['y'] - point2['y']) ** 2) ** 0.5


def change_line(_node):  # 电磁干扰车要换道躲开前方的打击车，然后再向压制点前进
    _node.apply_vehicle_control(1, 0.5, 0, False, 0)
    while True:
        yaw, pitch, roll, heading, sw_time = _node.get_attitude()
        if yaw > 50:
            _node.apply_vehicle_control(1, -0.5, 0, False, 0)
        if yaw < 0:
            _drive_brake(_node)
            time.sleep(2)
            break


def change_line_fight(_node):  # 电磁干扰车要换道躲开前方的打击车，然后再向压制点前进
    _node.apply_vehicle_control(1, 0.5, 0, False, 0)
    time.sleep(2.7)
    _node.apply_vehicle_control(1, -0.5, 0, False, 0)
    time.sleep(1)
    brakedown(_node)
    time.sleep(2)


def to_supress_point(_emc_node, _emc_path, _ang):
    veh_move(_emc_node, _emc_path)

    # 让道
    _emc_node.apply_vehicle_control(1, 1, 0, False, 0)
    time.sleep(15)
    _emc_node.apply_vehicle_control(0, 0, 0, True, 0)
    time.sleep(5)

    _yaw, _, _, _, _ = _emc_node.get_attitude()
    _emc_ang = _ang - _yaw
    _emc_node.set_emc_angle(_emc_ang)
    print(_emc_ang)
    time.sleep(2)
    _emc_node.control_emc(True)


def execute_emc(_emcs, _emc_dest, _emc_ang):
    emc_path = []
    emc_start_ue = [_emcs[0].get_location()[0], _emcs[0].get_location()[1]]
    emc_start_pixel = ue_to_pixel(emc_start_ue)
    emc_dest_pixel = ue_to_pixel(_emc_dest[0])
    emc_path_0 = find_path(map_no_mine, emc_start_pixel, emc_dest_pixel, 0)  # map传进来？？
    emc_path.append(emc_path_0)
    emc_start_ue = [_emcs[1].get_location()[0], _emcs[1].get_location()[1]]
    emc_start_pixel = ue_to_pixel(emc_start_ue)
    emc_dest_pixel = ue_to_pixel(_emc_dest[1])
    emc_path_1 = find_path(map_no_mine, emc_start_pixel, emc_dest_pixel, 0)  # map传进来？？
    emc_path.append(emc_path_1)

    # to_supress_point(_emcs[0], _emc_dest[0], _emc_ang[0])  # 测试用，最后记得删除
    # time.sleep(3)
    # to_supress_point(_emcs[1], _emc_dest[1], _emc_ang[1])  # 测试用，最后记得删除

    threads = []  # thread名称和飞机的重合，是否有风险？
    for i, _emc in enumerate(_emcs):
        if i != 0:
            time.sleep(10)  # 让各个车之间错开60秒
        # 先换道
        thread = threading.Thread(target=to_supress_point, args=(_emc, emc_path[i], _emc_ang[i]))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


'''——————————————————————————————————————————————————'''

''' ——————————————第6阶段协同侦打需要用到的函数———————————— '''


def img_edit_arena(img_RGB, list):  # 输入原始彩色图片，矩形四个角点（从左下开始逆时针），列表格式
    x_min = list[0]['x']
    y_min = list[0]['y']
    x_max = list[2]['x']
    y_max = list[2]['y']
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            img_RGB.putpixel((x, y), (0, 0, 0))
    return img_RGB


def in_striking_range(_hitter_location, _target_location):
    range = 65
    if ((_target_location[0] - range) < _hitter_location[0] < (_target_location[0] + range) and
            (_target_location[1] - range) < _hitter_location[1] < (_target_location[1] + range)):
        return True
    else:
        return False


def hit_enemy(_hitter, _weapon, _target_list):
    for _index, _target in enumerate(_target_list):
        if _weapon.get_weapon_ammo()[1].y or _weapon.get_weapon_ammo()[1].z:
            _hitter_cur_location_ue = [_hitter.get_location()[0], _hitter.get_location()[1]]
            _hitter_cur_location_pixel = ue_to_pixel(_hitter_cur_location_ue)
            _target_location_ue = [_target["Position"]["x"], _target["Position"]["y"]]
            _target_location_pixel = ue_to_pixel(_target_location_ue)
            _target_location_fine_pixel = find_element(map_array, _target_location_pixel)
            # 先找到前往打击目标附近的路径
            # to_target_path = find_path(map_img, _hitter_cur_location_pixel, _target_location_fine_pixel, 0)
            to_target_path = find_path(map_no_emc, _hitter_cur_location_pixel, _target_location_fine_pixel, 0)
            print(to_target_path[:-2])
            to_target_path_reduce = to_target_path[:-2]

            last_state = 'start'
            last_mode = 'z'
            last_toward = 'z'
            last_target_toward = 'z'
            last = [last_state, last_mode, last_toward, last_target_toward]
            # 把车开过去
            for index, _path_point in enumerate(to_target_path_reduce):  # 不要最后2个轨迹点
                _path_point_pixel = [_path_point["x"], _path_point["y"]]
                _path_point_ue = pixel_to_ue(_path_point_pixel)
                if index < len(to_target_path_reduce) - 1:
                    path_point_next_pixel = [to_target_path[index + 1]["x"], to_target_path[index + 1]["y"]]
                else:  # 最后一个点时
                    path_point_next_pixel = [_path_point["x"], _path_point["y"]]
                path_point_next_ue = pixel_to_ue(path_point_next_pixel)  # 当前要前往的坐标点的下一个坐标点
                _cur_location_ue = [_hitter.get_location()[0], _hitter.get_location()[1]]
                if _weapon.get_weapon_ammo()[1].y or _weapon.get_weapon_ammo()[1].z:
                    _, _, _, head, _ = _hitter.get_attitude()
                    print(head)
                    last = list(veh_go_to(_hitter, _cur_location_ue, _path_point_ue, path_point_next_ue,
                                          last))  # 如果没到打击范围之内，就开过去
                else:
                    break

                target_res = _hitter.detect_situation()
                target_res_list = list(target_res)
                dynamic_target_info = json.loads(target_res_list[2])
                for each_target_info in dynamic_target_info:
                    if (each_target_info["Type"] == "strike" and each_target_info["HealthPoint"] and
                            in_striking_range([_hitter.get_location()[0], _hitter.get_location()[1]],
                                              [round(each_target_info["Position"]["X"] / 100, 1),
                                               round(each_target_info["Position"]["Y"] / 100, 1)])):
                        print("find target!", each_target_info["Type_Name"])
                        if each_target_info["Type_Name"] == "Soldier":  # 对于士兵，用5发子弹
                            _hitter.apply_vehicle_control(0, 0, 0, True, 0)
                            time.sleep(2)
                            print("use gun")
                            _weapon.set_weapon_status(5, 0, round(each_target_info["Position"]["X"] / 100, 1),
                                                      round(each_target_info["Position"]["Y"] / 100, 1),
                                                      round(each_target_info["Position"]["Z"] / 100, 1))
                            time.sleep(2)
                        elif each_target_info["Type_Name"] == "Barricade_BP":  # 对于工事，用3发40炮
                            if _weapon.get_weapon_ammo()[1].y:
                                _hitter.apply_vehicle_control(0, 0, 0, True, 0)
                                time.sleep(2)
                                print("use 40")
                                _weapon.set_weapon_status(3, 1, round(each_target_info["Position"]["X"] / 100, 1),
                                                          round(each_target_info["Position"]["Y"] / 100, 1),
                                                          round(each_target_info["Position"]["Z"] / 100, 1) + 1)
                                print(each_target_info["HealthPoint"])
                                time.sleep(2)
                        elif each_target_info["Type_Name"] == "Building":  # 对于建筑物，用3发40炮
                            if _weapon.get_weapon_ammo()[1].y:
                                _hitter.apply_vehicle_control(0, 0, 0, True, 0)
                                time.sleep(2)
                                print("use 40")
                                _weapon.set_weapon_status(3, 1, round(each_target_info["Position"]["X"] / 100, 1),
                                                          round(each_target_info["Position"]["Y"] / 100, 1),
                                                          round(each_target_info["Position"]["Z"] / 100, 1) + 1.5)
                                print(each_target_info["HealthPoint"])
                                time.sleep(2)
                        else:  # 对于车辆，有反坦克就给2发反坦克炮弹，如果没有就用4发40炮
                            if _weapon.get_weapon_ammo()[1].z:
                                _hitter.apply_vehicle_control(0, 0, 0, True, 0)
                                time.sleep(2)
                                print("use mission")
                                _weapon.set_weapon_status(2, 2, round(each_target_info["Position"]["X"] / 100, 1),
                                                          round(each_target_info["Position"]["Y"] / 100, 1),
                                                          round(each_target_info["Position"]["Z"] / 100, 1) + 1)
                                time.sleep(2)
                            else:
                                if _weapon.get_weapon_ammo()[1].y:
                                    _hitter.apply_vehicle_control(0, 0, 0, True, 0)
                                    time.sleep(2)
                                    print("use 40")
                                    _weapon.set_weapon_status(4, 1, round(each_target_info["Position"]["X"] / 100, 1),
                                                              round(each_target_info["Position"]["Y"] / 100, 1),
                                                              round(each_target_info["Position"]["Z"] / 100, 1) + 1)
                                    time.sleep(2)
                        break
            _hitter.apply_vehicle_control(0, 0, 0, True, 0)
            time.sleep(2)
        else:
            break


def fight_plan(_hit_nodes, _weapon_nodes, _strike_info_list):
    hit_enemy(_hit_nodes[0], _weapon_nodes[0], _strike_info_list[0])
    change_line_fight(_hit_nodes[0])
    hit_enemy(_hit_nodes[1], _weapon_nodes[1], _strike_info_list[1])
    change_line_fight(_hit_nodes[1])
    hit_enemy(_hit_nodes[2], _weapon_nodes[2], _strike_info_list[2])


'''——————————————————————————————————————————————————'''


# 主函数
if __name__ == "__main__":
    # 固定接口
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--address", help="指定服务器IP", type=str,
                        default='127.0.0.1')  # 127.0.0.1 10.62.150.188
    parser.add_argument("-p", "--port", help="指定服务器Port", type=int, default=2000)
    parser.add_argument("-n", "--number", help="指定飞机数量", type=int, default=10)
    parser.add_argument('-s', '--subject', help="科目", type=int, default=1)
    args = parser.parse_args()
    ip = args.address.strip()
    port = args.port

    try:
        # 注册节点
        swarm = SwarmaeClass()
        client = swarm.create_client(args.address.strip(), args.port)
        timestamp, game, code = client.get_game()

        # 节点定义
        sw_code1, sw_node1 = swarm.create_node(swarm.ae_client, "四轮车", 1)
        sw_code2, sw_node2 = swarm.create_node(swarm.ae_client, "四轮车", 2)
        sw_code3, sw_node3 = swarm.create_node(swarm.ae_client, "四轮车", 3)
        sw_code4, sw_node4 = swarm.create_node(swarm.ae_client, "四轮车", 4)
        sw_code5, sw_node5 = swarm.create_node(swarm.ae_client, "四轮车", 5)
        sw_code6, sw_node6 = swarm.create_node(swarm.ae_client, "四轮车", 6)
        sw_code7, sw_node7 = swarm.create_node(swarm.ae_client, "四轮车", 7)
        sw_code8, sw_node8 = swarm.create_node(swarm.ae_client, "四旋翼", 8)
        sw_code9, sw_node9 = swarm.create_node(swarm.ae_client, "四旋翼", 9)
        sw_code10, sw_node10 = swarm.create_node(swarm.ae_client, "四旋翼", 10)
        sw = [sw_node1, sw_node2, sw_node3, sw_node4, sw_node5, sw_node6, sw_node7, sw_node8, sw_node9, sw_node10]

        # (0) 动态地图
        map_thread = threading.Thread(target=task_vis_map.core, args=(game, sw))
        map_thread.start()

        # #（1）无人机群侦察行动规划
        # rc_time, rc_code = game.stage_start("reconnaissance_start")
        # if rc_code == 200:
        #     print("reconnaissance_start")
        #     # print(rc_time)
        #     flight_routes = drone_plan()  # 要让drone_plan()返回json格式的路径点
        #     print(flight_routes)
        #     rc_submit_time, rc_submit_code = game.submit_plane_route(flight_routes)
        #     if rc_submit_code == 200:
        #         print("reconnaissance_submit")
        #         # print(rc_submit_time)
        #         rc_complete_time, rc_complete_code = game.stage_complete("reconnaissance_end")
        #         if rc_complete_code == 200:
        #             print("reconnaissance_end")
        #             # print(rc_complete_time)
        #
        # # （2）无人机群空中侦察
        # va_rc_time, va_rc_code = game.stage_start("vau_reconnaissance_start")
        # if va_rc_code == 200:
        #     print("vau_reconnaissance_start")
        #     # print(flight_routes)
        #     vau_routes = json.loads(flight_routes)
        #     drone_start_points = [
        #         list(sw_node8.get_location()[:3]),
        #         list(sw_node9.get_location()[:3]),
        #         list(sw_node10.get_location()[:3])
        #     ]
        #     print(drone_start_points)
        #     vanguard(vau_routes)  # 执行空中侦察、不断查询目标信息、目标位置信息存在列表target_info
        #     va_rc_complete_time, va_rc_complete_code = game.stage_complete("vau_reconnaissance_end")
        #     if va_rc_complete_code == 200:
        #         print("vau_reconnaissance_end")

        # #（3）局部路径规划
        # va_rc_complete_time, va_rc_complete_code = game.stage_complete("vau_reconnaissance_end")  # 测试用，不然调不出来路网地图，最后记得删除
        # plan_start_time, plan_start_code = game.stage_start("plan_start")
        # if plan_start_code == 200:
        #     print("【3阶段】集群路径规划任务开始")
        #     info = list(game.get_task_info())  # 将元组转换成列表 这两句话要优化，写一遍就行
        #     whole_arena_data = json.loads(info[0])  # 把区域信息提取出来，并解析为有效的字典
        #     start_line = whole_arena_data["subject_3"]["start_line"]
        #     end_line = whole_arena_data["subject_3"]["end_line"]
        #
        #     road_net_info = game.get_road_network()
        #     map_img = road_net_info[0]
        #     SCALE = road_net_info[1]
        #     OFFSET_X = road_net_info[2][0]
        #     OFFSET_Y = road_net_info[2][1]
        #
        #     plan_and_follow(sw, map_img, start_line, end_line)
        #
        #     print("打击车换道为之后科目做准备")
        #     change_line_fight(sw_node2)
        #     change_line_fight(sw_node1)
        #     change_line_fight(sw_node3)
        #     plan_complete_time, plan_complete_code = game.stage_complete("plan_end")
        #     if plan_complete_code == 200:
        #         print("【3阶段】集群路径规划任务结束")

        #  （4）道路开辟

        # 测试用，不然调不出来路网地图，最后记得删除
        va_rc_complete_time, va_rc_complete_code = game.stage_complete("vau_reconnaissance_end")
        info = list(game.get_task_info())  # 将元组转换成列表 这两句话要优化，写一遍就行
        whole_arena_data = json.loads(info[0])  # 把区域信息提取出来，并解析为有效的字典
        start_line = whole_arena_data["subject_3"]["start_line"]
        end_line = whole_arena_data["subject_3"]["end_line"]
        road_net_info = game.get_road_network()
        map_img = road_net_info[0]
        SCALE = road_net_info[1]
        OFFSET_X = road_net_info[2][0]
        OFFSET_Y = road_net_info[2][1]

        mine_start_time, mine_start_code = game.stage_start("minesweeper_start")
        if mine_start_code == 200:
            print("【4阶段】扫雷任务开始")
            mine_arena_ue = whole_arena_data["mine_arena"]
            mine_arena_pixel = [{"x": ue_to_pixel([mine_arena_ue[0]["x"], mine_arena_ue[0]["y"]])[0],
                                 "y": ue_to_pixel([mine_arena_ue[0]["x"], mine_arena_ue[0]["y"]])[1]},
                                {"x": ue_to_pixel([mine_arena_ue[1]["x"], mine_arena_ue[1]["y"]])[0],
                                 "y": ue_to_pixel([mine_arena_ue[1]["x"], mine_arena_ue[1]["y"]])[1]},
                                {"x": ue_to_pixel([mine_arena_ue[2]["x"], mine_arena_ue[2]["y"]])[0],
                                 "y": ue_to_pixel([mine_arena_ue[2]["x"], mine_arena_ue[2]["y"]])[1]},
                                {"x": ue_to_pixel([mine_arena_ue[3]["x"], mine_arena_ue[3]["y"]])[0],
                                 "y": ue_to_pixel([mine_arena_ue[3]["x"], mine_arena_ue[3]["y"]])[1]}]

            mine_lb_point_ue = whole_arena_data["mine_arena"][0]
            mine_rt_point_ue = whole_arena_data["mine_arena"][2]
            mine_lb_point_pixel = ue_to_pixel([mine_lb_point_ue["x"], mine_lb_point_ue["y"]])
            mine_rt_point_pixel = ue_to_pixel([mine_rt_point_ue["x"], mine_rt_point_ue["y"]])
            # 把路网图片中的横纵道路提取出来
            hor_roads, ver_roads = get_roads_in_region(map_img, mine_lb_point_pixel, mine_rt_point_pixel)
            # print("水平道路：", hor_roads)
            # print("竖直道路：", ver_roads)
            # 给两辆车分配横纵道路的扫雷方案
            sweepers = [sw_node6, sw_node7]
            plan_sweep_route(sweepers, hor_roads, ver_roads)
            mine_complete_time, mine_complete_code = game.stage_complete("minesweeper_end")
            if mine_complete_code == 200:
                print("【4阶段】扫雷任务结束")

        # （5）电磁压制
        # va_rc_complete_time, va_rc_complete_code = game.stage_complete("vau_reconnaissance_end")  # 测试用，不然调不出来路网地图，最后记得删除
        emc_start_time, emc_start_code = game.stage_start("emi_start")
        if emc_start_code == 200:
            print("emc_start")

            # road_net_info = game.get_road_network()
            # map_img = road_net_info[0]

            map_array = list(tga_to_array(map_img))  # 得留着

            # SCALE = road_net_info[1]
            # OFFSET_X = road_net_info[2][0]
            # OFFSET_Y = road_net_info[2][1]  # 要优化写一遍就行
            # info = list(game.get_task_info())  # 将元组转换成列表 这两句话要优化，写一遍就行！！！！！
            # whole_arena_data = json.loads(info[0])  # 把区域信息提取出来，并解析为有效的字典
            # mine_arena_ue = whole_arena_data["mine_arena"]
            # mine_arena_pixel = [{"x": ue_to_pixel([mine_arena_ue[0]["x"], mine_arena_ue[0]["y"]])[0],
            #                         "y": ue_to_pixel([mine_arena_ue[0]["x"], mine_arena_ue[0]["y"]])[1]},
            #                        {"x": ue_to_pixel([mine_arena_ue[1]["x"], mine_arena_ue[1]["y"]])[0],
            #                         "y": ue_to_pixel([mine_arena_ue[1]["x"], mine_arena_ue[1]["y"]])[1]},
            #                        {"x": ue_to_pixel([mine_arena_ue[2]["x"], mine_arena_ue[2]["y"]])[0],
            #                         "y": ue_to_pixel([mine_arena_ue[2]["x"], mine_arena_ue[2]["y"]])[1]},
            #                        {"x": ue_to_pixel([mine_arena_ue[3]["x"], mine_arena_ue[3]["y"]])[0],
            #                         "y": ue_to_pixel([mine_arena_ue[3]["x"], mine_arena_ue[3]["y"]])[1]}]

            emc_arena_ue = whole_arena_data["emc_arena"]
            emc_arena_pixel = [{"x": ue_to_pixel([emc_arena_ue[0]["x"], emc_arena_ue[0]["y"]])[0],
                                "y": ue_to_pixel([emc_arena_ue[0]["x"], emc_arena_ue[0]["y"]])[1]},
                               {"x": ue_to_pixel([emc_arena_ue[1]["x"], emc_arena_ue[1]["y"]])[0],
                                "y": ue_to_pixel([emc_arena_ue[1]["x"], emc_arena_ue[1]["y"]])[1]},
                               {"x": ue_to_pixel([emc_arena_ue[2]["x"], emc_arena_ue[2]["y"]])[0],
                                "y": ue_to_pixel([emc_arena_ue[2]["x"], emc_arena_ue[2]["y"]])[1]},
                               {"x": ue_to_pixel([emc_arena_ue[3]["x"], emc_arena_ue[3]["y"]])[0],
                                "y": ue_to_pixel([emc_arena_ue[3]["x"], emc_arena_ue[3]["y"]])[1]}]
            map_no_mine = img_edit_arena(map_img, mine_arena_pixel)
            subject_5__arena_ue = whole_arena_data["subject_5"]
            theta = 40
            radius = 1000
            # 保证找到的坐标点必须是可通行的区域
            emc1_pixel_value = 0
            emc2_pixel_value = 0
            find_range = 7
            while (emc1_pixel_value != 255
                   or emc2_pixel_value != 255
                   or map_array[emc1_pixel[1] + find_range][emc1_pixel[0]] != 255
                   or map_array[emc1_pixel[1]][emc1_pixel[0] + find_range] != 255
                   or map_array[emc1_pixel[1] - find_range][emc1_pixel[0]] != 255
                   or map_array[emc1_pixel[1]][emc1_pixel[0] - find_range] != 255
                   or map_array[emc2_pixel[1] + find_range][emc2_pixel[0]] != 255
                   or map_array[emc2_pixel[1]][emc2_pixel[0] + find_range] != 255
                   or map_array[emc2_pixel[1] - find_range][emc2_pixel[0]] != 255
                   or map_array[emc2_pixel[1]][emc2_pixel[0] - find_range] != 255):
                emc = EMC(emc_arena_ue, subject_5__arena_ue, theta, radius, 3, 4)
                emc1_pixel = ue_to_pixel([emc.target_point[0]['x'], emc.target_point[0]['y']])
                emc2_pixel = ue_to_pixel([emc.target_point[1]['x'], emc.target_point[1]['y']])
                emc1_pixel_value = map_array[emc1_pixel[1]][emc1_pixel[0]]
                emc2_pixel_value = map_array[emc2_pixel[1]][emc2_pixel[0]]
            emc1_dest = [round(emc.target_point[0]['x'], 1), round(emc.target_point[0]['y'], 1)]
            emc1_ang = -round((emc.target_point[0]['angle_range'][0] + emc.target_point[0]['angle_range'][1]) / 2, 1)
            emc2_dest = [round(emc.target_point[1]['x'], 1), round(emc.target_point[1]['y'], 1)]
            emc2_ang = -round((emc.target_point[1]['angle_range'][0] + emc.target_point[1]['angle_range'][1]) / 2, 1)
            print(map_array[emc1_pixel[1]][emc1_pixel[0]])
            print(map_array[emc2_pixel[1]][emc2_pixel[0]])
            print('第一辆车的坐标', emc1_dest)
            print('第一辆车角度', emc1_ang)
            print('第二辆车的坐标', emc2_dest)
            print('第二辆车角度', emc2_ang)
            emcs = [sw_node5, sw_node4]
            emc_dest = [emc1_dest, emc2_dest]
            emc_ang = [emc1_ang, emc2_ang]

            # change_line(sw_node2)
            # time.sleep(5)
            # change_line(sw_node1)
            # time.sleep(5)
            # change_line(sw_node3)

            execute_emc(emcs, emc_dest, emc_ang)
            emc_complete_time, emc_complete_code = game.stage_complete("emi_end")
            if emc_complete_code == 200:
                print("emc_end")

        # （6）协同侦打
        # va_rc_complete_time, va_rc_complete_code = game.stage_complete("vau_reconnaissance_end")  # 测试用，不然调不出来路网地图，最后记得删除
        hit_start_time, hit_start_code = game.stage_start("hit_start")
        if hit_start_code == 200:
            print("hit_start")

            # road_net_info = game.get_road_network()
            # map_img = road_net_info[0]
            # map_array = list(tga_to_array(map_img))  # 得留着
            # SCALE = road_net_info[1]
            # OFFSET_X = road_net_info[2][0]
            # OFFSET_Y = road_net_info[2][1]  # 要优化写一遍就行
            # info = list(game.get_task_info())  # 将元组转换成列表 这两句话要优化，写一遍就行！！！！！
            # whole_arena_data = json.loads(info[0])  # 把区域信息提取出来，并解析为有效的字典
            # emc_arena_ue = whole_arena_data["emc_arena"]  # 第五部分有了到时候去掉
            # emc_arena_pixel = [{"x": ue_to_pixel([emc_arena_ue[0]["x"], emc_arena_ue[0]["y"]])[0],
            #                     "y": ue_to_pixel([emc_arena_ue[0]["x"], emc_arena_ue[0]["y"]])[1]},
            #                    {"x": ue_to_pixel([emc_arena_ue[1]["x"], emc_arena_ue[1]["y"]])[0],
            #                     "y": ue_to_pixel([emc_arena_ue[1]["x"], emc_arena_ue[1]["y"]])[1]},
            #                    {"x": ue_to_pixel([emc_arena_ue[2]["x"], emc_arena_ue[2]["y"]])[0],
            #                     "y": ue_to_pixel([emc_arena_ue[2]["x"], emc_arena_ue[2]["y"]])[1]},
            #                    {"x": ue_to_pixel([emc_arena_ue[3]["x"], emc_arena_ue[3]["y"]])[0],
            #                     "y": ue_to_pixel([emc_arena_ue[3]["x"], emc_arena_ue[3]["y"]])[1]}]

            map_no_emc = img_edit_arena(map_no_mine, emc_arena_pixel)
            _, weapon2, _, _ = sw_node2.get_weapon()
            _, weapon1, _, _ = sw_node1.get_weapon()
            _, weapon3, _, _ = sw_node3.get_weapon()
            weapon_nodes = [weapon2, weapon1, weapon3]  # 按照出发顺序排序
            hit_nodes = [sw_node2, sw_node1, sw_node3]
            list1 = []
            list2 = []
            list3 = []
            strike_info_list = []
            for i, item in enumerate(strike_info):
                if i % 3 == 0:
                    list1.append(item)
                elif i % 3 == 1:
                    list2.append(item)
                else:
                    list3.append(item)
            strike_info_list.append(list1)
            strike_info_list.append(list2)
            strike_info_list.append(list3)
            fight_plan(hit_nodes, weapon_nodes, strike_info_list)
            hit_complete_time, hit_complete_code = game.stage_complete("hit_end")
            if hit_complete_code == 200:
                print("hit_end")

        # time.sleep(10)
    except KeyboardInterrupt:
        pass
