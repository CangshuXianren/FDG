import numpy as np
import json
import math
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class GameMapUpdater:
    def __init__(self, game, sw):
        self.game = game
        self.sw = sw

        self.fig = plt.figure(figsize=(10, 5))
        self.ax1 = self.fig.add_subplot()
        self.ani = None

        # 注释文本初始化
        self.fig.text(0.5, 0.05, "arena:   emc=blue,mine=red,nofly=purple\n"
                                 "node:    fight=red,emc=yellow,mine=blue,uav=pink",
                      ha='center', va='center', fontsize=12,
                      transform=self.fig.transFigure)

        # 调整布局以适应注释
        self.fig.tight_layout(rect=[0, 0.1, 1, 1])  # 为底部留出空间

        # 初始化地图边界的图形对象
        self.map_boundaries = []  # 存储所有边界图形对象的列表
        # 初始化图形对象
        self.init_map_boundaries()

        # 初始化动态元素的艺术家列表
        self.dynamic_artists = []
        # 初始化动态元素的缓存数据
        self.node_positions = np.empty((len(sw), 2))
        # 初始化scatter艺术家
        self.initialize_artists()

    def initialize_artists(self):
        pt_size = 25
        # 根据节点类型创建scatter艺术家
        colors = ['red', 'yellow', 'blue', 'pink']
        indices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 10)]
        for i, color in enumerate(colors):
            artist = self.ax1.scatter([], [], c=color, s=pt_size)
            self.dynamic_artists.append(artist)
            # 设置艺术家的初始数据
            artist.set_offsets(self.node_positions[indices[i]])

    def init_map_boundaries(self):
        """初始化地图边界的图形对象"""
        info = list(self.game.get_task_info())
        whole_arena_data = json.loads(info[0])

        # 初始化地图边界图形对象
        self.map_boundaries.append(self.ax1.add_patch(
            patches.Polygon(self.extract_rectangle_coords(whole_arena_data['arena']),
                            closed=True, fill=True, linewidth=1, edgecolor='black',
                            facecolor='black', alpha=0.5)))
        self.map_boundaries.append(self.ax1.add_patch(
            patches.Polygon(self.extract_rectangle_coords(whole_arena_data['emc_arena']),
                            closed=True, fill=True, linewidth=1, edgecolor='black',
                            facecolor='b', alpha=0.5)))
        self.map_boundaries.append(self.ax1.add_patch(
            patches.Polygon(self.extract_rectangle_coords(whole_arena_data['mine_arena']),
                            closed=True, fill=True, linewidth=1, edgecolor='black',
                            facecolor='r', alpha=0.5)))
        self.map_boundaries.append(self.ax1.add_patch(
            patches.Polygon(self.extract_rectangle_coords(whole_arena_data['no_fly_arena'][0]),
                            closed=True, fill=True, linewidth=1, edgecolor='black',
                            facecolor='purple', alpha=0.5)))
        self.map_boundaries.append(self.ax1.add_patch(
            patches.Polygon(self.extract_rectangle_coords(whole_arena_data['no_fly_arena'][1]),
                            closed=True, fill=True, linewidth=1, edgecolor='black',
                            facecolor='purple', alpha=0.5)))
        self.map_boundaries.append(self.ax1.add_patch(
            patches.Polygon(self.extract_rectangle_coords(whole_arena_data['no_fly_arena'][2]),
                            closed=True, fill=True, linewidth=1, edgecolor='black',
                            facecolor='purple', alpha=0.5)))

        line_vertices_x = []
        line_vertices_y = []
        # 绘制线段
        for i in range(0, 6):
            subject_name = 'subject_{}'.format(i + 1)
            line_vertices_x.append(
                [whole_arena_data[subject_name]['start_line'][0]['x'], whole_arena_data[subject_name]['start_line'][1]['x']])
            line_vertices_y.append(
                [whole_arena_data[subject_name]['start_line'][0]['y'], whole_arena_data[subject_name]['start_line'][1]['y']])
            line_vertices_x.append(
                [whole_arena_data[subject_name]['end_line'][0]['x'], whole_arena_data[subject_name]['end_line'][1]['x']])
            line_vertices_y.append(
                [whole_arena_data[subject_name]['end_line'][0]['y'], whole_arena_data[subject_name]['end_line'][1]['y']])

        for i in range(12):
            count = math.floor(i/2)
            subject_name = 'subject_{}'.format(count + 1)
            if count == 0:
                color_tmp = 'red'
            elif count == 1:
                color_tmp = 'orange'
            elif count == 2:
                color_tmp = 'green'
            elif count == 3:
                color_tmp = 'blue'
            elif count == 4:
                color_tmp = 'purple'
            elif count == 5:
                color_tmp = 'yellow'

            if (i % 2) == 0:
                plt.plot(line_vertices_x[i], line_vertices_y[i], color=color_tmp)
                plt.text(line_vertices_x[i][0], line_vertices_y[i][0] + i*50,
                         subject_name + '_S')
            else:
                plt.plot(line_vertices_x[i], line_vertices_y[i], color=color_tmp)
                plt.text(line_vertices_x[i][0], line_vertices_y[i][0] + i*50,
                         subject_name + '_E')


    def extract_rectangle_coords(self, rectangle_data):
        """从给定的矩形数据中提取坐标"""
        return [(point['x'], point['y']) for point in rectangle_data]


    def update(self):
        # 更新节点位置缓存
        for i, node in enumerate(self.sw):
            x, y, _, _ = node.get_location()
            self.node_positions[i] = [x, y]

        # 更新scatter艺术家的数据
        for artist, indices in zip(self.dynamic_artists, [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 10)]):
            artist.set_offsets(self.node_positions[indices])


def core(game, sw):
    print("任务可视化模块启动")
    updater = GameMapUpdater(game, sw)

    def animate(i):
        """更新动画的每一帧"""
        updater.update()
        return updater.dynamic_artists  # 返回所有动态艺术家，以便 FuncAnimation 进行 blitting

    # 设置动画
    updater.ani = animation.FuncAnimation(updater.fig, animate, blit=True, interval=500, init_func=lambda: updater.dynamic_artists)

    # 显示图形
    plt.show()


