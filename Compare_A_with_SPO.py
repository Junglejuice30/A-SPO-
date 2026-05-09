import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import heapq
import random
import time
import math


# 环境生成
def generate_obstacles(num=15, radius_range=(5, 8), robot_radius=2,
                       bounds=(0, 99), start=(5, 5), goal=(90, 90)):
    """
    随机生成圆形障碍物，确保起点和终点不在障碍物内。
    返回原始障碍物列表 [(cx, cy, r), ...] 和膨胀后列表 [(cx, cy, R)].
    """
    obstacles = []
    inflated = []
    count = 0
    while count < num:
        cx = np.random.uniform(bounds[0] + 10, bounds[1] - 10)
        cy = np.random.uniform(bounds[0] + 10, bounds[1] - 10)
        r = np.random.uniform(*radius_range)
        # 检查起点、终点是否在障碍物内部（考虑机器人半径）
        if np.hypot(cx - start[0], cy - start[1]) < r + robot_radius:
            continue
        if np.hypot(cx - goal[0], cy - goal[1]) < r + robot_radius:
            continue
        obstacles.append((cx, cy, r))
        inflated.append((cx, cy, r + robot_radius))
        count += 1
    return obstacles, inflated


# -------------------- 栅格地图与 A* --------------------
def build_grid(inflated_obstacles, res=1, bounds=(0, 99), margin=2):
    """构建二值栅格地图，0-可通行，1-障碍物（膨胀后）"""
    size = int((bounds[1] - bounds[0]) / res) + 1
    grid = np.zeros((size, size), dtype=np.uint8)
    # 边界收缩
    grid[:margin, :] = 1
    grid[-margin:, :] = 1
    grid[:, :margin] = 1
    grid[:, -margin:] = 1
    # 填充膨胀障碍物
    for cx, cy, R in inflated_obstacles:
        x_min = max(0, int(np.floor(cx - R)))
        x_max = min(size - 1, int(np.ceil(cx + R)))
        y_min = max(0, int(np.floor(cy - R)))
        y_max = min(size - 1, int(np.ceil(cy + R)))
        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                if np.hypot(i - cx, j - cy) <= R:
                    grid[i, j] = 1
    return grid


def astar(grid, start, goal):
    """A* 8邻域寻路，返回路径点列表 [(x,y), ...]"""
    rows, cols = grid.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_list:
        _, current = heapq.heappop(open_list)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            # 回溯路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        # 8邻域
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                cost = math.sqrt(dx * dx + dy * dy)
                tentative_g = g_score[current] + cost
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + math.hypot(goal[0] - nx, goal[1] - ny)
                    heapq.heappush(open_list, (f, (nx, ny)))
                    came_from[(nx, ny)] = current
    return None  # 无路径


def path_length(path):
    """计算路径总欧氏长度"""
    if path is None: return float('inf')
    return sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) for i in range(len(path) - 1))


# -------------------- PSO 路径规划 --------------------
def segment_collision(p1, p2, inflated_obstacles, samples=20):
    """检查线段 p1->p2 是否与膨胀障碍物碰撞，返回碰撞点数"""
    violations = 0
    for i in range(samples + 1):
        t = i / samples
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        for cx, cy, R in inflated_obstacles:
            if math.hypot(x - cx, y - cy) < R:
                violations += 1
                break
    return violations


def fitness(particle, start, goal, inflated_obstacles, penalty=1000):
    """PSO 适应度 = 路径长度 + 碰撞惩罚 + 边界惩罚"""
    # 重构路径点
    waypoints = [start]
    for i in range(0, len(particle), 2):
        waypoints.append((particle[i], particle[i + 1]))
    waypoints.append(goal)

    length = 0.0
    violations = 0
    bound_penalty = 0.0
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]
        p2 = waypoints[i + 1]
        length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        # 中间点边界惩罚（起点终点已在安全区内）
        if i > 0 and i < len(waypoints) - 1:
            if not (2 <= p1[0] <= 97 and 2 <= p1[1] <= 97):
                bound_penalty += penalty
        # 线段碰撞
        violations += segment_collision(p1, p2, inflated_obstacles)

    return length + penalty * violations + bound_penalty


def pso_path_planning(start, goal, inflated_obstacles,
                      num_waypoints=6, swarm_size=50, iterations=200):
    """粒子群优化路径中间点，返回最佳路径"""
    dim = 2 * num_waypoints
    bounds = (2, 97)  # 中间点允许范围

    # 初始化粒子群
    positions = np.random.uniform(bounds[0], bounds[1], (swarm_size, dim))
    velocities = np.zeros((swarm_size, dim))

    pbest_pos = positions.copy()
    pbest_val = np.array([fitness(p, start, goal, inflated_obstacles) for p in positions])
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    w_start, w_end = 0.9, 0.4
    c1, c2 = 1.5, 1.5

    for it in range(iterations):
        w = w_start - (w_start - w_end) * it / iterations
        r1 = np.random.rand(swarm_size, dim)
        r2 = np.random.rand(swarm_size, dim)

        velocities = (w * velocities +
                      c1 * r1 * (pbest_pos - positions) +
                      c2 * r2 * (gbest_pos - positions))
        positions += velocities

        # 边界反射
        positions = np.clip(positions, bounds[0], bounds[1])

        # 评价
        for i in range(swarm_size):
            val = fitness(positions[i], start, goal, inflated_obstacles)
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = positions[i].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = positions[i].copy()

    # 重建最优路径
    best_path = [start]
    for i in range(0, dim, 2):
        best_path.append((gbest_pos[i], gbest_pos[i + 1]))
    best_path.append(goal)
    return best_path, gbest_val


# -------------------- 可视化与对比 --------------------
def plot_results(grid, start, goal, obstacles, inflated,
                 astar_path, pso_path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # 原始障碍物与膨胀
    for i, (title, path, color) in enumerate(zip(
            ['A* Path', 'PSO Path'],
            [astar_path, pso_path],
            ['cyan', 'lime'])):
        ax[i].set_title(title)
        ax[i].set_xlim(0, 99);
        ax[i].set_ylim(0, 99)
        ax[i].set_aspect('equal')
        # 网格背景
        extent = [0, 99, 0, 99]
        ax[i].imshow(grid.T, origin='lower', extent=extent, cmap='gray_r', alpha=0.3)
        # 膨胀障碍物
        for cx, cy, R in inflated:
            circle = Circle((cx, cy), R, edgecolor='red', facecolor='red', alpha=0.15)
            ax[i].add_patch(circle)
        # 原始障碍物
        for cx, cy, r in obstacles:
            circle = Circle((cx, cy), r, edgecolor='darkred', facecolor='darkred', alpha=0.5)
            ax[i].add_patch(circle)
        # 起点终点
        ax[i].plot(start[0], start[1], 'go', markersize=8)
        ax[i].plot(goal[0], goal[1], 'ro', markersize=8)
        # 路径
        if path:
            px, py = zip(*path)
            ax[i].plot(px, py, color=color, linewidth=2, marker='.', markersize=3)
        ax[i].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def compute_turn_angle(path):
    """计算路径的总转折角度 (弧度)"""
    if len(path) < 3:
        return 0.0
    total_angle = 0.0
    for i in range(1, len(path) - 1):
        v1 = np.array(path[i]) - np.array(path[i - 1])
        v2 = np.array(path[i + 1]) - np.array(path[i])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue
        cosang = np.dot(v1, v2) / (norm1 * norm2)
        cosang = np.clip(cosang, -1.0, 1.0)
        total_angle += math.acos(cosang)
    return total_angle


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 固定随机种子保证可复现
    np.random.seed(42)
    random.seed(42)

    start = (5, 5)
    goal = (90, 90)

    # 生成障碍物，保证 A* 有解
    while True:
        obstacles, inflated = generate_obstacles(
            num=15, radius_range=(5, 8), robot_radius=2,
            start=start, goal=goal)
        grid = build_grid(inflated)
        astar_path = astar(grid, start, goal)
        if astar_path is not None:
            break

    # 运行 A*
    t0 = time.time()
    astar_path = astar(grid, start, goal)
    t_astar = time.time() - t0
    len_astar = path_length(astar_path)
    angle_astar = compute_turn_angle(astar_path)

    # 运行 PSO
    t0 = time.time()
    pso_path, pso_fit = pso_path_planning(start, goal, inflated,
                                          num_waypoints=6, swarm_size=50, iterations=200)
    t_pso = time.time() - t0
    len_pso = path_length(pso_path)
    angle_pso = compute_turn_angle(pso_path)

    # 打印对比结果
    print("========== 性能对比 ==========")
    print(f"A* 算法: 时间 = {t_astar:.4f}s, 路径长度 = {len_astar:.2f}, 总转折角度 = {angle_astar:.2f} rad")
    print(f"PSO 算法: 时间 = {t_pso:.4f}s, 路径长度 = {len_pso:.2f}, 总转折角度 = {angle_pso:.2f} rad")
    improvement = (len_astar - len_pso) / len_astar * 100
    print(f"PSO 相比 A* 路径长度变化: {improvement:+.1f}%")
    print(f"PSO 适应度最终值: {pso_fit:.2f}")

    # 绘制结果
    plot_results(grid, start, goal, obstacles, inflated, astar_path, pso_path)