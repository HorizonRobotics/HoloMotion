import struct


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16
        # 添加滤波器参数
        self.alpha = 0.3  # 滤波系数 (0-1之间，越小滤波效果越强)
        self.deadzone = 0.05  # 死区阈值
        
        # 添加上一次的状态值用于滤波
        self.lx_prev = 0
        self.ly_prev = 0
        self.rx_prev = 0
        self.ry_prev = 0
        self.smooth = 0.03  # 平滑系数，值越小越平滑
        self.dead_zone = 0.01  # 死区阈值

        # 速度映射参数
        self.max_linear_speed = 1.0  # 最大线速度 (m/s)
        self.max_angular_speed = 1.0  # 最大角速度 (rad/s)

    def apply_filter_and_deadzone(self, value, prev_value):
        # 结合死区判断和平滑处理
        if abs(value) < self.dead_zone:
            value = 0.0
        return prev_value * (1 - self.smooth) + value * self.smooth

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
            
        # 读取原始值
        lx_raw = struct.unpack("f", data[4:8])[0]
        rx_raw = struct.unpack("f", data[8:12])[0]
        ry_raw = struct.unpack("f", data[12:16])[0]
        ly_raw = struct.unpack("f", data[20:24])[0]
        
        # 应用滤波和死区
        self.lx = self.apply_filter_and_deadzone(lx_raw, self.lx_prev)
        self.ly = self.apply_filter_and_deadzone(ly_raw, self.ly_prev)
        self.rx = self.apply_filter_and_deadzone(rx_raw, self.rx_prev)
        self.ry = self.apply_filter_and_deadzone(ry_raw, self.ry_prev)
        
        # 更新前一次的值
        self.lx_prev = self.lx
        self.ly_prev = self.ly
        self.rx_prev = self.rx
        self.ry_prev = self.ry

    def get_velocity_commands(self):
        """
        将摇杆值转换为速度命令
        Returns:
            tuple: (vx, vy, vyaw) 
            - vx: 前进/后退速度 (m/s)，由左摇杆前后(ly)控制
            - vy: 左右平移速度 (m/s)，由左摇杆左右(lx)控制
            - vyaw: 转向角速度 (rad/s)，由右摇杆左右(rx)控制
        """
        # 前进/后退速度，使用左摇杆的y轴
        vx = self.ly * self.max_linear_speed  # 注意：通常需要取反，因为向前推摇杆时ly为负
        
        # 左右平移速度，使用左摇杆的x轴
        vy = 0.5*self.lx * self.max_linear_speed  # 注意：可能需要取反，取决于坐标系定义
        
        # 转向角速度，使用右摇杆的x轴
        vyaw = -self.rx * self.max_angular_speed  # 注意：可能需要取反，取决于坐标系定义
        
        return vx, vy, vyaw
