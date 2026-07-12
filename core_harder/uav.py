from dataclasses import dataclass, field
from typing import List, ClassVar, TYPE_CHECKING
import math
import numpy as np

from .env_config import UAV_HEIGHT, _BATTERY_CAPACITY


@dataclass
class UAVNode:
    # ------------------- 实例属性 (每个UAV独立拥有) -------------------
    loc: np.ndarray = field(
        default_factory=lambda: np.array([20, 20], dtype=np.float32)
    )
    velocity: np.ndarray = field(
        default_factory=lambda: np.array([0, 0], dtype=np.float32)
    )
    # 当前电量 - 必须是实例属性，确保每个UAV电量独立
    e_battery: float = field(default=_BATTERY_CAPACITY)
    # 当前飞行速度 - 如果它表示当前状态
    flying_speed: float = field(default=20.0)
    # 是否坠毁标志
    is_crashed: bool = field(default=False)
    harvested_energy_last_step: float = field(default=0.0)
    node_id: int = field(default=-1)
    cpu_freq: float = field(default=2.4e9)  # 2.4 GHz
    charging_power: float = field(default=2000.0)  # 2000 W

    # ------------------- 类属性 (所有UAV共享的常量配置) -------------------
    # 常量配置 (不变的，所以作为类属性是安全的)
    battery_capacity: float = _BATTERY_CAPACITY
    flying_height: float = UAV_HEIGHT
    f_uav: float = 2.4e9
    m_uav: float = 9.65
    t_fly: float = 1.0
    r: float = 10 ** (-27)
    max_speed: float = 20.0

    # 最大的加速度数组 - 设为类属性是为了方便引用和共享。
    # 因为 NumPy 数组是可变的，如果担心未来误修改，可以使用 tuple 或 frozen=True
    # 或者用 @property 保护它。
    MAX_ACCELERATION: ClassVar[np.ndarray] = np.array([[5.0, 5.0]], dtype=np.float32)

    # Unified moveto kept below; remove moveto2 to avoid duplicate APIs.

    def moveto(self, target_loc, target_velocity, average_velocity, dt):
        """Move UAV to `target_loc`.

        Args:
            target_loc (array-like): destination coordinate [x,y].
            target_velocity (array-like): velocity vector to set after move [vx,vy].
            average_velocity (float): scalar average speed magnitude used for power calc.
            dt (float): time duration of the move (s).

        Behavior:
            - compute power via `calculate_uav_power(average_velocity)`,
            - consume energy = power * dt (uses `consume_energy` to handle crash),
            - update `loc` and `velocity`, and return `energy_consumed`.

        Returns:
            float: energy_consumed during this move.
        """
        # Defensive casts
        tgt = np.array(target_loc, dtype=np.float32)
        vel = np.array(target_velocity, dtype=np.float32)

        # Calculate power (supports scalar or 1-element array)
        pw = calculate_uav_power(np.array([float(average_velocity)]))[0]
        energy_consumed = float(pw) * float(dt)

        # Consume energy with safety: marks `is_crashed` if battery drained
        self.consume_energy(energy_consumed)

        # Update kinematic state
        self.loc = tgt
        self.velocity = vel

        return energy_consumed

    def consume_energy(self, energy: float) -> bool:
        """Consume energy from the UAV battery.

        Args:
            energy: amount of energy to consume (J). Must be non-negative.

        Returns:
            bool: True if the UAV became crashed (battery depleted) after consumption, else False.
        """
        energy = float(energy)
        if energy < 0:
            raise ValueError("energy must be non-negative")

        actual_consumed = min(energy, self.e_battery)
        self.e_battery -= actual_consumed
        if self.e_battery <= 0:
            self.e_battery = 0.0
            self.is_crashed = True
        return bool(self.is_crashed)

    # 返回飞行耗能和奖励

    def receive_energy(self, energy):
        """接收能量并更新电池状态"""
        actual_energy = min(energy, self.battery_capacity - self.e_battery)
        self.e_battery += actual_energy
        return actual_energy

    def record_harvested_energy(self, energy: float, apply: bool = True) -> float:
        """Record energy harvested from a charger.

        By default (`apply=True`) this will add the energy to the UAV via
        `receive_energy` and store the actual amount added in
        `harvested_energy_last_step`.

        If `apply=False`, the method will only record the reported harvested
        amount in `harvested_energy_last_step` without modifying the battery
        (useful when the caller already applied the energy, e.g. when
        `LaserCharger.charge()` has already called `receive_energy`).

        Returns the recorded amount (float).
        """
        energy = float(energy)
        if apply:
            actual = self.receive_energy(energy)
        else:
            actual = energy
        self.harvested_energy_last_step = actual
        return float(actual)

    def clear_harvested_energy(self) -> None:
        """Clear the per-step harvested energy record."""
        self.harvested_energy_last_step = 0.0

    def reset(self, start_loc=None):
        self.loc = (
            start_loc.astype(np.float32)
            if start_loc is not None
            else np.array([20, 20], dtype=np.float32)
        )
        self.e_battery = self.battery_capacity  # 重置电池电量
        self.is_crashed = False  # 重置坠毁状态
        self.flying_speed = 0.0
        self.velocity = np.array([0, 0], dtype=np.float32)


# -----------------------------------------------------------
# 1. 定义常量 (根据 Table 1. KEY SIMULATION SETTINGS)
# -----------------------------------------------------------
P0 = 79.86  # 悬停时的叶型功率 (W)
P1 = 88.63  # 悬停时的诱导功率 (W)
U_tip = 120  # 桨叶尖端速度 (m/s)
v_0 = 4.03  # 悬停时的平均诱导速度 (m/s)
d_0 = 0.6  # 机身阻力系数 (dimensionless)
rho = 1.225  # 空气密度 (kg/m^3)
s = 0.05  # 旋翼实体比 (dimensionless)
A = 0.503  # 旋翼桨盘面积 (m^2)


def calculate_uav_power(v_h):
    """
    计算无人机在给定水平速度数组下的总功率消耗。

        Args:
            v_h (np.ndarray): 包含所有无人机平均合速度的数组 (m/s).

        Returns:
            np.ndarray: 对应速度下无人机所需的总功率数组 (W).
    """

    # 第一部分：桨叶外形功率 (Profile Power)
    power_profile = P0 * (1 + (3 * (v_h**2)) / (U_tip**2))

    # 第二部分：诱导功率 (Induced Power)
    # 先计算根号下的复杂项
    term_under_sqrt = 1 + (v_h**4) / (4 * (v_0**4)) - (v_h**2) / (2 * (v_0**2))

    # ⚠️ 安全检查：确保根号下的项不为负数（理论上不应发生，但数值计算可能出现）
    term_under_sqrt = np.maximum(term_under_sqrt, 0)

    power_induced = P1 * np.sqrt(term_under_sqrt)
    # 第三部分：寄生功率 (Parasite Power)
    power_parasite = 0.5 * d_0 * rho * s * A * (v_h**3)

    # 计算总功率
    total_power = power_profile + power_induced + power_parasite

    return total_power


# -----------------------------------------------------------
# 3. 主程序入口：演示如何使用这个函数
# -----------------------------------------------------------
if __name__ == "__main__":
    # 假设我们想计算无人机以 10 m/s 速度飞行时的功率
    example_velocity = 10.0  # m/s

    # 调用函数进行计算
    power_needed = calculate_uav_power(example_velocity)

    # 打印结果，并格式化输出，保留两位小数
    print(
        f"无人机以 {example_velocity} m/s 的速度飞行时, 所需的总功率为: {power_needed:.2f} W"
    )

    # 再测试一个速度
    example_velocity_2 = 20.0  # m/s
    power_needed_2 = calculate_uav_power(example_velocity_2)
    print(
        f"无人机以 {example_velocity_2} m/s 的速度飞行时, 所需的总功率为: {power_needed_2:.2f} W"
    )

    example_velocity_3 = 0  # m/s
    power_needed_3 = calculate_uav_power(example_velocity_3)
    print(
        f"无人机以 {example_velocity_3} m/s 的速度飞行时, 所需的总功率为: {power_needed_3:.2f} W"
    )
