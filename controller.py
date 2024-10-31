import math
from . import data_struct as struct
from . import utils as utils

'''
创建 PID 控制器类
'''
class PID:
    previous_error_ = 0 # 上一个控制周期的误差
    accum_error_    = 0 # 累计误差
    
    def __init__(self, Kp, Ki, Kd):
        self.Kp_ = Kp # 初始化比例系数
        self.Ki_ = Ki # 初始化积分系数
        self.Kd_ = Kd # 初始化微分系数
        
    def update_control(self, target_v, current_v, dt):
        dt = max(dt, 1e-3)
        # 计算比例项
        error = target_v - current_v
        proportion_term = self.Kp_ * error
        
        # 计算积分项
        self.accum_error_ += error
        integral_term = self.Ki_ * self.accum_error_
        
        # 计算微分项
        derivative_term = self.Kd_ * (error - self.previous_error_) / dt
        
        # 计算控制量
        output = proportion_term + integral_term + derivative_term
        
        # 更新上一个控制周期的误差
        self.previous_error_ = error
            
        return output

'''
创建 Stanley 控制器类
'''
class Stanley:
    def __init__(self, k):
        self.k_ = k # 初始化预瞄距离增益参数
        
    def update_control(self, P1: struct.Transform, ego: struct.State ):
        # 计算横向误差及控制量
        d_y = P1.y_ - ego.pose_.y_
        d_x = P1.x_ - ego.pose_.x_

        e_y = d_y * math.cos(P1.theta_) - d_x * math.sin(P1.theta_)
        
        delta_y = math.atan2(self.k_ * e_y, ego.v_ + 1e-3)
        
        # 计算航向误差及控制量
        theta_phi = utils.normalize_angle(
            utils.normalize_angle(P1.theta_) 
            - utils.normalize_angle(ego.pose_.theta_)
            ) 
        
        delta_theta = theta_phi

        # 计算最终的转向角度
        #delta = delta_theta + delta_y
        delta = delta_y * (0.35 - ego.v_ * 0.01 ) + delta_theta * (0.35 + ego.v_ *0.01)
        # delta = delta_y * 0.35 + delta_theta * 0.3
        delta = delta * 0.35

        return delta 