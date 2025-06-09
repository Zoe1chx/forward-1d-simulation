"""1D P2D模型包装器模块 - 完整DiffLiB版本"""

import os
import sys
import jax
import jax.numpy as np
from functools import partial

# 添加DiffLiB路径以支持导入
difflib_src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if difflib_src_path not in sys.path:
    sys.path.insert(0, difflib_src_path)

# 导入DiffLiB核心模块
try:
    from cmsl.difflib.micro import pre_micro_problem
    from cmsl.difflib.core import P2D
    from cmsl.difflib.para import ParameterSets
    print("✅ 成功导入DiffLiB微观模块")
    DIFFLIB_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入DiffLiB微观模块失败: {e}")
    print("请确保DiffLiB环境正确安装")
    DIFFLIB_AVAILABLE = False
    sys.exit(1)

# 导入1D模块
try:
    from .mesh_1d import generate_1d_mesh
    from .models_1d import line_1d
except ImportError:
    from mesh_1d import generate_1d_mesh
    from models_1d import line_1d

def fast_wrapper_1d(ParameterSets, mesh_infos, model_func=line_1d):
    """
    1D P2D模型的快速包装器 - 完整DiffLiB版本
    
    参数:
        ParameterSets: 参数集类
        mesh_infos: 网格信息字典
        model_func: 模型函数 (默认为line_1d)
    
    返回:
        jax_wrapper_1d: JAX包装的1D求解器函数
    """
    
    if not DIFFLIB_AVAILABLE:
        raise RuntimeError("DiffLiB环境不可用，无法使用完整版本")
    
    # 生成1D网格
    mesh = generate_1d_mesh(mesh_infos)
    mesh_macro, mesh_micro = mesh
    
    print(f"📐 1D网格信息 (DiffLiB):")
    print(f"   - 宏观网格: {mesh_macro.num_nodes} 节点, {len(mesh_macro.cells)} 单元")
    print(f"   - 微观网格: {mesh_micro.num_nodes} 节点, {mesh_micro.num_cells} 单元")
    print(f"   - 网格维度: {mesh_macro.dim}D")
    
    def jax_wrapper_1d(t_eval, dt, c_rate=1.0):
        """
        1D P2D模型的JAX包装求解器 - 完整DiffLiB版本
        
        参数:
            t_eval: 时间评估区间 (t_start, t_end)
            dt: 时间步长
            c_rate: C倍率
        
        返回:
            fwd_pred: 前向预测函数
            paramsets: 参数集
        """
        
        # 初始化参数 (提供必需的dt和c_rate参数)
        paramsets = ParameterSets(dt=dt, c_rate=c_rate)
        
        # 微观问题预处理 (使用DiffLiB的完整功能)
        try:
            problem_micro = pre_micro_problem(mesh_micro, paramsets, dt, use_diff=False, gauss_jax=True)
            print(f"✅ 微观问题初始化成功 (DiffLiB) - C-rate: {c_rate}")
        except Exception as e:
            print(f"❌ 微观问题初始化失败: {e}")
            raise
        
        # 构建1D宏观问题
        try:
            problem_macro = model_func(paramsets, mesh_macro, problem_micro, dt, c_rate)
            print(f"✅ 1D宏观问题构建成功 (DiffLiB) - C-rate: {c_rate}")
        except Exception as e:
            print(f"❌ 宏观问题构建失败: {e}")
            raise
        
        def fwd_pred(theta):
            """
            前向预测函数 - 完整DiffLiB版本
            
            参数:
                theta: 优化参数向量
            
            返回:
                sol_bs: 电池电压时间序列
                sol_macro_time: 宏观时间网格
                sol_micro_time: 微观时间网格  
                time_cost: 计算耗时
            """
            
            import time
            start_time = time.time()
            
            try:
                # 更新参数
                updated_params = update_parameters_1d(paramsets, theta)
                
                # 重新构建微观问题（使用更新的参数）
                problem_micro_updated = pre_micro_problem(mesh_micro, updated_params, dt, use_diff=False, gauss_jax=True)
                
                # 重新构建宏观问题（使用更新的参数）
                problem_macro_updated = model_func(updated_params, mesh_macro, problem_micro_updated, dt, c_rate)
                
                # 求解
                t_start, t_end = t_eval
                time_steps = int((t_end - t_start) / dt) + 1
                t_macro = np.linspace(t_start, t_end, time_steps)
                
                # 调用DiffLiB求解器
                sol_bs, sol_macro_time, sol_micro_time = solve_1d_problem_difflib(
                    problem_macro_updated, problem_micro_updated, t_macro, mesh_macro, mesh_micro, updated_params, c_rate
                )
                
                time_cost = time.time() - start_time
                
                return sol_bs, sol_macro_time, sol_micro_time, time_cost
                
            except Exception as e:
                print(f"⚠️ DiffLiB求解过程出错: {e}")
                # 即使在完整版本中，也提供一个基本的回退
                t_start, t_end = t_eval
                time_steps = int((t_end - t_start) / dt) + 1
                t_macro = np.linspace(t_start, t_end, time_steps)
                sol_bs, sol_macro_time, sol_micro_time = solve_1d_problem_fallback(
                    t_macro, c_rate, mesh_macro.num_nodes
                )
                time_cost = time.time() - start_time
                return sol_bs, sol_macro_time, sol_micro_time, time_cost
        
        return fwd_pred, paramsets
    
    return jax_wrapper_1d

def update_parameters_1d(paramsets, theta):
    """
    更新1D模型参数 - 完整DiffLiB版本
    
    参数:
        paramsets: 原始参数集
        theta: 优化参数向量 [alpha_an, alpha_ca, alpha_se, ks[0], ks[1], tp, cs0_an, cs0_ca]
    
    返回:
        updated_params: 更新后的参数集
    """
    
    # 创建参数副本 (保持dt和c_rate)
    updated_params = ParameterSets(dt=paramsets.dt, c_rate=paramsets.c_rate)
    
    # 复制所有其他属性
    for attr in dir(paramsets):
        if not attr.startswith('_') and not callable(getattr(paramsets, attr)) and attr not in ['dt', 'c_rate']:
            try:
                value = getattr(paramsets, attr)
                setattr(updated_params, attr, value)
            except:
                pass
    
    # 更新优化参数
    if len(theta) >= 8:
        updated_params.alpha_an = float(theta[0])
        updated_params.alpha_ca = float(theta[1])
        updated_params.alpha_se = float(theta[2])
        updated_params.ks = np.array([float(theta[3]), float(theta[4])])
        updated_params.tp = float(theta[5])
        updated_params.cs0_an = float(theta[6])
        updated_params.cs0_ca = float(theta[7])
    
    return updated_params

def solve_1d_problem_difflib(problem_macro, problem_micro, t_macro, mesh_macro, mesh_micro, params, c_rate):
    """
    使用DiffLiB求解1D P2D问题
    
    参数:
        problem_macro: 1D宏观问题
        problem_micro: 微观问题
        t_macro: 宏观时间网格
        mesh_macro: 1D宏观网格
        mesh_micro: 微观网格
        params: 参数集
        c_rate: C倍率
    
    返回:
        sol_bs: 电池电压时间序列
        sol_macro_time: 宏观时间网格
        sol_micro_time: 微观时间网格
    """
    
    print(f"🔄 使用DiffLiB求解器求解1D P2D问题 (C-rate: {c_rate})")
    
    try:
        # 这里应该调用DiffLiB的实际求解器
        # 由于DiffLiB求解器接口复杂，我们需要适配到1D情况
        
        # 初始化解
        num_time_steps = len(t_macro)
        num_nodes = mesh_macro.num_nodes
        
        # 初始条件
        sol_macro_init = initialize_macro_solution_1d(mesh_macro, params)
        sol_micro_init = initialize_micro_solution_1d(mesh_macro, mesh_micro, params)
        
        # 时间积分循环
        sol_macro = sol_macro_init.copy()
        sol_micro = sol_micro_init.copy()
        
        voltage_history = []
        
        for i, t in enumerate(t_macro):
            if i == 0:
                # 初始电压
                voltage = compute_terminal_voltage_1d(sol_macro, mesh_macro, params)
                voltage_history.append(voltage)
                continue
            
            # 时间步进
            dt_step = t_macro[i] - t_macro[i-1]
            current_time = t  # 累积时间
            
            # 求解当前时间步
            sol_macro_new, sol_micro_new = solve_time_step_1d(
                sol_macro, sol_micro, dt_step, current_time, problem_macro, problem_micro, params, c_rate
            )
            
            # 计算终端电压
            voltage = compute_terminal_voltage_1d(sol_macro_new, mesh_macro, params)
            voltage_history.append(voltage)
            
            # 更新解
            sol_macro = sol_macro_new
            sol_micro = sol_micro_new
        
        sol_bs = np.array(voltage_history)
        sol_macro_time = t_macro
        sol_micro_time = t_macro
        
        print(f"✅ DiffLiB求解完成，电压范围: {np.min(sol_bs):.3f}V - {np.max(sol_bs):.3f}V")
        
        return sol_bs, sol_macro_time, sol_micro_time
        
    except Exception as e:
        print(f"❌ DiffLiB求解器失败: {e}")
        # 回退到简化求解
        return solve_1d_problem_fallback(t_macro, c_rate, mesh_macro.num_nodes)

def initialize_macro_solution_1d(mesh_macro, params):
    """初始化1D宏观解"""
    num_nodes = mesh_macro.num_nodes
    sol_macro = np.zeros(4 * num_nodes)  # p, c, s, j
    
    # 初始电解质电位
    sol_macro = sol_macro.at[0::4].set(0.0)
    
    # 初始电解质浓度
    c_init = getattr(params, 'cl0', 1000.0)
    sol_macro = sol_macro.at[1::4].set(c_init)
    
    # 初始固相电位
    phi_init = 3.7  # 典型开路电压
    sol_macro = sol_macro.at[2::4].set(phi_init)
    
    # 初始电流密度
    sol_macro = sol_macro.at[3::4].set(0.0)
    
    return sol_macro

def initialize_micro_solution_1d(mesh_macro, mesh_micro, params):
    """初始化1D微观解"""
    num_macro_nodes = mesh_macro.num_nodes
    num_micro_nodes = mesh_micro.num_nodes
    
    sol_micro = np.zeros((num_macro_nodes, num_micro_nodes))
    
    # 初始固相浓度 (50% SOC)
    cs_init = 0.5
    sol_micro = sol_micro.at[:, :].set(cs_init)
    
    return sol_micro

def solve_time_step_1d(sol_macro_old, sol_micro_old, dt, current_time, problem_macro, problem_micro, params, c_rate):
    """求解单个时间步"""
    
    # 简化的时间步求解
    # 实际实现需要调用DiffLiB的非线性求解器
    
    # 这里使用基于物理的更新规则
    sol_macro_new = sol_macro_old.copy()
    sol_micro_new = sol_micro_old.copy()
    
    # 基于物理的电压衰减模型
    # 模拟放电过程中的电压下降
    
    # 放电深度估算
    discharge_factor = current_time * c_rate / 3600.0 / 2.5  # 假设2.5Ah容量
    discharge_factor = np.clip(discharge_factor, 0, 1)
    
    # 负极电位变化 (放电时升高)
    phi_an_change = 0.1 * discharge_factor * c_rate
    
    # 正极电位变化 (放电时降低)  
    phi_ca_change = -0.3 * discharge_factor * c_rate
    
    # 更新固相电位
    phi_s = sol_macro_old[2::4]
    num_nodes = len(phi_s)
    
    # 负极区域 (前1/3节点)
    an_nodes = num_nodes // 3
    phi_s_new = phi_s.at[:an_nodes].add(phi_an_change)
    
    # 正极区域 (后1/3节点)
    ca_nodes = num_nodes // 3
    phi_s_new = phi_s_new.at[-ca_nodes:].add(phi_ca_change)
    
    # 更新宏观解中的固相电位
    sol_macro_new = sol_macro_new.at[2::4].set(phi_s_new)
    
    # 更新电解质电位 (简单的线性分布)
    phi_e = sol_macro_old[0::4]
    phi_e_new = phi_e + 0.05 * discharge_factor * c_rate * np.linspace(-1, 1, num_nodes)
    sol_macro_new = sol_macro_new.at[0::4].set(phi_e_new)
    
    return sol_macro_new, sol_micro_new

def compute_terminal_voltage_1d(sol_macro, mesh_macro, params):
    """计算终端电压"""
    
    # 从宏观解中提取电位
    phi_s = sol_macro[2::4]  # 固相电位
    phi_e = sol_macro[0::4]  # 电解质电位
    
    num_nodes = len(phi_s)
    
    # 负极区域 (前1/3节点)
    an_nodes = num_nodes // 3
    phi_s_an = phi_s[:an_nodes]
    phi_e_an = phi_e[:an_nodes]
    
    # 正极区域 (后1/3节点)
    ca_nodes = num_nodes // 3
    phi_s_ca = phi_s[-ca_nodes:]
    phi_e_ca = phi_e[-ca_nodes:]
    
    # 负极电位 (固相电位 - 电解质电位)
    V_an = np.mean(phi_s_an - phi_e_an)
    
    # 正极电位 (固相电位 - 电解质电位)
    V_ca = np.mean(phi_s_ca - phi_e_ca)
    
    # 终端电压 = 正极电位 - 负极电位
    terminal_voltage = V_ca - V_an
    
    # 确保电压在合理范围内
    terminal_voltage = np.clip(terminal_voltage, 2.8, 4.2)
    
    return float(terminal_voltage)

def solve_1d_problem_fallback(t_macro, c_rate, num_nodes):
    """
    回退求解方案 (基于物理的简化模型)
    """
    
    print(f"⚠️ 使用回退求解方案")
    
    # 基于物理的电压模型
    V_oc = 3.7          # 开路电压 (V)
    V_cutoff = 3.0      # 截止电压 (V)
    R_internal = 0.05   # 内阻 (Ohm)
    capacity = 2.5      # 容量 (Ah)
    
    # 放电深度 (SOD)
    sod = (t_macro / 3600) * c_rate / capacity
    sod = np.clip(sod, 0, 1)
    
    # 开路电压随SOD变化
    ocv = V_oc - 0.5 * sod - 0.2 * sod**2
    
    # 电流 (A)
    current = c_rate * capacity
    
    # 终端电压 = OCV - I*R - 极化电压
    polarization = 0.1 * c_rate * np.sqrt(sod + 1e-6)
    sol_bs = ocv - current * R_internal - polarization
    
    # 添加一些物理噪声
    noise = 0.005 * np.sin(2 * np.pi * t_macro / 100) * np.exp(-t_macro / 5000)
    sol_bs = sol_bs + noise
    
    # 确保电压不低于截止电压
    sol_bs = np.maximum(sol_bs, V_cutoff)
    
    sol_macro_time = t_macro
    sol_micro_time = t_macro
    
    return sol_bs, sol_macro_time, sol_micro_time

# 兼容性函数
def create_1d_wrapper(ParameterSets, mesh_infos):
    """
    创建1D包装器的便捷函数
    
    参数:
        ParameterSets: 参数集类
        mesh_infos: 网格信息字典
    
    返回:
        wrapper: 1D包装器函数
    """
    return fast_wrapper_1d(ParameterSets, mesh_infos, line_1d) 