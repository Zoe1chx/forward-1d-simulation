"""1D P2D电池模型基准测试 - 基于原始2D代码，使用真正的DiffLiB求解器"""

import os
import sys
import pickle
import time
import psutil

import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False,  # 禁用LaTeX以避免错误
    "font.family": 'sans-serif',
    "font.size": 12,
    "lines.linewidth": 2.5
})

# 添加DiffLiB路径以支持导入
difflib_src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if difflib_src_path not in sys.path:
    sys.path.insert(0, difflib_src_path)

# 导入DiffLiB核心模块 (与原始2D代码完全相同)
try:
    from cmsl.difflib.para import ParameterSets
    from cmsl.difflib.mesh import generate_mesh
    from cmsl.difflib.models import rectangle_cube
    from cmsl.difflib.wrap import fast_wrapper
    print("✅ 成功导入DiffLiB核心模块")
except ImportError as e:
    print(f"❌ 导入DiffLiB失败: {e}")
    print("请确保DiffLiB环境正确安装")
    sys.exit(1)

def print_resources_and_params_1d(stage_name, theta=None, additional_info=None):
    """打印1D模型资源使用情况和参数信息 (基于原始2D版本)"""
    print(f"\n📊 {stage_name} [1D模型]")
    print("-" * 60)
    
    # 系统资源
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_used_gb = (memory.total - memory.available) / 1024**3
    memory_percent = memory.percent
    
    print(f"💻 CPU使用率: {cpu_percent:.1f}%")
    print(f"🧠 内存使用: {memory_used_gb:.2f} GB ({memory_percent:.1f}%)")
    print(f"🖥️ JAX设备: {jax.devices()}")
    
    # 参数信息
    if theta is not None:
        print(f"📝 优化参数数量: {len(theta)}")
        param_names = ['alpha_an', 'alpha_ca', 'alpha_se', 'ks[0]', 'ks[1]', 'tp', 'cs0_an', 'cs0_ca']
        print("🔧 当前参数值:")
        for i, (name, value) in enumerate(zip(param_names, theta)):
            print(f"   {i+1}. {name}: {float(value):.6e}")
    
    # 额外信息
    if additional_info:
        for key, value in additional_info.items():
            print(f"📈 {key}: {value}")
    
    print("-" * 60)

def run_pybamm_1d_model(c_rates, ts_max, dt):
    """
    运行PyBaMM 1D P2D模型
    
    参数:
        c_rates: C倍率列表
        ts_max: 最大时间列表
        dt: 时间步长
    
    返回:
        voltages: 电压数据列表
    """
    
    try:
        import pybamm
        print("✅ 成功导入PyBaMM")
    except ImportError:
        raise ImportError("PyBaMM未安装，无法运行1D对比模型")
    
    # 创建1D P2D模型
    model = pybamm.lithium_ion.DFN(options={"dimensionality": 1})
    
    # 使用Marquis2019参数集 (与DiffLiB相同)
    parameter_values = pybamm.ParameterValues("Marquis2019")
    
    voltages = []
    
    for i, (c_rate, t_max) in enumerate(zip(c_rates, ts_max)):
        print(f"🔄 PyBaMM 1D仿真 {c_rate}C...")
        
        # 设置实验
        experiment = pybamm.Experiment([
            f"Discharge at {c_rate}C until 3.0V"
        ])
        
        # 创建仿真
        sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
        
        # 运行仿真
        solution = sim.solve()
        
        # 提取电压数据
        voltage_data = solution["Terminal voltage [V]"].data
        
        # 重采样到与DiffLiB相同的时间步长
        time_steps = int(t_max / dt) + 1
        voltage_resampled = onp.interp(
            onp.linspace(0, len(voltage_data)-1, time_steps),
            onp.arange(len(voltage_data)),
            voltage_data
        )
        
        voltages.append(voltage_resampled)
        print(f"✅ PyBaMM 1D {c_rate}C 完成")
    
    return voltages

def print_time_summary(stage_times, init_time, simulation_time, pybamm_time, plotting_time, total_time, c_rates, total_data_points):
    """打印详细的时间统计摘要"""
    print("\n" + "="*80)
    print("⏱️                     详细运行时间统计")
    print("="*80)
    
    # 初始化阶段
    print(f"🔧 模型初始化阶段:     {init_time:.4f} 秒 ({init_time/total_time*100:.1f}%)")
    
    # DiffLiB仿真阶段 
    print(f"🔋 DiffLiB仿真阶段:    {simulation_time:.4f} 秒 ({simulation_time/total_time*100:.1f}%)")
    
    # 分C-rate显示每个阶段的时间
    for i, (c_rate, stage_time) in enumerate(zip(c_rates, stage_times)):
        percentage = stage_time/simulation_time*100
        print(f"   ├─ {c_rate:>3}C 放电仿真:  {stage_time:.4f} 秒 ({percentage:.1f}%)")
    
    # PyBaMM处理阶段
    pybamm_percentage = pybamm_time/total_time*100
    print(f"📊 PyBaMM对比阶段:     {pybamm_time:.4f} 秒 ({pybamm_percentage:.1f}%)")
    
    # 绘图保存阶段
    plot_percentage = plotting_time/total_time*100
    print(f"📈 结果绘图阶段:       {plotting_time:.4f} 秒 ({plot_percentage:.1f}%)")
    
    print("-"*80)
    
    # 总时间 - 突出显示
    print(f"🕐 **总运行时间:        {total_time:.4f} 秒**")
    print(f"⚡ 平均单C-rate时间:   {simulation_time/len(c_rates):.4f} 秒")
    
    # 计算效率
    if simulation_time > 0:
        efficiency = total_data_points / simulation_time
        print(f"🚀 计算效率:           {efficiency:.1f} 个数据点/秒")
    
    print("="*80)

if __name__ == "__main__":
    
    print("🔋 启动1D P2D电池模型 (真正的DiffLiB求解器)")
    print("=" * 60)
    
    # 记录程序开始时间
    program_start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(program_start_time))
    print(f"🕐 程序开始时间: {start_time_str}")
    print("-" * 60)
    
    # 1D网格配置 (基于原始2D配置，去除y方向)
    mesh_infos_1d = {
        'z_an': 100., 'n_an': 20,
        'z_se': 25.,  'n_se': 20,
        'z_ca': 100., 'n_ca': 20,
        'y': 1.,      'n_y': 1,    # 1D模型：y方向设为最小值
        'r': 1.,      'n_r': 10
    }
    
    # 计算1D网格信息 (基于原始2D计算方式)
    total_mesh_points_1d = mesh_infos_1d['n_an'] * mesh_infos_1d['n_se'] * mesh_infos_1d['n_ca'] * mesh_infos_1d['n_r']
    estimated_dofs_1d = (mesh_infos_1d['n_an'] + mesh_infos_1d['n_se'] + mesh_infos_1d['n_ca']) * mesh_infos_1d['n_y'] * 4
    
    # 生成1D网格 (使用DiffLiB原生函数)
    init_start = time.time()
    mesh_1d = generate_mesh(mesh_infos_1d)
    
    # 1D前向预测包装器 (使用DiffLiB原生函数，与2D完全相同的调用方式)
    jax_wrapper_1d = fast_wrapper(ParameterSets, mesh_1d, rectangle_cube)
    init_time = time.time() - init_start
    
    # 当前负载条件 (与原始2D完全相同)
    dt = 5.
    c_rates = [0.2, 0.5, 1.0, 1.5, 2.]
    ts_max = [18475, 7330, 3620, 2385, 1770]
    
    # 初始化显示 (与原始2D完全相同的调用方式)
    _, paramsets = jax_wrapper_1d((0, 100), dt, c_rate=1.0)
    theta = np.array([paramsets.alpha_an, paramsets.alpha_ca, paramsets.alpha_se,
                      paramsets.ks[0], paramsets.ks[1],
                      paramsets.tp,
                      paramsets.cs0_an, paramsets.cs0_ca])
    
    mesh_info_1d = {
        '1D网格点总数': f"{total_mesh_points_1d:,}",
        '估算自由度': f"{estimated_dofs_1d:,}",
        '时间步长': f"{dt}秒",
        '模型维度': "1D (去除y方向)",
        '初始化耗时': f"{init_time:.3f}秒"
    }
    print_resources_and_params_1d("1D模型初始化参数和网格信息", theta, mesh_info_1d)
    
    # DiffLiB 1D仿真 (与原始2D完全相同的求解流程)
    dfb_voltages_1d = []
    simulation_start_time = time.time()
    stage_times = []  # 记录每个阶段的时间
    
    for i, (c_rate, t_max) in enumerate(zip(c_rates, ts_max)):
        stage_start = time.time()
        t_eval = (0, t_max)
        
        # 使用真正的DiffLiB求解器 (与原始2D完全相同)
        fwd_pred, paramsets = jax_wrapper_1d(t_eval, dt, c_rate=c_rate)
        theta = np.array([paramsets.alpha_an, paramsets.alpha_ca, paramsets.alpha_se,
                          paramsets.ks[0], paramsets.ks[1],
                          paramsets.tp,
                          paramsets.cs0_an, paramsets.cs0_ca])
        
        # 调用真正的DiffLiB求解器 (与原始2D完全相同)
        sol_bs, sol_macro_time, sol_micro_time, time_cost = fwd_pred(theta)
        stage_time = time.time() - stage_start
        stage_times.append(stage_time)
        
        dfb_voltages_1d.append(onp.array(sol_bs))
        
        # 输出当前阶段信息 (与原始2D相同格式)
        stage_info = {
            f'C-rate': f"{c_rate}C",
            '仿真进度': f"{i+1}/{len(c_rates)}",
            '时间步数': f"{t_max:,}",
            '仿真耗时': f"{stage_time:.3f}秒",
            '电压点数': f"{len(sol_bs):,}",
            '电压范围': f"{float(np.min(sol_bs)):.3f}V - {float(np.max(sol_bs)):.3f}V",
            '累计时间': f"{sum(stage_times):.3f}秒"
        }
        print_resources_and_params_1d(f"1D {c_rate}C 仿真完成", theta, stage_info)
    
    simulation_time = time.time() - simulation_start_time
    
    # 运行PyBaMM 1D P2D模型进行对比
    pybamm_start = time.time()
    try:
        print("🔄 运行PyBaMM 1D P2D模型进行对比...")
        pbm_voltages_1d = run_pybamm_1d_model(c_rates, ts_max, dt)
        has_pybamm_1d = True
        print("✅ PyBaMM 1D模型运行成功")
    except Exception as e:
        print(f"⚠️ PyBaMM 1D模型运行失败: {e}")
        print("🔄 尝试加载2D参考数据...")
        # 回退到加载2D参考数据
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        input_path = os.path.join(parent_dir, 'forward', 'input', 'pybamm_data.pkl')
        
        try:
            with open(input_path, 'rb') as f:
                pbm_voltages_1d = pickle.load(f)
            has_pybamm_1d = False
            print("✅ 成功加载PyBaMM 2D参考数据")
        except FileNotFoundError:
            print("⚠️ 未找到PyBaMM参考数据，将仅绘制1D DiffLiB结果")
            pbm_voltages_1d = None
            has_pybamm_1d = False
    
    pybamm_time = time.time() - pybamm_start
    
    # 后处理和绘图 (分开保存DiffLiB和PyBaMM结果)
    plotting_start = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制DiffLiB 1D结果
    plt.figure(figsize=(10, 8))
    count = 0
    for c_rate, dfb_data_1d in zip(c_rates, dfb_voltages_1d):
        x = dt * onp.arange(1, len(dfb_data_1d))/int(3600/c_rate) * 24
        plt.plot(x, dfb_data_1d[1:], label=fr'${c_rate}\,C$')
        count += 1
    
    plt.xlim([-1, 25])
    plt.xticks([0,6,12,18,24])
    plt.ylim([3.1, 3.9])
    plt.yticks([3.1,3.3,3.5,3.7,3.9])
    plt.xlabel('Discharge capacity (Ah/m²)')
    plt.ylabel('Terminal voltage (V)')
    plt.title('1D P2D Model Results - DiffLiB')
    plt.legend(frameon=False, loc='lower left')
    
    # 保存DiffLiB结果
    difflib_output_path = os.path.join(output_dir, 'voltage_1d_difflib.png')
    plt.savefig(difflib_output_path, dpi=300, format="png", bbox_inches='tight')
    plt.close()
    
    # 如果有PyBaMM数据，绘制PyBaMM结果
    if pbm_voltages_1d is not None:
        plt.figure(figsize=(10, 8))
        count = 0
        for c_rate in c_rates:
            if count < len(pbm_voltages_1d):
                pbm_data = pbm_voltages_1d[count]
                x_ref = dt * onp.arange(1, len(pbm_data))/int(3600/c_rate) * 24
                model_type = "1D" if has_pybamm_1d else "2D"
                plt.plot(x_ref, pbm_data[1:], label=fr'${c_rate}\,C$')
            count += 1
        
        plt.xlim([-1, 25])
        plt.xticks([0,6,12,18,24])
        plt.ylim([3.1, 3.9])
        plt.yticks([3.1,3.3,3.5,3.7,3.9])
        plt.xlabel('Discharge capacity (Ah/m²)')
        plt.ylabel('Terminal voltage (V)')
        model_type = "1D" if has_pybamm_1d else "2D"
        plt.title(f'{model_type} P2D Model Results - PyBaMM')
        plt.legend(frameon=False, loc='lower left')
        
        # 保存PyBaMM结果
        model_type_lower = "1d" if has_pybamm_1d else "2d"
        pybamm_output_path = os.path.join(output_dir, f'voltage_{model_type_lower}_pybamm.png')
        plt.savefig(pybamm_output_path, dpi=300, format="png", bbox_inches='tight')
        plt.close()
    
    # 绘制对比图 (如果有PyBaMM数据) - 基于原始2D代码的绘图方式
    if pbm_voltages_1d is not None:
        plt.figure(figsize=(10, 8))
        count = 0
        for c_rate, dfb_data_1d in zip(c_rates, dfb_voltages_1d):
            if count < len(pbm_voltages_1d):
                pbm_data = pbm_voltages_1d[count]
                x = dt * onp.arange(1, len(pbm_data))/int(3600/c_rate) * 24
                plt.plot(x, dfb_data_1d[1:], label=fr'${c_rate}\,C$')
                plt.scatter(x, pbm_data[1:], s=60, color=f'C{count}', facecolor='None', label=None)
            count += 1
        
        plt.xlim([-1, 25])
        plt.xticks([0,6,12,18,24])
        plt.ylim([3.1, 3.9])
        plt.yticks([3.1,3.3,3.5,3.7,3.9])
        plt.xlabel('Discharge capacity (Ah/m²)')
        plt.ylabel('Terminal voltage (V)')
        plt.legend(frameon=False, loc='lower left')
        
        # 保存对比图
        comparison_type_file = "1d_vs_1d" if has_pybamm_1d else "1d_vs_2d"
        comparison_output_path = os.path.join(output_dir, f'voltage_{comparison_type_file}_comparison.png')
        plt.savefig(comparison_output_path, dpi=300, format="png", bbox_inches='tight')
        plt.close()
    
    plotting_time = time.time() - plotting_start
    
    # 计算总运行时间
    total_program_time = time.time() - program_start_time
    
    # 计算文件大小
    difflib_size = os.path.getsize(difflib_output_path) / 1024
    output_files = [f"voltage_1d_difflib.png ({difflib_size:.1f} KB)"]
    
    if pbm_voltages_1d is not None:
        pybamm_size = os.path.getsize(pybamm_output_path) / 1024
        comparison_size = os.path.getsize(comparison_output_path) / 1024
        model_type_lower = "1d" if has_pybamm_1d else "2d"
        comparison_type_file = "1d_vs_1d" if has_pybamm_1d else "1d_vs_2d"
        output_files.extend([
            f"voltage_{model_type_lower}_pybamm.png ({pybamm_size:.1f} KB)",
            f"voltage_{comparison_type_file}_comparison.png ({comparison_size:.1f} KB)"
        ])
    
    # 输出详细的时间统计
    total_data_points = sum(len(v) for v in dfb_voltages_1d)
    print_time_summary(stage_times, init_time, simulation_time, pybamm_time, plotting_time, total_program_time, c_rates, total_data_points)
    
    # 显示程序结束时间
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"🏁 程序结束时间: {end_time_str}")
    
    final_info = {
        '总程序运行时间': f"**{total_program_time:.3f}秒**",
        'DiffLiB仿真时间': f"{simulation_time:.3f}秒",
        '平均单C-rate时间': f"{simulation_time/len(c_rates):.3f}秒",
        '生成数据点': f"{total_data_points:,}",
        '输出文件数量': f"{len(output_files)}个",
        '输出文件': "; ".join(output_files),
        '求解器类型': "DiffLiB原生求解器 (与2D相同)"
    }
    print_resources_and_params_1d("1D测试完成 - 最终统计", theta, final_info) 