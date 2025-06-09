"""1D P2D模型网格生成模块"""

import numpy as onp
from dataclasses import dataclass

@dataclass
class Mesh1D:
    name: str
    
    def to_dict(self):
        return self.__dict__

def generate_1d_mesh(infos):
    """
    生成1D P2D模型网格
    
    参数:
        infos: 包含网格信息的字典
               - z_an: 负极厚度
               - n_an: 负极网格数
               - z_se: 隔膜厚度  
               - n_se: 隔膜网格数
               - z_ca: 正极厚度
               - n_ca: 正极网格数
               - r: 颗粒半径
               - n_r: 颗粒径向网格数
    """
    
    # 宏观1D网格 (厚度方向)
    mesh_macro = generate_line_mesh(infos['z_an'], infos['n_an'], 
                                   infos['z_se'], infos['n_se'], 
                                   infos['z_ca'], infos['n_ca'])
    
    # 微观1D网格 (颗粒径向)
    mesh_micro = generate_interval_mesh(infos['r'], infos['n_r'])
    
    return [mesh_macro, mesh_micro]

def generate_interval_mesh(hr, Nr, name='jax_mesh_micro_1d'):
    """生成径向间隔网格"""
    
    mesh_micro = Mesh1D(name) 
    points = onp.linspace(0, hr, Nr+1).reshape(-1,1)
    mesh_micro.points = points
    mesh_micro.cells = onp.hstack((onp.linspace(0,len(points)-2,len(points)-1).reshape(-1,1),
                        onp.linspace(1,len(points)-1,len(points)-1).reshape(-1,1))).astype(onp.int32)
    
    mesh_micro.num_nodes = len(points)
    mesh_micro.num_cells = len(mesh_micro.cells)
    
    nodes_total = onp.linspace(0, len(points)-1, len(points), dtype=onp.int32)
    mesh_micro.bound_right = nodes_total[onp.isclose(points[:,0], hr)]
    
    return mesh_micro

def create_simple_line_mesh(n_elements, domain_length):
    """
    创建简单的1D线性网格
    
    参数:
        n_elements: 单元数量
        domain_length: 域长度
    
    返回:
        points: 节点坐标
        cells: 单元连接
    """
    
    # 节点坐标
    points = onp.linspace(0, domain_length, n_elements + 1).reshape(-1, 1)
    
    # 单元连接 (每个单元连接两个相邻节点)
    cells = onp.array([[i, i+1] for i in range(n_elements)], dtype=onp.int32)
    
    return points, cells

def generate_line_mesh(z_an, n_an, z_sp, n_sp, z_ca, n_ca):
    """
    生成1D线性网格 (负极 + 隔膜 + 正极)
    
    只适用于结构化LINE2网格 (宏观) 和结构化间隔网格 (微观)
    """
    
    # =========== 宏观1D网格 ===========
    
    # 创建各区域的1D网格
    an_points, an_cells = create_simple_line_mesh(n_an, z_an)
    sp_points, sp_cells = create_simple_line_mesh(n_sp, z_sp)
    ca_points, ca_cells = create_simple_line_mesh(n_ca, z_ca)
    
    # 调整隔膜和正极的坐标
    sp_points = sp_points + z_an
    ca_points = ca_points + z_an + z_sp
    
    # 合并节点，去除重复点
    points = onp.concatenate([an_points[:-1], sp_points[:-1], ca_points])
    
    # 调整单元连接索引
    sp_cells = sp_cells + n_an
    ca_cells = ca_cells + n_an + n_sp
    
    # 合并单元
    cells = onp.concatenate([an_cells, sp_cells, ca_cells])
    
    # 节点索引
    nodes_total = onp.arange(len(points), dtype=onp.int32)
    nind_an = n_an + 1
    nind_se = nind_an + n_sp
    nind_ca = len(points)
    
    # 单元索引
    cind_an = n_an
    cind_se = cind_an + n_sp
    cind_ca = cind_se + n_ca
    
    # 组装网格对象
    mesh_macro = Mesh1D('jax_mesh_macro_1d')
    
    mesh_macro.model = 'P2D_1D'
    mesh_macro.cell_type = 'line'
    mesh_macro.dim = 1
    
    mesh_macro.points = points
    mesh_macro.num_nodes = len(points)
    
    # 区域节点
    mesh_macro.nodes_anode = nodes_total[:nind_an]
    mesh_macro.nodes_separator = nodes_total[nind_an:nind_se]
    mesh_macro.nodes_cathode = nodes_total[nind_se:nind_ca]
    
    # 边界节点
    mesh_macro.nodes_bound_anright = nodes_total[onp.isclose(points.flatten(), z_an)]
    mesh_macro.nodes_bound_caleft = nodes_total[onp.isclose(points.flatten(), z_an + z_sp)]
    
    # 单元
    mesh_macro.cells = cells
    mesh_macro.cells_anode = cells[:cind_an]
    mesh_macro.cells_separator = cells[cind_an:cind_se]
    mesh_macro.cells_cathode = cells[cind_se:cind_ca]
    
    # 终端位置
    mesh_macro.terminal = len(mesh_macro.points) - 1
    
    # 分配自由度和辅助变量
    mesh_macro = assign_macro_dofs_1d(mesh_macro)
    mesh_macro = assign_aux_vars_1d(mesh_macro)
    
    return mesh_macro

def assign_macro_dofs_1d(mesh_macro):
    """为1D宏观网格分配自由度"""
    
    # 获取各区域的单元
    cells_anode = mesh_macro.cells_anode
    cells_separator = mesh_macro.cells_separator  
    cells_cathode = mesh_macro.cells_cathode
    
    # 为每个物理量分配自由度
    # p: 电解质电位, c: 电解质浓度, s: 固相电位, j: 电流密度
    
    # 电解质电位 (p) - 所有区域
    dofs_p = onp.arange(mesh_macro.num_nodes)
    
    # 电解质浓度 (c) - 所有区域  
    dofs_c = onp.arange(mesh_macro.num_nodes)
    
    # 固相电位 (s) - 仅电极区域
    dofs_s_anode = mesh_macro.nodes_anode
    dofs_s_cathode = mesh_macro.nodes_cathode
    dofs_s = onp.concatenate([dofs_s_anode, dofs_s_cathode])
    
    # 电流密度 (j) - 仅电极区域
    dofs_j = dofs_s.copy()
    
    mesh_macro.dofs_p = dofs_p
    mesh_macro.dofs_c = dofs_c  
    mesh_macro.dofs_s = dofs_s
    mesh_macro.dofs_j = dofs_j
    
    return mesh_macro

def assign_aux_vars_1d(mesh_macro):
    """为1D宏观网格分配辅助变量"""
    
    # 单元变量映射
    cells_vars = {}
    
    # 负极单元
    for i, cell in enumerate(mesh_macro.cells_anode):
        cells_vars[tuple(cell)] = {'region': 'anode', 'index': i}
    
    # 隔膜单元
    for i, cell in enumerate(mesh_macro.cells_separator):
        cells_vars[tuple(cell)] = {'region': 'separator', 'index': i}
        
    # 正极单元
    for i, cell in enumerate(mesh_macro.cells_cathode):
        cells_vars[tuple(cell)] = {'region': 'cathode', 'index': i}
    
    mesh_macro.cells_vars = cells_vars
    
    return mesh_macro 