"""
Graph-WaveNet训练脚本
使用示例：
python train.py --data data/FRANCE --device cuda:0 --batch_size 64 --epochs 5 \
--seq_length 3 --pred_length 3 --learning_rate 0.001 --dropout 0.3 --nhid 64 \
--weight_decay 0.0001 --print_every 50 --gcn_bool --addaptadj --randomadj \
--adjtype doubletransition
"""

# 导入PyTorch深度学习框架
import torch
# 导入NumPy数值计算库
import numpy as np
# 导入命令行参数解析库
import argparse
# 导入时间处理库
import time
# 导入项目自定义工具函数
import util
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入训练引擎模块
from engine import trainer
# 导入操作系统接口
import os
# 导入CSV文件处理库
import csv
# 导入pandas数据处理库
import pandas as pd
# 导入日志记录库
import logging
# 导入系统相关库
import sys
# 导入日期时间处理库
from datetime import datetime
# 导入Weights & Biases实验跟踪库
import wandb

# === 新增：需要用到的PyTorch函数模块 ===
# 导入PyTorch神经网络函数库（用于激活函数、池化等操作）
import torch.nn.functional as F

# === 日志配置模块 ===
def setup_logging(log_dir="./logs", log_level=logging.INFO):
    """
    设置日志记录系统，同时输出到控制台和文件
    参数:
        log_dir: 日志文件保存目录，默认为"./logs"
        log_level: 日志级别，默认为INFO级别
    返回:
        logger: 配置好的日志记录器对象
        log_filename: 生成的日志文件完整路径
    """
    # 创建日志目录，如果目录不存在则创建，exist_ok=True表示目录已存在时不报错
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带时间戳的日志文件名，格式为：training_YYYYMMDD_HHMMSS.log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 定义日志输出格式：时间戳 - 日志级别 - 消息内容
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    # 定义时间戳格式：年-月-日 时:分:秒
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志记录器，设置日志级别、格式和处理器
    logging.basicConfig(
        level=log_level,                    # 设置日志记录级别
        format=log_format,                  # 设置日志输出格式
        datefmt=date_format,                # 设置时间戳格式
        handlers=[                          # 设置日志处理器列表
            logging.FileHandler(log_filename, encoding='utf-8'),  # 文件处理器，输出到日志文件
            logging.StreamHandler(sys.stdout)                     # 控制台处理器，输出到标准输出
        ]
    )
    
    # 获取当前模块的日志记录器
    logger = logging.getLogger(__name__)
    # 记录日志系统启动信息
    logger.info(f"日志记录已启动，日志文件: {log_filename}")
    # 返回日志记录器和日志文件路径
    return logger, log_filename

# === Weights & Biases实验跟踪配置模块 ===
def setup_wandb(args, project_name="Graph-WaveNet"):
    """
    初始化Weights & Biases实验跟踪系统
    参数:
        args: 命令行参数对象，包含所有实验配置
        project_name: W&B项目名称，默认为"Graph-WaveNet"
    返回:
        run_name: 生成的实验运行名称
    """
    # 生成带时间戳的实验运行名称，格式：数据集名_YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(args.data)  # 从数据路径中提取数据集名称
    run_name = f"{dataset_name}_{timestamp}"
    
    # 构建W&B配置字典，包含所有实验超参数
    config = {
        # 基础数据配置
        "data": args.data,                    # 数据集路径
        "device": args.device,                # 计算设备（CPU/GPU）
        "batch_size": args.batch_size,        # 批次大小
        "epochs": args.epochs,                # 训练轮数
        "seq_length": args.seq_length,        # 输入序列长度
        "pred_length": args.pred_length,      # 预测序列长度
        "learning_rate": args.learning_rate,  # 学习率
        "dropout": args.dropout,              # Dropout概率
        "nhid": args.nhid,                    # 隐藏层维度
        "weight_decay": args.weight_decay,    # 权重衰减系数
        "print_every": args.print_every,      # 打印频率
        # 图神经网络配置
        "gcn_bool": args.gcn_bool,            # 是否使用图卷积
        "addaptadj": args.addaptadj,          # 是否添加自适应邻接矩阵
        "randomadj": args.randomadj,          # 是否随机初始化邻接矩阵
        "adjtype": args.adjtype,              # 邻接矩阵类型
        "num_nodes": args.num_nodes,          # 节点数量
        "in_dim": args.in_dim,                # 输入特征维度
        # 早停机制配置
        "early_stop": args.early_stop,                    # 是否启用早停
        "early_stop_patience": args.early_stop_patience,  # 早停耐心值
        "early_stop_min_delta": args.early_stop_min_delta, # 早停最小改善阈值
        "early_stop_monitor": args.early_stop_monitor,     # 早停监控指标
        "early_stop_mode": args.early_stop_mode,           # 早停模式（min/max）
        # 数据增强配置
        "enhance": args.enhance,              # 增强模式（none/series/graph/both）
        "series_kernel": args.series_kernel,  # 时间序列分解核大小
        "graph_mode": args.graph_mode,        # 图滤波模式
        "graph_alpha": args.graph_alpha,      # 图滤波强度
        # 图卷积变体配置
        "diag_mode": args.diag_mode,          # 对角连接模式
        "use_power": args.use_power,          # 是否使用幂律传播
        "power_order": args.power_order,      # 幂律阶数
        "power_init": args.power_init,        # 幂律系数初始化方式
        "use_cheby": args.use_cheby,          # 是否使用Chebyshev卷积
        "cheby_k": args.cheby_k,              # Chebyshev阶数
        "use_mixprop": args.use_mixprop,      # 是否使用MixPropDual
        "mixprop_k": args.mixprop_k,          # MixPropDual递推步长
        "adj_dropout": args.adj_dropout,      # 邻接矩阵dropout率
        "adj_temp": args.adj_temp,            # 邻接矩阵温度参数
        "use_powermix": args.use_powermix,    # 是否使用PowerMixDual
        "powermix_k": args.powermix_k,        # PowerMixDual递推步长
        "powermix_dropout": args.powermix_dropout, # PowerMixDual dropout率
        "powermix_temp": args.powermix_temp,  # PowerMixDual温度参数
        "powermix_embed_init": args.powermix_embed_init, # PowerMixDual嵌入初始化方式
        "powermix_emb_dim": args.powermix_emb_dim  # PowerMixDual嵌入维度
    }
    
    # 初始化W&B实验跟踪
    wandb.init(
        project=project_name,                 # 设置项目名称
        name=run_name,                        # 设置运行名称
        config=config,                        # 传入配置字典
        tags=[dataset_name, args.enhance, "Graph-WaveNet"]  # 设置标签用于分类
    )
    
    # 返回生成的运行名称
    return run_name

def _series_decomp_bcnt(x, kernel_size):
    """
    时间域分解函数：使用滑动平均对时间序列进行趋势-季节性分解
    参数:
        x: 输入张量，形状为 [B, C, N, T] (批次, 通道, 节点, 时间)
        kernel_size: 滑动平均的核大小（必须是奇数）
    返回:
        seasonal: 季节性分量，形状与输入相同
        trend: 趋势分量，形状与输入相同
    原理: 通过滑动平均提取趋势，季节性分量 = 原始数据 - 趋势分量
    """
    # 获取输入张量的维度信息
    B, C, N, T = x.shape  # B=批次大小, C=通道数, N=节点数, T=时间步数
    
    # 计算填充大小，确保滑动平均后时间维度不变
    pad = (kernel_size - 1) // 2  # 左右各填充 (kernel_size-1)/2 个点
    
    # 将4D张量重塑为3D，便于使用1D平均池化
    # 从 [B, C, N, T] 变为 [B*C*N, 1, T]
    x1 = x.permute(0, 1, 2, 3).contiguous().view(B * C * N, 1, T)
    
    # 对时间维度进行边界填充，使用复制模式保持边界值
    xpad = F.pad(x1, (pad, pad), mode='replicate')
    
    # 使用1D平均池化进行滑动平均，提取趋势分量
    trend = F.avg_pool1d(xpad, kernel_size=kernel_size, stride=1)
    
    # 将趋势分量重塑回原始形状 [B, C, N, T]
    trend = trend.view(B, C, N, T)
    
    # 计算季节性分量：原始数据减去趋势分量
    seasonal = x - trend
    
    # 返回季节性分量和趋势分量
    return seasonal, trend

def _graph_filter_on_input(btnc, A, alpha=0.5, mode='lowpass'):
    """
    图域滤波函数：在节点维度进行邻接矩阵传播，实现图信号滤波
    参数:
        btnc: 输入张量，形状为 [B, T, N, C] (批次, 时间, 节点, 通道)
        A: 邻接矩阵，形状为 [N, N]，表示节点间的连接关系
        alpha: 滤波强度参数，控制邻接传播的权重
        mode: 滤波模式，'lowpass'为低通滤波，'highpass'为高通滤波
    返回:
        Xf: 滤波后的张量，形状与输入相同 [B, T, N, C]
    原理: 通过邻接矩阵传播实现图信号的低通或高通滤波
    """
    # 如果邻接矩阵为空，直接返回原始输入
    if A is None:
        return btnc
    
    # 获取输入张量的维度信息
    B, T, N, C = btnc.shape  # B=批次, T=时间, N=节点, C=通道
    
    # 重塑张量以便进行矩阵乘法
    # 将 (B, T, C) 维度合并作为batch维度，节点维度单独处理
    # 从 [B, T, N, C] 变为 [B*T*C, N]
    X = btnc.permute(0, 1, 3, 2).contiguous().view(B * T * C, N)
    
    # 根据滤波模式进行不同的图信号处理
    if mode == 'lowpass':
        # 低通滤波：保留低频信息，平滑信号
        # 公式：Xf = (X + α * X * A^T) / (1 + α)
        Xf = (X + alpha * (X @ A.t())) / (1.0 + alpha)
    elif mode == 'highpass':
        # 高通滤波：保留高频信息，突出变化
        # 公式：Xf = X - α * X * A^T
        Xf = X - alpha * (X @ A.t())
    else:
        # 其他模式：不进行滤波，直接返回
        Xf = X
    
    # 将滤波结果重塑回原始形状
    # 从 [B*T*C, N] 变回 [B, T, N, C]
    Xf = Xf.view(B, T, C, N).permute(0, 1, 3, 2).contiguous()
    
    # 返回滤波后的张量
    return Xf
# === 数据增强函数模块结束 ===

# === 命令行参数解析配置 ===
# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Graph-WaveNet训练脚本')

# === 基础配置参数 ===
# 计算设备选择（CPU或GPU）
parser.add_argument('--device', type=str, default='cuda:0', help='计算设备，如cuda:0或cpu')
# 数据集路径
parser.add_argument('--data', type=str, default='data/METR-LA', help='数据集路径')
# 邻接矩阵数据路径
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='邻接矩阵数据文件路径')
# 邻接矩阵类型
parser.add_argument('--adjtype', type=str, default='doubletransition', help='邻接矩阵类型（doubletransition等）')

# === 图神经网络配置参数 ===
# 是否使用图卷积层
parser.add_argument('--gcn_bool', action='store_true', help='是否添加图卷积层')
# 是否仅使用自适应邻接矩阵
parser.add_argument('--aptonly', action='store_true', help='是否仅使用自适应邻接矩阵')
# 是否添加自适应邻接矩阵
parser.add_argument('--addaptadj', action='store_true', help='是否添加自适应邻接矩阵')
# 是否随机初始化自适应邻接矩阵
parser.add_argument('--randomadj', action='store_true', help='是否随机初始化自适应邻接矩阵')

# === 序列长度配置参数 ===
# 输入序列长度（历史时间步数）
parser.add_argument('--seq_length', type=int, default=96, help='输入序列长度（历史时间步数）')
# 预测序列长度（未来时间步数）
parser.add_argument('--pred_length', type=int, default=96, help='预测序列长度（输出序列长度）')

# === 模型结构配置参数 ===
# 隐藏层维度
parser.add_argument('--nhid', type=int, default=32, help='隐藏层维度大小')
# 输入特征维度
parser.add_argument('--in_dim', type=int, default=2, help='输入特征维度')
# 图节点数量
parser.add_argument('--num_nodes', type=int, default=207, help='图节点数量')

# === 训练配置参数 ===
# 批次大小
parser.add_argument('--batch_size', type=int, default=4, help='训练批次大小')
# 学习率
parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
# Dropout概率
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout概率')
# 权重衰减系数
parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减系数')
# 训练轮数
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
# 打印频率
parser.add_argument('--print_every', type=int, default=50, help='训练过程中打印信息的频率')

# === 实验配置参数 ===
# 随机种子（已注释）
#parser.add_argument('--seed', type=int, default=99, help='随机种子')
# 模型保存路径
parser.add_argument('--save', type=str, default='./garage/metr', help='模型保存路径')
# 实验ID
parser.add_argument('--expid', type=int, default=1, help='实验ID')
# 是否运行多序列长度实验
parser.add_argument('--run_multiple_experiments', action='store_true', help='是否运行不同序列长度的实验')

# === 日志和实验跟踪相关参数 ===
# 日志文件保存目录
parser.add_argument('--log_dir', type=str, default='./logs', help='日志文件保存目录')
# 是否使用Weights & Biases记录实验
parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb记录实验')
# W&B项目名称
parser.add_argument('--wandb_project', type=str, default='Graph-WaveNet-france2', help='wandb项目名称')
# 日志记录级别
parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='日志级别')

# === 早停机制相关参数 ===
# 启用早停功能（默认启用）
parser.add_argument('--early_stop', action='store_true', default=True,
                    help='启用早停功能 (默认启用)')
# 禁用早停功能
parser.add_argument('--no_early_stop', action='store_true', 
                    help='禁用早停功能')
# 早停耐心值：连续多少个epoch无改善则停止
parser.add_argument('--early_stop_patience', type=int, default=10,
                    help='若验证集 loss 连续 patience 个 epoch 无显著下降则提前停止 (默认: 10)')
# 早停最小改善阈值：判断是否有显著改善的最小差值
parser.add_argument('--early_stop_min_delta', type=float, default=0.0001,
                    help='判定"显著下降"的阈值 (new_best < best - min_delta) (默认: 0.0001)')
# 早停监控指标：选择监控哪个指标进行早停判断
parser.add_argument('--early_stop_monitor', type=str, default='val_loss',
                    choices=['val_loss', 'val_mape', 'val_rmse'],
                    help='早停监控的指标 (默认: val_loss)')
# 早停模式：min表示越小越好，max表示越大越好
parser.add_argument('--early_stop_mode', type=str, default='min',
                    choices=['min', 'max'],
                    help='早停模式: min表示越小越好, max表示越大越好 (默认: min)')


# === 数据增强模块相关参数 ===
# 数据增强模式选择
parser.add_argument('--enhance', type=str, default='none',
                    choices=['none', 'series', 'graph', 'both'],
                    help='数据增强模块: none(无增强), series(时间域), graph(图域), both(两者都使用)')
# 时间序列分解的核大小
parser.add_argument('--series_kernel', type=int, default=25,
                    help='时间序列分解的核大小 (建议使用奇数)')
# 图滤波模式
parser.add_argument('--graph_mode', type=str, default='lowpass',
                    choices=['lowpass', 'highpass', 'none'],
                    help='图滤波模式: lowpass(低通滤波), highpass(高通滤波), none(无滤波)')
# 图滤波强度参数
parser.add_argument('--graph_alpha', type=float, default=0.5,
                    help='图滤波强度参数alpha，控制邻接传播的权重')

# === 图卷积变体相关参数 ===
# 是否启用幂律传播
parser.add_argument("--use_power", action="store_true", help="启用 PowerLaw 幂律传播")
# 是否启用Chebyshev谱域卷积
parser.add_argument("--use_cheby", action="store_true", help="启用 Chebyshev 谱域传播")
# 是否启用MixPropDual双图递推
parser.add_argument("--use_mixprop", action="store_true", help="启用 MixPropDual 双图递推")
# 是否启用PowerMixDual幂律双图递推
parser.add_argument("--use_powermix", action="store_true", help="启用 PowerMixDual 幂律双图递推")

# === 共享结构相关参数 ===
# 对角连接模式：是否包含自环
parser.add_argument("--diag_mode", type=str, default="self_and_neighbor",
                    choices=["self_and_neighbor", "neighbor"], help="对角连接模式: self_and_neighbor(包含自环), neighbor(仅邻居)")

# === PowerLaw幂律传播专用参数 ===
# 幂律传播的最大阶数
parser.add_argument("--power_order", type=int, default=3, help="PowerLaw 幂律传播的最大阶数")
# 幂律系数的初始化策略
parser.add_argument("--power_init", type=str, default="plain",
                    choices=["plain", "decay", "softmax"], help="幂律系数初始化策略: plain(全1), decay(指数衰减), softmax(随机softmax)")

# === Chebyshev谱域卷积专用参数 ===
# Chebyshev多项式的阶数
parser.add_argument("--cheby_k", type=int, default=3, help="Chebyshev 多项式的K阶数")

# === MixPropDual双图递推专用参数 ===
# MixPropDual的递推步长
parser.add_argument("--mixprop_k", type=int, default=3, help="MixPropDual 双图递推的步长")
# 邻接矩阵的dropout率
parser.add_argument("--adj_dropout", type=float, default=0.1, help="邻接矩阵的dropout概率")
# 邻接矩阵的温度参数
parser.add_argument("--adj_temp", type=float, default=1.0, help="邻接矩阵的温度参数，控制softmax的锐度")

# === PowerMixDual幂律双图递推专用参数 ===
# PowerMixDual的递推步长
parser.add_argument("--powermix_k", type=int, default=3, help="PowerMixDual 幂律双图递推的步长")
# PowerMixDual的dropout率
parser.add_argument("--powermix_dropout", type=float, default=0.3, help="PowerMixDual 的A-dropout概率")
# PowerMixDual的温度参数
parser.add_argument("--powermix_temp", type=float, default=1.0, help="PowerMixDual 的温度参数")
# PowerMixDual的嵌入初始化方式
parser.add_argument("--powermix_embed_init", type=str, default="xavier", 
                    choices=["xavier", "normal"], help="PowerMixDual 嵌入层初始化方式")
# PowerMixDual的嵌入维度
parser.add_argument("--powermix_emb_dim", type=int, default=16, help="PowerMixDual 嵌入层维度")

# 解析所有命令行参数
args = parser.parse_args()

# === 数据集自动检测和参数配置模块 ===

def configure_dataset_params(args, logger=None):
    """
    根据数据路径自动检测数据集类型并配置相应的参数
    参数:
        args: 命令行参数对象，将被修改以适配特定数据集
        logger: 日志记录器，用于输出配置信息
    返回:
        args: 配置后的参数对象
    """
    # 将数据路径转换为大写，便于字符串匹配
    data_path = args.data.upper()

    # === France数据集配置 ===
    if 'FRANCE' in data_path:
        args.num_nodes = 10  # France数据集有10个节点
        args.adjdata = 'data/sensor_graph/adj_mx_france.pkl'  # France邻接矩阵路径
        args.save = './garage/france/'  # France模型保存路径
        if logger:
            logger.info(f"检测到原始France数据集")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # === Germany数据集配置 ===
    elif 'GERMANY' in data_path:
        args.num_nodes = 16  # Germany数据集有16个节点
        args.adjdata = 'data/sensor_graph/adj_mx_germany.pkl'  # Germany邻接矩阵路径
        args.save = './garage/germany/'  # Germany模型保存路径
        if logger:
            logger.info(f"检测到Germany数据集")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # === METR-LA数据集配置 ===
    elif 'METR' in data_path:
        args.num_nodes = 207  # METR-LA数据集有207个节点
        args.adjdata = 'data/sensor_graph/adj_mx.pkl'  # METR-LA邻接矩阵路径
        args.save = './garage/metr/'  # METR-LA模型保存路径
        if logger:
            logger.info(f"检测到METR-LA数据集")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # === PEMS-BAY数据集配置 ===
    elif 'BAY' in data_path:
        args.num_nodes = 325  # PEMS-BAY数据集有325个节点
        args.adjdata = 'data/sensor_graph/adj_mx_bay.pkl'  # PEMS-BAY邻接矩阵路径
        args.save = './garage/bay/'  # PEMS-BAY模型保存路径
        if logger:
            logger.info(f"检测到PEMS-BAY数据集")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # === Traffic数据集配置 ===
    elif 'TRAFFIC' in data_path:
        args.num_nodes = 963  # Traffic数据集有963个节点
        args.adjdata = 'data/sensor_graph/adj_mx_traffic.pkl'  # Traffic邻接矩阵路径
        args.save = './garage/traffic/'  # Traffic模型保存路径
        if logger:
            logger.info(f"检测到Traffic数据集")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # === 合成数据集配置（四套不同难度级别） ===
    # 简单合成数据集
    elif 'SYNTHETIC_EASY' in data_path:
        args.num_nodes = 12  # 简单合成数据集有12个节点
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_easy.pkl'  # 简单合成数据邻接矩阵
        args.save = './garage/synth_easy/'  # 简单合成数据模型保存路径
        if logger:
            logger.info(f"检测到合成数据集：SYNTHETIC_EASY")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # 中等难度合成数据集
    elif 'SYNTHETIC_MEDIUM' in data_path:
        args.num_nodes = 12  # 中等合成数据集有12个节点
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_medium.pkl'  # 中等合成数据邻接矩阵
        args.save = './garage/synth_medium/'  # 中等合成数据模型保存路径
        if logger:
            logger.info(f"检测到合成数据集：SYNTHETIC_MEDIUM")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # 困难合成数据集
    elif 'SYNTHETIC_HARD' in data_path:
        args.num_nodes = 12  # 困难合成数据集有12个节点
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_hard.pkl'  # 困难合成数据邻接矩阵
        args.save = './garage/synth_hard/'  # 困难合成数据模型保存路径
        if logger:
            logger.info(f"检测到合成数据集：SYNTHETIC_HARD")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # 极困难合成数据集
    elif 'SYNTHETIC_VERY_HARD' in data_path:
        args.num_nodes = 12  # 极困难合成数据集有12个节点
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_very_hard.pkl'  # 极困难合成数据邻接矩阵
        args.save = './garage/synth_very_hard/'  # 极困难合成数据模型保存路径
        if logger:
            logger.info(f"检测到合成数据集：SYNTHETIC_VERY_HARD")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
        
    # === 其他数据集配置 ===
    
    # Solar数据集配置
    elif 'SOLAR' in data_path:
        args.num_nodes = 137  # Solar数据集有137个节点
        args.adjdata = 'data/sensor_graph/adj_mx_solar.pkl'  # Solar邻接矩阵路径
        args.save = './garage/solar/'  # Solar模型保存路径
        if logger:
            logger.info(f"检测到Solar数据集")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # Electricity数据集配置
    elif 'ELECTRICITY' in data_path:
        args.num_nodes = 321  # Electricity数据集有321个节点
        args.adjdata = 'data/sensor_graph/adj_mx_electricity.pkl'  # Electricity邻接矩阵路径
        args.save = './garage/electricity/'  # Electricity模型保存路径
        if logger:
            logger.info(f"检测到Electricity数据集")
            logger.info(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # === 未识别数据集的默认处理 ===
    else:
        if logger:
            logger.warning(f"未识别的数据集: {data_path}")
            logger.info(f"使用默认配置: 节点数={args.num_nodes}")

    # 创建模型保存目录，如果目录不存在则创建
    os.makedirs(args.save, exist_ok=True)
    # 返回配置后的参数对象
    return args
    


# === 日志系统和实验跟踪初始化 ===
# 根据命令行参数获取日志级别
log_level = getattr(logging, args.log_level.upper())
# 初始化日志系统，返回日志记录器和日志文件路径
logger, log_file = setup_logging(args.log_dir, log_level)

# 根据数据集类型配置参数（在日志系统初始化后执行）
args = configure_dataset_params(args, logger)

# === 早停机制参数处理 ===
# 检查是否禁用早停功能
if args.no_early_stop:
    args.early_stop = False  # 禁用早停
    logger.info("早停功能已禁用")
else:
    args.early_stop = True  # 启用早停
    logger.info(f"早停功能已启用: patience={args.early_stop_patience}, min_delta={args.early_stop_min_delta}, monitor={args.early_stop_monitor}, mode={args.early_stop_mode}")

# === 实验配置信息记录 ===
# 记录分隔线
logger.info("="*60)
logger.info("实验配置信息")
logger.info("="*60)
# 遍历所有命令行参数并记录到日志
for key, value in vars(args).items():
    logger.info(f"{key}: {value}")
# 记录分隔线
logger.info("="*60)

# === Weights & Biases实验跟踪初始化 ===
# 初始化W&B运行名称变量
wandb_run_name = None
# 如果启用了W&B，则尝试初始化
if args.use_wandb:
    try:
        # 调用W&B设置函数，传入参数和项目名称
        wandb_run_name = setup_wandb(args, args.wandb_project)
        logger.info(f"Wandb已启动，运行名称: {wandb_run_name}")
    except Exception as e:
        # 如果W&B初始化失败，记录警告并禁用W&B
        logger.warning(f"Wandb初始化失败: {e}")
        args.use_wandb = False

def run_experiments_with_different_seq_lengths():
    """
    运行不同序列长度的实验并保存结果到CSV文件
    返回:
        results: 包含所有实验结果的结果列表
    """
    # 定义要测试的序列长度列表
    seq_lengths = [6, 12]
    # 初始化结果存储列表
    results = []
    
    # 记录实验开始信息
    logger.info("Starting experiments with different sequence lengths...")
    logger.info(f"Sequence lengths to test: {seq_lengths}")
    logger.info("Note: pred_length will be set equal to seq_length for each experiment")
    
    # 遍历每个序列长度进行实验
    for seq_len in seq_lengths:
        # 记录当前实验开始信息
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        logger.info(f"{'='*60}")
        
        # 更新参数配置
        args.seq_length = seq_len      # 设置输入序列长度
        args.pred_length = seq_len     # 设置预测序列长度（与输入长度相同）
        args.expid = seq_len           # 设置实验ID为序列长度
        
        # 为当前序列长度生成数据
        logger.info(f"为 seq_length={seq_len}, pred_length={seq_len} 生成数据...")
        generate_data_for_seq_length(seq_len, seq_len)
        
        # 记录实验开始时间
        experiment_start_time = time.time()
        # 运行主实验
        result = main_experiment()
        # 记录实验结束时间
        experiment_end_time = time.time()
        
        # 添加实验元信息到结果中
        result['seq_length'] = seq_len
        result['pred_length'] = seq_len
        result['total_experiment_time'] = experiment_end_time - experiment_start_time
        # 将结果添加到结果列表
        results.append(result)
        
        # 记录当前实验完成信息
        logger.info(f"Completed experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        logger.info(f"Experiment time: {experiment_end_time - experiment_start_time:.4f} seconds")
    
    # 保存所有结果到CSV文件
    save_results_to_csv(results)
    logger.info(f"\nAll experiments completed! Results saved to 'experiment_results.csv'")
    # 返回所有实验结果
    return results

def generate_data_for_seq_length(seq_length, pred_length):
    """
    根据数据集类型为特定序列长度生成数据集
    参数:
        seq_length: 输入序列长度
        pred_length: 预测序列长度
    """
    # 导入子进程和系统模块
    import subprocess
    import sys
    
    # 获取数据路径并转换为大写
    data_path = args.data.upper()
    
    # 根据数据集类型配置相应的处理脚本和数据文件路径
    if 'FRANCE' in data_path:
        dataset_name = 'FRANCE'  # 数据集名称
        process_script = 'process_france_reduced_with_dataloader.py'  # France数据处理脚本
        data_file = f'data/FRANCE_REDUCED/train.npz'  # France训练数据文件路径
    elif 'GERMANY' in data_path:
        dataset_name = 'GERMANY'  # 数据集名称
        process_script = 'process_germany_reduced_with_dataloader.py'  # Germany数据处理脚本
        data_file = f'data/GERMANY_REDUCED/train.npz'  # Germany训练数据文件路径
    elif 'PEMSBAY' in data_path or 'BAY' in data_path:
        dataset_name = 'PEMSBAY'  # 数据集名称
        process_script = None  # PEMSBay数据已经预处理完成，不需要额外处理脚本
        data_file = f'data/PEMSBAY/train.npz'  # PEMSBay训练数据文件路径
    elif 'TRAFFIC' in data_path:
        dataset_name = 'TRAFFIC'  # 数据集名称
        process_script = 'process_traffic_with_dataloader.py'  # Traffic数据处理脚本
        data_file = f'data/TRAFFIC/train.npz'  # Traffic训练数据文件路径
    else:
        # 对于其他数据集类型，使用现有数据
        print(f"新的数据集类型: {data_path}")
        print("使用现有数据")
        return
    
    # 默认需要重新生成数据
    regenerate = True
    
    # 检查数据文件是否已存在
    if os.path.exists(data_file):
        try:
            # 加载现有数据文件
            data = np.load(data_file)
            # 获取现有数据的序列长度
            existing_seq_len = data['x'].shape[1]
            # 检查序列长度是否匹配
            if existing_seq_len == seq_length:
                logger.info(f"{dataset_name}数据已存在且序列长度匹配 (seq_length={seq_length})")
                regenerate = False  # 不需要重新生成
            else:
                logger.info(f"现有{dataset_name}数据序列长度不匹配 ({existing_seq_len} != {seq_length})，重新生成...")
        except Exception as e:
            # 如果检查现有数据时出错，记录警告
            logger.warning(f"检查现有{dataset_name}数据时出错: {e}")
    
    # 如果需要重新生成数据
    if regenerate:
        if process_script is None:
            # 对于PEMSBay等已经预处理的数据集，跳过重新生成
            logger.info(f"{dataset_name}数据已存在，跳过重新生成")
        else:
            logger.info(f"生成{dataset_name}数据: seq_length={seq_length}, pred_length={pred_length}...")
            # 构建子进程命令
            cmd = [
                sys.executable, process_script,  # Python解释器路径
                '--step', 'process',             # 处理步骤
                '--seq_length', str(seq_length), # 输入序列长度
                '--pred_length', str(pred_length) # 预测序列长度
            ]
            
            try:
                # 运行数据处理脚本
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"{dataset_name}数据生成完成")
                # 记录脚本输出的最后5行
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[-5:]:
                    if line.strip():
                        logger.info(f"  {line}")
            except subprocess.CalledProcessError as e:
                # 如果数据处理失败，记录错误并抛出异常
                logger.error(f"{dataset_name}数据生成失败: {e}")
                logger.error(f"错误输出: {e.stderr}")
                raise

def main_experiment():
    """
    主实验函数：执行完整的训练和测试流程，返回测试结果
    返回:
        dict: 包含验证损失、训练时间、测试指标等结果的字典
    """
    # === 设备和数据初始化 ===
    # 设置计算设备（CPU或GPU）
    device = torch.device(args.device)
    # 加载邻接矩阵数据：传感器ID、ID到索引映射、邻接矩阵
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    # 加载数据集：训练、验证、测试数据加载器
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    # 获取数据标准化器
    scaler = dataloader['scaler']
    # 将邻接矩阵转换为张量并移动到指定设备
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    # 记录训练参数信息
    logger.info(f"训练参数: {args}")

    # === 邻接矩阵初始化配置 ===
    # 如果使用随机邻接矩阵初始化
    if args.randomadj:
        adjinit = None  # 不提供初始邻接矩阵
    else:
        adjinit = supports[0]  # 使用第一个支持矩阵作为初始邻接矩阵
    
    # 自适应邻接矩阵初始化
    # 如果使用随机邻接矩阵，则使用随机初始化
    # 否则使用第一个支持矩阵进行SVD初始化
    if args.randomadj:
        aptinit = None  # 使用随机初始化
    else:
        aptinit = supports[0] if supports is not None and len(supports) > 0 else None  # 使用SVD初始化
    
    # 如果仅使用自适应邻接矩阵
    if args.aptonly:
        supports = None  # 不使用预定义的邻接矩阵
        
    # === 训练器初始化 ===
    # 原始训练器初始化代码（已注释，保留作为参考）
    ######################################
    # engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
    #                      args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
    #                      adjinit, pred_length=args.pred_length)
    ######################################
    
    # 创建训练器实例，传入所有配置参数
    engine = trainer(
        # 基础参数
        scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
        args.learning_rate, args.weight_decay, args.device, supports, args.gcn_bool,
        args.addaptadj, aptinit, pred_length=args.pred_length,
        # 图卷积变体参数
        diag_mode=args.diag_mode,                    # 对角连接模式
        use_power=args.use_power, power_order=args.power_order, power_init=args.power_init,  # 幂律传播参数
        use_cheby=args.use_cheby, cheby_k=args.cheby_k,  # Chebyshev卷积参数
        use_mixprop=args.use_mixprop, mixprop_k=args.mixprop_k,  # MixPropDual参数
        adj_dropout=args.adj_dropout, adj_temp=args.adj_temp,  # 邻接矩阵参数
        use_powermix=args.use_powermix, powermix_k=args.powermix_k,  # PowerMixDual参数
        powermix_dropout=args.powermix_dropout, powermix_temp=args.powermix_temp,  # PowerMixDual参数
        powermix_embed_init=args.powermix_embed_init, powermix_emb_dim=args.powermix_emb_dim  # PowerMixDual嵌入参数
    )

    # 记录训练开始信息
    logger.info("开始训练...")
    
    # === Weights & Biases模型监控 ===
    # 如果启用了W&B，则监控模型参数和梯度
    if args.use_wandb:
        wandb.watch(engine.model, log="all", log_freq=100)

    # === 训练过程记录变量初始化 ===
    his_loss = []      # 历史验证损失列表
    val_time = []      # 验证时间列表
    train_time = []    # 训练时间列表
    
    # === 早停相关变量 ===
    if args.early_stop:
        if args.early_stop_mode == 'min':
            best_val = float('inf')
        else:  # max
            best_val = float('-inf')
        epochs_no_improve = 0
        best_epoch = 0
        logger.info(f"早停监控指标: {args.early_stop_monitor}, 模式: {args.early_stop_mode}")
    else:
        best_val = float('inf')  # 保持兼容性
        epochs_no_improve = 0
    
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            # x: numpy [B, T, N, C]
            # === NEW: 图域增强（输入前，沿节点维）===
            if args.enhance in ['graph', 'both'] and (supports is not None) and (len(supports) > 0):
                x_t = torch.Tensor(x).to(device)  # [B, T, N, C]
                x_t = _graph_filter_on_input(x_t, supports[0], alpha=args.graph_alpha, mode=args.graph_mode)
                x = x_t.detach().cpu().numpy()

            trainx = torch.Tensor(x).to(device)          # [B, T, N, C]
            trainx = trainx.transpose(1, 3)              # -> [B, C, N, T]

            # === NEW: 时间域增强（在 time 维做平滑）===
            if args.enhance in ['series', 'both']:
                seasonal, trend = _series_decomp_bcnt(trainx, kernel_size=args.series_kernel)
                # 这里默认用 seasonal + trend_scale*trend（可调），简单起见用1.0（等于原数据）。可改成只 seasonal。
                trainx = seasonal + trend  # 等价原数据；要加强滤波可改成：trainx = seasonal + 0.5*trend

            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)              # [B, C, N, T]
            # metrics = engine.train(trainx, trainy[:,0,:,:])  # 目标仍用原第一特征
            metrics = engine.train(trainx, trainy[:, 0, :, :args.pred_length])

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log_msg = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                logger.info(log_msg.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))
                
                # 记录到wandb
                if args.use_wandb:
                    wandb.log({
                        "train/loss": train_loss[-1],
                        "train/mape": train_mape[-1],
                        "train/rmse": train_rmse[-1],
                        "epoch": i,
                        "iteration": iter
                    })
        t2 = time.time()
        train_time.append(t2-t1)

        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            # === NEW: 图域增强（验证）===
            if args.enhance in ['graph', 'both'] and (supports is not None) and (len(supports) > 0):
                x_t = torch.Tensor(x).to(device)
                x_t = _graph_filter_on_input(x_t, supports[0], alpha=args.graph_alpha, mode=args.graph_mode)
                x = x_t.detach().cpu().numpy()

            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)  # [B, C, N, T]

            # === NEW: 时间域增强（验证）===
            if args.enhance in ['series', 'both']:
                seasonal, trend = _series_decomp_bcnt(testx, kernel_size=args.series_kernel)
                testx = seasonal + trend

            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            #metrics = engine.eval(testx, testy[:,0,:,:]) ##########################
            metrics = engine.eval(testx, testy[:,0,:,:args.pred_length])

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log_msg = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        logger.info(log_msg.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log_msg = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        logger.info(log_msg.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))
        
        # 记录到wandb
        if args.use_wandb:
            wandb_log = {
                "epoch": i,
                "train/loss_epoch": mtrain_loss,
                "train/mape_epoch": mtrain_mape,
                "train/rmse_epoch": mtrain_rmse,
                "val/loss": mvalid_loss,
                "val/mape": mvalid_mape,
                "val/rmse": mvalid_rmse,
                "time/train_time": t2 - t1,
                "time/val_time": s2 - s1
            }
            
            # 添加早停相关信息
            if args.early_stop:
                wandb_log.update({
                    "early_stop/epochs_no_improve": epochs_no_improve,
                    "early_stop/best_val": best_val,
                    "early_stop/best_epoch": best_epoch
                })
            
            wandb.log(wandb_log)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        
        # === EARLY STOPPING ===
        if args.early_stop:
            # 根据监控指标选择当前值
            if args.early_stop_monitor == 'val_loss':
                current_val = mvalid_loss
            elif args.early_stop_monitor == 'val_mape':
                current_val = mvalid_mape
            elif args.early_stop_monitor == 'val_rmse':
                current_val = mvalid_rmse
            else:
                current_val = mvalid_loss  # 默认
            
            # 判断是否有改善
            improved = False
            if args.early_stop_mode == 'min':
                if current_val < best_val - args.early_stop_min_delta:
                    improved = True
            else:  # max
                if current_val > best_val + args.early_stop_min_delta:
                    improved = True
            
            if improved:
                best_val = current_val
                best_epoch = i
                epochs_no_improve = 0
                logger.info(f"Epoch {i}: {args.early_stop_monitor} 改善至 {best_val:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Epoch {i}: {args.early_stop_monitor} 无改善 ({current_val:.4f}), 已连续 {epochs_no_improve} 个epoch")

            # 检查是否应该早停
            if epochs_no_improve >= args.early_stop_patience:
                logger.info(f"早停触发! 在第 {i} 个epoch停止训练.")
                logger.info(f"最佳 {args.early_stop_monitor}: {best_val:.4f} (第 {best_epoch} 个epoch)")
                logger.info(f"连续 {epochs_no_improve} 个epoch无改善，超过patience={args.early_stop_patience}")
                break
        # === END EARLY STOPPING ===

    logger.info("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    logger.info("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    outputs = []
    #realy = torch.Tensor(dataloader['y_test']).to(device)
    #realy = realy.transpose(1,3)[:,0,:,:]
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:args.pred_length]


    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        try:
            # === NEW: 图域增强（测试）===
            if args.enhance in ['graph', 'both'] and (supports is not None) and (len(supports) > 0):
                x_t = torch.Tensor(x).to(device)
                x_t = _graph_filter_on_input(x_t, supports[0], alpha=args.graph_alpha, mode=args.graph_mode)
                x = x_t.detach().cpu().numpy()

            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)  # [B, C, N, T]

            # === NEW: 时间域增强（测试）===
            if args.enhance in ['series', 'both']:
                seasonal, trend = _series_decomp_bcnt(testx, kernel_size=args.series_kernel)
                testx = seasonal + trend

            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            if len(preds.shape) == 4:
                preds = preds[:, 0, :, :]
            outputs.append(preds)

            del testx, preds
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                if args.batch_size > 1:
                    args.batch_size = args.batch_size // 2
                    logger.warning(f"显存不足，尝试减小 batch_size 至 {args.batch_size}")
                logger.warning(f"第 {iter} 批次推理时显存不足，跳过该批次")
                torch.cuda.empty_cache()
            else:
                raise e
        
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    logger.info("Training finished")
    logger.info("The valid loss on best model is %s", str(round(his_loss[bestid],4)))
    
    amae = []
    amape = []
    armse = []
    horizon_results = []
    
    for i in range(args.pred_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log_msg = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        logger.info(log_msg.format(i+1, metrics[0], metrics[1], metrics[2]))
        
        # 记录到wandb
        if args.use_wandb:
            wandb.log({
                f"test/horizon_{i+1}_mae": metrics[0],
                f"test/horizon_{i+1}_mape": metrics[1],
                f"test/horizon_{i+1}_rmse": metrics[2]
            })
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        
        horizon_results.append({
            'horizon': i+1,
            'mae': metrics[0],
            'mape': metrics[1],
            'rmse': metrics[2]
        })

    avg_mae = np.mean(amae)
    avg_mape = np.mean(amape)
    avg_rmse = np.mean(armse)
    
    log_msg = 'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    logger.info(log_msg.format(args.pred_length, avg_mae, avg_mape, avg_rmse))
    
    # 记录最终结果到wandb
    if args.use_wandb:
        wandb.log({
            "test/mae_avg": avg_mae,
            "test/mape_avg": avg_mape,
            "test/rmse_avg": avg_rmse,
            "best_val_loss": his_loss[bestid]
        })
        wandb.finish()
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")

    return {
        'valid_loss_best': his_loss[bestid],
        'avg_train_time_per_epoch': np.mean(train_time),
        'avg_inference_time': np.mean(val_time),
        'test_mae_avg': avg_mae,
        'test_mape_avg': avg_mape,
        'test_rmse_avg': avg_rmse,
        'horizon_results': horizon_results
    }

def save_results_to_csv(results):
    """
    Save experiment results to CSV file
    """
    csv_data = []
    for result in results:
        row = {
            'seq_length': result['seq_length'],
            'pred_length': result['pred_length'],
            'total_experiment_time': result['total_experiment_time'],
            'valid_loss_best': result['valid_loss_best'],
            'avg_train_time_per_epoch': result['avg_train_time_per_epoch'],
            'avg_inference_time': result['avg_inference_time'],
            'test_mae_avg': result['test_mae_avg'],
            'test_mape_avg': result['test_mape_avg'],
            'test_rmse_avg': result['test_rmse_avg']
        }
        for horizon_result in result['horizon_results']:
            row[f'horizon_{horizon_result["horizon"]}_mae'] = horizon_result['mae']
            row[f'horizon_{horizon_result["horizon"]}_mape'] = horizon_result['mape']
            row[f'horizon_{horizon_result["horizon"]}_rmse'] = horizon_result['rmse']
        csv_data.append(row)
    df = pd.DataFrame(csv_data)
    df.to_csv('experiment_results.csv', index=False)
    logger.info(f"\nResults saved to 'experiment_results.csv'")
    logger.info(f"Columns saved: {list(df.columns)}")
    
    summary_data = []
    for result in results:
        summary_data.append({
            'seq_length': result['seq_length'],
            'pred_length': result['pred_length'],
            'total_time_hours': result['total_experiment_time'] / 3600,
            'valid_loss': result['valid_loss_best'],
            'test_mae': result['test_mae_avg'],
            'test_mape': result['test_mape_avg'],
            'test_rmse': result['test_rmse_avg']
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('experiment_summary.csv', index=False)
    logger.info(f"Summary saved to 'experiment_summary.csv'")

if __name__ == "__main__":
    total_start_time = time.time()
    
    if args.run_multiple_experiments:
        logger.info("运行多个序列长度实验...")
        results = run_experiments_with_different_seq_lengths()
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        logger.info(f"\nAll experiments completed!")
        logger.info(f"Total time for all experiments: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        logger.info(f"\n{'='*60}")
        logger.info("EXPERIMENT SUMMARY")
        logger.info(f"{'='*60}")
        for result in results:
            logger.info(f"Seq Length {result['seq_length']:2d}, Pred Length {result['pred_length']:2d}: MAE={result['test_mae_avg']:.4f}, MAPE={result['test_mape_avg']:.4f}, RMSE={result['test_rmse_avg']:.4f}")
    else:
        logger.info("运行单个实验...")
        logger.info(f"配置: seq_length={args.seq_length}, pred_length={args.pred_length}, enhance={args.enhance}")
        logger.info(f"为 seq_length={args.seq_length}, pred_length={args.pred_length} 生成数据...")
        generate_data_for_seq_length(args.seq_length, args.pred_length)
        result = main_experiment()
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        logger.info(f"\nExperiment completed!")
        logger.info(f"Total time: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        logger.info(f"Results: MAE={result['test_mae_avg']:.4f}, MAPE={result['test_mape_avg']:.4f}, RMSE={result['test_rmse_avg']:.4f}")
        
        # 记录最终日志文件路径
        logger.info(f"日志文件已保存到: {log_file}")


