# Wentian Inference

Wentian 的 Kunpeng 920F 推理实现。仓库包含完整模型、FP32/FP64 NUMA 并行运行时、
ERA5 预处理常量和 15 天自回归预测入口。两个精度均可用一条命令提交到 608 核
`kp920F` 节点，不需要手动配置分块、线程、检查点或输出路径。

## Repository layout

```text
.
├── run_fp32.sh                 # 一键提交 FP32 预测
├── run_fp64.sh                 # 一键提交 FP64 预测
├── scripts/
│   ├── submit_920f.sh          # 参数检查与 Slurm 提交
│   ├── run_920f.sbatch         # 920F 资源申请
│   ├── run_920f_job.sh         # 节点内执行入口
│   ├── check_model.py          # 严格加载检查点
│   └── verify_artifacts.py     # 权重和常量校验
├── src/wentian/
│   ├── data/                   # ERA5 输入预处理
│   ├── model/                  # Wentian 网络定义
│   ├── runtime/                # 分块、Halo、NUMA/HBM 并行运行时
│   ├── forecast.py             # 60 步自回归预测
│   └── precision.py            # FP32/FP64 类型选择
├── tests/                      # 无需 920F 的仓库接口测试
└── weights/wentian_beta.pth.gz # Git LFS 压缩检查点
```

## Environment

目标环境为 Linux AArch64、单个 Kunpeng 920F 节点、608 个 CPU 核和 Slurm。
请先激活适配 SVE2 的 PyTorch 2.8 环境，再安装仓库依赖：

```bash
git lfs install
git clone https://github.com/lqzzy/wentian-inference.git
cd wentian-inference
git lfs pull
python3 -m pip install -e .
```

首次推理会自动将压缩检查点恢复为 `weights/wentian_beta.pth` 并核对 SHA256；恢复后的
文件已被 Git 忽略，后续运行不会重复解压。

通用 PyPI PyTorch 可以检查模型功能，但不能复现 920F 性能；性能运行应使用目标平台提供的
SVE2 构建。仓库不会写入或依赖个人目录。

## Input data

输入目录需要包含起始时刻及其前 6 小时的数据：

```text
input/
├── 20201231_18/
│   ├── 2020123118_plevel.pt
│   └── 2020123118_surface.pt
└── 20210101_00/
    ├── 2021010100_plevel.pt
    └── 2021010100_surface.pt
```

原始张量形状为：

- pressure level: `(8, 13, 721, 1440)`
- surface: `(7, 721, 1440)`

## One-command inference

FP32：

```bash
./run_fp32.sh /path/to/input 2021010100
```

FP64：

```bash
./run_fp64.sh /path/to/input 2021010100
```

脚本会自动申请一个独占 `kp920F` 节点并等待作业结束。每次运行固定执行 60 个 6 小时
预测步，即 15 天。输出分别写入：

```text
outputs/fp32/2021010100_pred/
outputs/fp64/2021010100_pred/
```

每一步生成一个物理量空间的 `.pt` 文件，目录中的 `manifest.json` 记录每步耗时、精度、
起止时间和检查点位置。Slurm 日志位于 `outputs/logs/`。

如果已经处于 Slurm 分配的 920F 节点中，同样的命令会直接运行，不会再次提交作业。

## Automatic runtime profiles

公开入口只保留精度、输入目录和起始时间三个必要信息。其余参数由精度配置自动确定：

| Precision | Spatial partitions | Workers | Threads per worker |
|---|---:|---:|---:|
| FP32 | 4 × 4 | 16 | 38 |
| FP64 | 3 × 5 | 15 | 38 |

运行时对空间块使用 Halo 区域完成局部卷积，并以 owner-computes-once 方式执行窗口注意力；
共享帧位于 DDR，私有热数据优先放置在相邻 HBM。

## Verification

不启动完整推理即可检查接口、文件布局、脚本和精度配置：

```bash
make test
make verify
```

在已安装 PyTorch 且权重已下载的环境中，可严格构建模型并加载全部参数：

```bash
make model-check
```

检查点信息和模型限制见 [MODEL_CARD.md](MODEL_CARD.md)。

## License

代码、随仓库发布的预处理常量及 Wentian 检查点采用
[Apache License 2.0](LICENSE)。第三方依赖保留各自许可证。
