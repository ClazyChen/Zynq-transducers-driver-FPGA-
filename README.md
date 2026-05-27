# Zynq FPGA 超声换能器阵列驱动

8×8 超声换能器（40 kHz）FPGA 驱动，通过 8 路独立 SER 控制 8 片 74HC595 移位寄存器。

## 项目结构

```
├── src/main/scala/          # FPGA 固件（Chisel）
├── host/                    # ★ 上位机（PC）控制软件
│   ├── algorithms/          # 渲染算法引擎（U-Net + GS-PAT）
│   ├── core/                # 核心控制逻辑（TCP 通信、实时渲染循环）
│   ├── gui/                 # PySide6 可视化界面
│   ├── checkpoints/         # U-Net(32b) checkpoint
│   ├── main.py              # 上位机入口
│   └── requirements.txt     # Python 依赖
├── build.sbt
└── README.md
```

---

## FPGA 部分

### 技术栈

- Chisel 6.6.0 + Scala 2.13
- 主时钟 100 MHz
- 输出：8×SER、SRCLK、RCLK + BRAM 读接口

### 设计原则

**参数化一切**
时序相关的"魔法数字"全部提取为编译期参数（`DIV`、`SRCLK_LOW_CYCS`、`RCLK_HIGH_CYCS` 等），可在 `GenerateVerilog.scala` 中修改后重新生成 Verilog。

**乒乓双缓冲**
A/B 两帧缓冲交替工作：OutputDriver 输出当前周期数据的同时，后台从 BRAM 读取并准备下一周期数据，消除帧间毛刺。

**周期序号校验**
BRAM 每 64 行数据的头部携带 16bit 周期序号。FPGA 维护 `expectedCycle`，严格匹配才消费数据；失配则整周期输出全 0，换能器处于安全关闭状态。

**周期回绕对齐（关键）**
BRAM 深度约束为 `64 × 2^m`（默认 **262144** 字 = **4096** 帧槽，约 102 ms @40 kHz）。这使得 16bit 周期序号回绕（约 1.64s）后，同一周期必然落到同一物理地址。临时超速写入导致的覆盖将在 1.64s 后自动恢复同步（环缓覆盖为预期行为，不做写前检查）。

**RCLK 时序重叠**
RCLK 上升沿与最后一 SRCLK 下降沿同时产生，RCLK 高电平期间允许与下一帧的第一个 SRCLK 低电平完全重叠。这使 30 分频下 8×30×100ns = 24μs 刚好嵌入 25μs 周期，剩余 1μs 为周期间隔。

### 模块划分

| 模块 | 职责 |
|------|------|
| `Top` | 顶层，参数汇聚，子模块互联 |
| `BramReader` | 顺序读取 64 行 BRAM，周期序号校验，维护读指针 |
| `DataMapper` | 查 LUT 将占空比+相位映射为 DIV-bit 帧位图 |
| `FrameBuffer` | A/B 乒乓双缓冲，写入时完成转置（64×DIV → DIV×64） |
| `OutputDriver` | 生成精确的 SRCLK / RCLK / SER 时序 |
| `LutGenerator` | Scala 编译期生成占空比+相位 → PWM 位图常量表 |

### 构建

```bash
sbt test                    # 运行全部 16 个测试
sbt "runMain ultrasound.GenerateVerilog"   # 生成 Top.sv
```

生成的 Verilog 位于 `generated/Top.sv`，使用 `--preserve-aggregate=1d-vec` 保留数组结构、`--disable-all-randomization` 去除随机化宏。

### 顶层接口

```verilog
input         clock, reset;
input  [31:0] io_bramData;      // BRAM 读数据
output [17:0] io_bramAddr;      // BRAM 读地址（宽度随 bramDepth 变化，262144 时为 18 位）
output        io_bramRen;       // BRAM 读使能
output        io_ser_0..7;      // 8 路移位寄存器串行输入
output        io_srclk;         // 移位时钟
output        io_rclk;          // 锁存时钟
```

具体参数与内部时序逻辑详见各 `.scala` 源文件头部注释。

---

## 上位机（PC）部分

### 功能概述

上位机提供可视化 GUI 界面，用于：
- 配置焦点位置（至多 3 个）
- 选择渲染模式：**静态焦点**（同时渲染）或 **动态焦点**（循环切换，可配置停留时间）
- 选择算法：**U-Net(32b)**（质量优先）或 **GS-PAT**（速度优先）
- 配置 **Lateral Modulation** 参数（调制频率、幅度、采样点数、方向）
- 实时显示声场强度热力图
- 通过 TCP 以太网向下位机（Zynq ARM + FPGA）发送控制数据

### 技术栈

| 组件 | 选择 |
|------|------|
| 语言 | Python 3.11+ |
| GUI | PySide6 |
| 深度学习 | PyTorch 2.9+ |
| 网络 | TCP (socket) |
| 可视化 | matplotlib |

### 安装依赖

```bash
cd host/
pip install -r requirements.txt
```

依赖列表：
- `torch>=2.0.0`（需匹配 CUDA 版本，或 CPU 版本）
- `numpy>=1.24.0`
- `PySide6>=6.5.0`
- `matplotlib>=3.7.0`

### 运行

```bash
cd host/
python main.py
```

### 使用流程

1. **配置网络连接**
   - 输入 Zynq 设备的 IP 地址（默认 `192.168.1.10`）和端口（默认 `5000`）
   - 点击 **Connect** 建立 TCP 连接
   - 可选勾选 **Mock Mode** 进行本地调试（不连接真实设备）

2. **配置焦点**
   - 在左侧 **Focus Positions** 面板中输入焦点坐标（X, Y），单位 mm
   - 点击 **Add Focus** / **Remove Focus** 调整焦点数量（1–3 个）
   - 2D 可视化区域实时显示焦点位置和换能器阵列布局

3. **选择模式和算法**
   - **Rendering Mode**: Static（同时渲染所有焦点）或 Dynamic（循环切换焦点）
   - **Algorithm**: U-Net(32b)（质量优先，需 GPU）或 GS-PAT（速度优先）

4. **配置 Lateral Modulation**
   - **Frequency**: 调制频率，默认 25 Hz
   - **Amplitude**: 调制幅度，默认 4 mm
   - **Samples/Period**: 每周期采样点数，默认 12（实际不同点约 7 个）
   - **Direction**: X 或 Y 轴调制方向

5. **开始渲染**
   - 点击 **Start Rendering**
   - 实时声场热力图将显示在下方
   - 状态栏显示当前 FPS、连接状态和渲染模式
   - 点击 **Stop Rendering** 停止

### 数据格式与发送策略

上位机以 **~40 kHz 平均速率**（约 **10.2 MB/s**）向设备送帧；`configure()` 时预计算 BRAM 模板，运行时向量化组批，避免 Python 逐帧循环成为瓶颈：约每 **12.8 ms**（512/40000 s）唤醒一次，根据距上次发送的时间差计算本批帧数 `n = round(Δt × 40000)`（至少 1 帧），通过 **一次 TCP 写入** 发送 `n × 256` 字节。**每次 Start 后首包** 至少发送 `BURST_NOMINAL_FRAMES × BURST_PRIME_MULTIPLIER` 帧（默认 512×2=1024），用于垫高 BRAM 环，便于 PS 侧简化实现（见 `docs/ps_minimal_tcp_bram.md`）。LM 图案在 `configure()` 时预推理（去重后约数十个），运行时按全局帧序号查表；同一 LM 步在 40 kHz 流上持有约 `40000/(lm_freq×lm_samples)` 帧。

每帧 64 个 32-bit 整数（little-endian），对应 FPGA BRAM 的 64 行：

```
[31:16] cycle_index (16-bit unsigned, 严格递增)
[15:8]  duty (8-bit, 实际使用低 5-bit, 0–29)
[7:0]   phase (8-bit, 实际使用低 5-bit, 0–29)
```

转换公式：
- 相位映射：`phase_idx = round((phase + π) / (2π) × 30) % 30`
- 振幅→占空比：`duty_idx = round(arcsin(amplitude) / π × 30)`

### 算法说明

**U-Net(32b)**
- 输入：焦点位置生成的 positive mask `(128, 128)`
- 输出：phase `(8, 8)` 弧度 + amplitude `(8, 8)` `[0,1]`
- 自动使用 GPU（CUDA），无 CUDA 时回退到 CPU 多线程
- Checkpoint 位于 `host/checkpoints/checkpoint_best.pth`

**GS-PAT**
- Gerchberg-Saxton 迭代算法
- GPU 批处理版本优先，自动分桶处理不同焦点数量
- 无 CUDA 时回退到 NumPy CPU 版本

### 性能指标

| 算法 | 目标性能 | 备注 |
|------|----------|------|
| U-Net(32b) | ≥ 175 FPS (batch=7) | GPU + torch.compile |
| GS-PAT | ≥ 10,000 samples/s | GPU 批处理 |

### 文件说明

| 文件 | 说明 |
|------|------|
| `host/main.py` | 入口点 |
| `host/config.py` | 全局常量（物理参数、FPGA 参数、默认值） |
| `host/algorithms/engine.py` | U-Net / GS-PAT 推理引擎封装 |
| `host/core/renderer.py` | 实时渲染循环（LM 预推理 + 40 kHz 块发送） |
| `host/core/converter.py` | phase/amplitude → BRAM 格式转换 |
| `host/core/device_client.py` | TCP 客户端（`send_burst` 变长块） |
| `host/gui/main_window.py` | 主窗口组装 |
| `host/gui/focus_panel.py` | 焦点配置 + 2D 可视化 |
| `host/gui/param_panel.py` | 算法/LM/网络参数 |
| `host/gui/visualize_widget.py` | 声场热力图 (matplotlib) |
