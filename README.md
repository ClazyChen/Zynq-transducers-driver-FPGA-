# Zynq FPGA 超声换能器阵列驱动

8×8 超声换能器（40 kHz）FPGA 驱动，通过 8 路独立 SER 控制 8 片 74HC595 移位寄存器。

## 技术栈

- Chisel 6.6.0 + Scala 2.13
- 主时钟 100 MHz
- 输出：8×SER、SRCLK、RCLK + BRAM 读接口

## 设计原则

**参数化一切**
时序相关的"魔法数字"全部提取为编译期参数（`DIV`、`SRCLK_LOW_CYCS`、`RCLK_HIGH_CYCS` 等），可在 `GenerateVerilog.scala` 中修改后重新生成 Verilog。

**乒乓双缓冲**
A/B 两帧缓冲交替工作：OutputDriver 输出当前周期数据的同时，后台从 BRAM 读取并准备下一周期数据，消除帧间毛刺。

**周期序号校验**
BRAM 每 64 行数据的头部携带 16bit 周期序号。FPGA 维护 `expectedCycle`，严格匹配才消费数据；失配则整周期输出全 0，换能器处于安全关闭状态。

**周期回绕对齐（关键）**
BRAM 深度约束为 `64 × 2^m`（如 1024、4096、65536）。这使得 16bit 周期序号回绕（约 1.64s）后，同一周期必然落到同一物理地址。临时超速写入导致的覆盖将在 1.64s 后自动恢复同步。

**RCLK 时序重叠**
RCLK 上升沿与最后一 SRCLK 下降沿同时产生，RCLK 高电平期间允许与下一帧的第一个 SRCLK 低电平完全重叠。这使 30 分频下 8×30×100ns = 24μs 刚好嵌入 25μs 周期，剩余 1μs 为周期间隔。

## 模块划分

| 模块 | 职责 |
|------|------|
| `Top` | 顶层，参数汇聚，子模块互联 |
| `BramReader` | 顺序读取 64 行 BRAM，周期序号校验，维护读指针 |
| `DataMapper` | 查 LUT 将占空比+相位映射为 DIV-bit 帧位图 |
| `FrameBuffer` | A/B 乒乓双缓冲，写入时完成转置（64×DIV → DIV×64） |
| `OutputDriver` | 生成精确的 SRCLK / RCLK / SER 时序 |
| `LutGenerator` | Scala 编译期生成占空比+相位 → PWM 位图常量表 |

## 构建

```bash
sbt test                    # 运行全部 16 个测试
sbt "runMain ultrasound.GenerateVerilog"   # 生成 Top.sv
```

生成的 Verilog 位于 `generated/Top.sv`，使用 `--preserve-aggregate=1d-vec` 保留数组结构、`--disable-all-randomization` 去除随机化宏。

## 顶层接口

```verilog
input         clock, reset;
input  [31:0] io_bramData;      // BRAM 读数据
output [9:0]  io_bramAddr;      // BRAM 读地址（宽度随 bramDepth 变化）
output        io_bramRen;       // BRAM 读使能
output        io_ser_0..7;      // 8 路移位寄存器串行输入
output        io_srclk;         // 移位时钟
output        io_rclk;          // 锁存时钟
```

具体参数与内部时序逻辑详见各 `.scala` 源文件头部注释。
