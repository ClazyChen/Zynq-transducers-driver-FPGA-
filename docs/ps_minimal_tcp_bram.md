# Zynq PS：最小 TCP → BRAM 实现说明

面向 **lwIP + AXI BRAM** 直连、现场 demo（千兆网线、短时使用）的极简约定。与 PC 端 [`host/core/device_client.py`](../host/core/device_client.py) 及 FPGA [`BramReader`](../src/main/scala/BramReader.scala) 对齐。

## PC 发送行为

| 项 | 值 |
|----|-----|
| 协议 | TCP，默认端口 **55555**（PS IP 示例 **192.168.1.20**） |
| 负载 | 无应用层包头，连续帧字节流 |
| 每帧 | **256 字节** = 64 × uint32，**小端** |
| 稳态 | 约每 12.8 ms 一批，`n ≈ round(Δt × 40000)` 帧 |
| **首包（每次 Start）** | `n ≥ BURST_NOMINAL_FRAMES × BURST_PRIME_MULTIPLIER`（默认 **1024** 帧） |

## PS 推荐实现（尽量简单）

1. **TCP 监听** 55555（或与 PC GUI 一致），accept 后 `write_byte_offset = 0`。
2. **每次 `recv`**：`memcpy` 到 `bram_base + write_byte_offset`，`write_byte_offset += len`，再对 `262144×4` 取模回绕。
3. **BRAM**：深度 **262144** 字（32 bit/字）；与 PL 共用同一物理 BRAM。
4. **地址属性**：PS 映射为 **non-cacheable**（或写后 cache flush），避免 PL 读到旧数据。
5. **重连**：新连接时 `write_byte_offset = 0`；建议复位 PL 或等 PC 重新 Start。

不必实现：读 PL `readPtr`、水位、cycle 校验、LM、40 kHz 定时。

## 帧格式（与 PC 一致）

每帧 64 字，字内布局：

```
[31:16] cycle_index（16 bit，严格递增，回绕 65536）
[15:8]  duty
[7:0]   phase
```

PC 在 64 个字上填入相同 `cycle_index`。PL 每 40 kHz 从 `readPtr` 读 64 字，校验 word0 的 `cycle_index`（及可选更严规则）。

## 设计假设（demo）

- 首包加倍垫高后，**写指针领先于 PL 读指针**，PL 一般不读到 TCP 尚未凑齐的尾部（PS 按字节流写入即可）。
- 写快于读时的 **环覆盖** 为预期行为，不做写前检查。
- 偶发 `cycle` 失配时 PL 输出全 0，下一帧可能恢复。

## 联调检查

- Wireshark / 日志：首包 TCP 长度应为 **1024×256 = 262144** 字节（默认配置）。
- ILA：`expectedCycle`、`cycleMatch`、`readPtr` 在 Start 后数十微秒内应开始连续匹配。
