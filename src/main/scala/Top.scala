package ultrasound

import chisel3._
import chisel3.util._

/**
  * Top-level module for the Zynq ultrasound transducer array driver.
  *
  * Parameters (all in 100MHz clock cycles unless noted):
  *   div           - number of frames per 40kHz cycle
  *   srclkLowCycs  - SRCLK low duration
  *   srclkHighCycs - SRCLK high duration
  *   rclkHighCycs  - RCLK high pulse duration
  *   cycleCycs     - total 40kHz cycle duration (2500 for 25us)
  */
class Top(
    div: Int = 30,
    srclkLowCycs: Int = 5,
    srclkHighCycs: Int = 5,
    rclkHighCycs: Int = 5,
    cycleCycs: Int = 2500,
    bramDepth: Int = 262144
) extends Module {

    val phaseDutyBits = log2Ceil(div)
    val bramAddrWidth = log2Ceil(bramDepth)

    val io = IO(new Bundle {
        // BRAM read interface
        val bramData = Input(UInt(32.W))
        val bramAddr = Output(UInt(bramAddrWidth.W))
        val bramRen  = Output(Bool())

        // 595 outputs
        val ser   = Output(Vec(8, Bool()))
        val srclk = Output(Bool())
        val rclk  = Output(Bool())
    })

    // ------------------------------------------------------------------
    // Instantiate submodules
    // ------------------------------------------------------------------
    val bramReader = Module(new BramReader(bramDepth))
    val dataMapper = Module(new DataMapper(div, phaseDutyBits))
    val frameBuffer = Module(new FrameBuffer(div))
    val outputDriver = Module(new OutputDriver(div, srclkLowCycs, srclkHighCycs, rclkHighCycs, cycleCycs))

    // ------------------------------------------------------------------
    // Connect BRAM interface
    // ------------------------------------------------------------------
    io.bramAddr := bramReader.io.bramAddr
    io.bramRen  := bramReader.io.bramRen
    bramReader.io.bramData := io.bramData

    // ------------------------------------------------------------------
    // Connect cycle start from OutputDriver to BramReader
    // ------------------------------------------------------------------
    bramReader.io.cycleStart := outputDriver.io.cycleStart

    // ------------------------------------------------------------------
    // Connect BramReader -> DataMapper
    // ------------------------------------------------------------------
    dataMapper.io.rawData := bramReader.io.rawData
    dataMapper.io.valid   := bramReader.io.valid

    // ------------------------------------------------------------------
    // Connect DataMapper -> FrameBuffer
    // ------------------------------------------------------------------
    frameBuffer.io.transducerData := dataMapper.io.frameData
    frameBuffer.io.writeValid     := dataMapper.io.ready
    frameBuffer.io.cycleStart     := outputDriver.io.cycleStart

    // ------------------------------------------------------------------
    // Connect FrameBuffer <-> OutputDriver
    // ------------------------------------------------------------------
    frameBuffer.io.readFrameIdx := outputDriver.io.readFrameIdx
    outputDriver.io.readFrameData := frameBuffer.io.readFrameData

    // ------------------------------------------------------------------
    // Connect 595 outputs
    // ------------------------------------------------------------------
    io.ser   := outputDriver.io.ser
    io.srclk := outputDriver.io.srclk
    io.rclk  := outputDriver.io.rclk
}
