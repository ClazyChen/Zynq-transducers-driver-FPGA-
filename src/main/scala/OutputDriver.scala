package ultrasound

import chisel3._
import chisel3.util._

/**
  * OutputDriver: generates precise 74HC595 timing for the 8x8 transducer array.
  *
  * Timing parameters (all in 100MHz clock cycles):
  *   srclkLowCycs  - SRCLK low duration
  *   srclkHighCycs - SRCLK high duration
  *   rclkHighCycs  - RCLK high pulse duration
  *   cycleCycs     - total 40kHz cycle duration
  *
  * Frame timing (example with default 50ns values, DIV=30):
  *   - Each frame = 8 SRCLK periods = 8 * 10 = 80 cycles (800ns)
  *   - RCLK rises on the falling edge of the 8th SRCLK (i.e. at the frame boundary)
  *   - RCLK high overlaps the next frame's first SRCLK low period
  *   - 30 frames = 2400 cycles, remaining 100 cycles = 1us cycle tail gap
  */
class OutputDriver(
    div: Int,
    srclkLowCycs: Int,
    srclkHighCycs: Int,
    rclkHighCycs: Int,
    cycleCycs: Int
) extends Module {

    // Compile-time sanity check
    val srclkPeriod = srclkLowCycs + srclkHighCycs
    val frameCycs = 8 * srclkPeriod
    require(div * frameCycs + rclkHighCycs <= cycleCycs,
        s"Timing violation: $div frames ($div * $frameCycs) + RCLK tail ($rclkHighCycs) = ${div * frameCycs + rclkHighCycs} > $cycleCycs")

    val io = IO(new Bundle {
        // Frame buffer interface
        val readFrameIdx  = Output(UInt(log2Ceil(div).W))
        val readFrameData = Input(UInt(64.W))

        // 595 outputs
        val ser   = Output(Vec(8, Bool()))
        val srclk = Output(Bool())
        val rclk  = Output(Bool())

        // Pulse to BramReader at start of each 40kHz cycle
        val cycleStart = Output(Bool())
    })

    // ------------------------------------------------------------------
    // Cycle / frame / position counters
    // ------------------------------------------------------------------
    val cycleCnt = RegInit(0.U(log2Ceil(cycleCycs).W))
    val frameCnt = RegInit(0.U(log2Ceil(div).W))
    val framePos = RegInit(0.U(log2Ceil(frameCycs).W))

    // Advance global cycle counter every clock
    cycleCnt := Mux(cycleCnt === (cycleCycs - 1).U, 0.U, cycleCnt + 1.U)

    // cycleStart pulse on rollover
    io.cycleStart := cycleCnt === (cycleCycs - 1).U

    // Are we inside the active frame-output window?
    val inFrameWindow = cycleCnt < (div * frameCycs).U

    // Frame counter & intra-frame position counter
    when(cycleCnt === (cycleCycs - 1).U) {
        // Reset at end of 40kHz cycle
        frameCnt := 0.U
        framePos := 0.U
    } .elsewhen(inFrameWindow) {
        when(framePos === (frameCycs - 1).U) {
            framePos := 0.U
            frameCnt := Mux(frameCnt === (div - 1).U, frameCnt, frameCnt + 1.U)
        } .otherwise {
            framePos := framePos + 1.U
        }
    }

    // ------------------------------------------------------------------
    // Decode column and half-period within the current SRCLK cycle
    // ------------------------------------------------------------------
    val colIdx  = (framePos / srclkPeriod.U)(2, 0)   // which of 8 columns
    val halfIdx = framePos % srclkPeriod.U            // position within SRCLK period

    // ------------------------------------------------------------------
    // SRCLK generation: high during second half of each SRCLK period
    // ------------------------------------------------------------------
    io.srclk := inFrameWindow && (halfIdx >= srclkLowCycs.U)

    // ------------------------------------------------------------------
    // RCLK generation: high for rclkHighCycs cycles at each frame boundary.
    // Frame boundary = cycleCnt where framePos == 0 AND not the very first cycle.
    // This naturally overlaps the next frame's first SRCLK low period.
    // ------------------------------------------------------------------
    val inRclkWindow = (cycleCnt >= frameCycs.U) &&
                       (cycleCnt < (div * frameCycs + rclkHighCycs).U) &&
                       ((cycleCnt % frameCycs.U) < rclkHighCycs.U)
    io.rclk := inRclkWindow

    // ------------------------------------------------------------------
    // Frame buffer read address
    // ------------------------------------------------------------------
    io.readFrameIdx := Mux(inFrameWindow, frameCnt, 0.U)

    // ------------------------------------------------------------------
    // SER generation: mux the correct column for each row
    // frameData bit layout: bit (r*8 + c) = transducer (row=r, col=c)
    // During inactive periods, drive all SER low.
    // ------------------------------------------------------------------
    // Extract each bit of the 64-bit frame data into a Vec for clean indexing
    val frameBits = VecInit((0 until 64).map(i => io.readFrameData(i)))

    for (r <- 0 until 8) {
        val baseIdx = (r * 8).U(6.W)
        val bitIdx  = baseIdx + colIdx
        io.ser(r) := inFrameWindow && frameBits(bitIdx)
    }
}
