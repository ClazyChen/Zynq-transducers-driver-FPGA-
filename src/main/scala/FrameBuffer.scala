package ultrasound

import chisel3._
import chisel3.util._

/**
  * FrameBuffer: A/B ping-pong double buffer for 40kHz cycle frame data.
  *
  * Storage format: Vec(DIV, UInt(64.W))
  *   - Each entry corresponds to one frame (sub-period) within the 40kHz cycle.
  *   - Bit i (0 <= i < 64) is the ON/OFF state of transducer i during that frame.
  *
  * Write port accepts Vec(64, UInt(div.W)) from DataMapper (transducer-major)
  * and transposes it into frame-major format on write.
  *
  * Read port provides random access by frame index for OutputDriver.
  *
  * Ping-pong: on cycleStart, the active buffer flips to the one that was
  * prepared in the background during the previous 40kHz cycle.
  */
class FrameBuffer(div: Int) extends Module {
    val io = IO(new Bundle {
        // ---- Write port (from DataMapper) ----
        // transducerData(i)(f) = transducer i's state at frame f
        val transducerData = Input(Vec(64, UInt(div.W)))
        val writeValid     = Input(Bool())

        // ---- Read port (to OutputDriver) ----
        val readFrameIdx  = Input(UInt(log2Ceil(div).W))
        val readFrameData = Output(UInt(64.W))

        // ---- Control ----
        val cycleStart = Input(Bool())
    })

    // ------------------------------------------------------------------
    // Storage: two buffers, each holds DIV frames x 64 transducers
    // ------------------------------------------------------------------
    val bufA = RegInit(VecInit(Seq.fill(div)(0.U(64.W))))
    val bufB = RegInit(VecInit(Seq.fill(div)(0.U(64.W))))

    // Active buffer selector: 0 = A, 1 = B
    // OutputDriver reads from active; background writes to inactive.
    val activeBuf = RegInit(0.U(1.W))

    // ------------------------------------------------------------------
    // Ping-pong switch on cycle start
    // ------------------------------------------------------------------
    when(io.cycleStart) {
        activeBuf := ~activeBuf
    }

    // ------------------------------------------------------------------
    // Write path: transpose [transducer][frame] -> [frame][transducer]
    // ------------------------------------------------------------------
    val writeTarget = Mux(activeBuf === 0.U, 1.U, 0.U)

    // Pre-compute the transposed frame bits combinationally.
    // frameBits(f) is a 64-bit vector where bit i = transducerData(i)(f)
    val frameBits = Wire(Vec(div, UInt(64.W)))
    for (f <- 0 until div) {
        val bitsVec = VecInit((0 until 64).map(i => io.transducerData(i)(f)))
        frameBits(f) := bitsVec.asUInt
    }

    when(io.writeValid) {
        for (f <- 0 until div) {
            when(writeTarget === 0.U) {
                bufA(f) := frameBits(f)
            } .otherwise {
                bufB(f) := frameBits(f)
            }
        }
    }

    // ------------------------------------------------------------------
    // Read path
    // ------------------------------------------------------------------
    io.readFrameData := Mux(activeBuf === 0.U, bufA(io.readFrameIdx), bufB(io.readFrameIdx))
}
