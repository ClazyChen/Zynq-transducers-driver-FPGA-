package ultrasound

import chisel3._
import chisel3.util._

/**
  * DataMapper: converts 64 rows of BRAM raw data into frame buffer data.
  *
  * Input:  64 rows of 32-bit raw data (16-bit cycle index + 8-bit duty + 8-bit phase)
  * Output: 64 entries of DIV-bit frame bitmap per transducer.
  *
  * The lower PHASE_DUTY_BITS of duty and phase are extracted and used to
  * index into the compile-time generated LUT.
  */
class DataMapper(div: Int, phaseDutyBits: Int) extends Module {
    val io = IO(new Bundle {
        // Input: 64 rows of raw BRAM data, valid when ren is true
        val rawData = Input(Vec(64, UInt(32.W)))
        val valid   = Input(Bool())

        // Output: 64 transducers' frame bitmaps
        val frameData = Output(Vec(64, UInt(div.W)))
        val ready     = Output(Bool())
    })

    // Compile-time generated LUT: Vec(duty)(phase) -> UInt(div.W)
    val lut = LutGenerator.chiselLut(div, phaseDutyBits)

    // Extract duty and phase from each row.
    // BRAM data layout: [31:16] cycle index, [15:8] duty, [7:0] phase
    val duties  = Wire(Vec(64, UInt(phaseDutyBits.W)))
    val phases  = Wire(Vec(64, UInt(phaseDutyBits.W)))

    for (i <- 0 until 64) {
        // Extract lower PHASE_DUTY_BITS from the duty and phase fields
        duties(i) := io.rawData(i)(15, 8)(phaseDutyBits - 1, 0)
        phases(i) := io.rawData(i)(7, 0)(phaseDutyBits - 1, 0)
    }

    // Lookup frame bitmap for each transducer using the LUT.
    // Since the LUT is a compile-time constant Vec, this is combinational logic.
    for (i <- 0 until 64) {
        io.frameData(i) := lut(duties(i))(phases(i))
    }

    // DataMapper is purely combinational; ready follows valid.
    io.ready := io.valid
}
