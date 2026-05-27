package ultrasound

import chisel3._
import chisel3.util._

/**
  * BramReader: BRAM read controller for 64-row ultrasound pattern data.
  *
  * Each 40kHz cycle, this module sequentially reads 64 rows from BRAM.
  * It checks whether the 16-bit cycle index matches the expected cycle number.
  * If matched, the 64 rows are output as valid raw data and expected_cycle increments.
  * If mismatched, all-zero data is output (transducers off for this cycle) and the
  * read pointer does NOT advance, so the same address will be retried next cycle.
  *
  * BRAM is treated as a circular buffer of depth `bramDepth` (must be a multiple of 64).
  * The read pointer advances by 64 on each successful consumption.
  *
  * BRAM interface: combinational/1-cycle read latency assumed.
  *   - bram_addr + bram_ren presented on cycle N
  *   - bram_data valid on cycle N+1
  */
class BramReader(bramDepth: Int = 262144) extends Module {

    require(bramDepth % 64 == 0, s"bramDepth ($bramDepth) must be a multiple of 64")
    val frameSlots = bramDepth / 64
    require((frameSlots & (frameSlots - 1)) == 0,
        s"frameSlots ($frameSlots = bramDepth/64) must be a power of 2 to ensure cycle-index wrap-around alignment")
    val addrWidth = log2Ceil(bramDepth)

    val io = IO(new Bundle {
        // Trigger from OutputDriver: start of a new 40kHz cycle
        val cycleStart = Input(Bool())

        // BRAM read interface
        val bramAddr = Output(UInt(addrWidth.W))
        val bramRen  = Output(Bool())
        val bramData = Input(UInt(32.W))

        // Output to DataMapper
        val rawData  = Output(Vec(64, UInt(32.W)))
        val valid    = Output(Bool())

        // Debug/monitoring
        val expectedCycle = Output(UInt(16.W))
        val cycleMatch    = Output(Bool())
    })

    // ------------------------------------------------------------------
    // State machine
    // ------------------------------------------------------------------
    val sIdle :: sReading :: sCheck :: sDone :: Nil = Enum(4)
    val state = RegInit(sIdle)

    // Expected cycle counter: starts at 1, increments on successful match
    val expectedCycle = RegInit(1.U(16.W))

    // Base read pointer: advances by 64 rows on each successful cycle match.
    // Wraps around to 0 after reaching bramDepth - 64.
    val readPtr = RegInit(0.U(addrWidth.W))

    // Offset counter within the current 64-row block
    val readCnt = RegInit(0.U(6.W))

    // Pipeline register for returned BRAM data.
    val dataRegs = Reg(Vec(64, UInt(32.W)))
    val cycleMatchReg = Reg(Bool())

    // Capture BRAM data into the appropriate slot (1-cycle delayed).
    val bramRenDly = RegNext(io.bramRen)
    val addrDly    = RegNext(readCnt)

    when(bramRenDly) {
        dataRegs(addrDly) := io.bramData
    }

    // ------------------------------------------------------------------
    // Default outputs
    // ------------------------------------------------------------------
    io.bramAddr := readPtr + readCnt
    io.bramRen  := false.B
    io.rawData  := dataRegs
    io.valid    := false.B
    io.expectedCycle := expectedCycle
    io.cycleMatch    := cycleMatchReg

    // ------------------------------------------------------------------
    // State transitions
    // ------------------------------------------------------------------
    switch(state) {
        is(sIdle) {
            when(io.cycleStart) {
                state   := sReading
                readCnt := 0.U
            }
        }

        is(sReading) {
            io.bramRen := true.B
            io.bramAddr := readPtr + readCnt

            when(readCnt === 63.U) {
                state := sCheck
            } .otherwise {
                readCnt := readCnt + 1.U
            }
        }

        is(sCheck) {
            val cycleIdx = dataRegs(0)(31, 16)
            val matched  = cycleIdx === expectedCycle

            cycleMatchReg := matched

            when(matched) {
                // Valid data: advance read pointer and expected cycle
                expectedCycle := expectedCycle + 1.U
                val nextPtr = readPtr + 64.U
                readPtr := Mux(nextPtr === bramDepth.U, 0.U, nextPtr)
            } .otherwise {
                // Mismatch: zero out all data, read pointer stays for retry
                for (i <- 0 until 64) {
                    dataRegs(i) := 0.U
                }
            }

            state := sDone
        }

        is(sDone) {
            io.valid := true.B
            io.rawData := dataRegs
            state := sIdle
        }
    }
}
