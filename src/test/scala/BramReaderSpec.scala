package ultrasound

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
  * Tests for BramReader module.
  */
class BramReaderSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {

    behavior of "BramReader"

    val bramDepth = 256  // small depth for fast test (must be multiple of 64)

    it should "read 64 rows and match expected cycle" in {
        test(new BramReader(bramDepth)) { dut =>
            dut.io.cycleStart.poke(true.B)
            dut.io.bramData.poke(0.U)
            dut.clock.step()
            dut.io.cycleStart.poke(false.B)

            // Feed BRAM data for 64 cycles
            // Each row: cycle index = 1 (expected), duty = 2, phase = 3
            val cycleIdx = 1
            val rowData = (cycleIdx << 16) | (2 << 8) | 3

            for (addr <- 0 until 64) {
                dut.io.bramAddr.expect(addr.U)
                dut.io.bramRen.expect(true.B)
                dut.io.bramData.poke(rowData.U)
                dut.clock.step()
            }

            // Wait for CHECK -> DONE (sCheck takes 1 cycle, sDone takes 1 cycle)
            while (!dut.io.valid.peekBoolean()) { dut.clock.step() }

            dut.io.valid.expect(true.B)
            dut.io.cycleMatch.expect(true.B)
            // expectedCycle increments during sCheck, so by sDone it is already 2
            dut.io.expectedCycle.expect(2.U)

            // Verify all 64 rows
            for (i <- 0 until 64) {
                dut.io.rawData(i).expect(rowData.U)
            }
        }
    }

    it should "output zeros and not increment when cycle index mismatches" in {
        test(new BramReader(bramDepth)) { dut =>
            // First do a successful read to set expectedCycle = 2
            dut.io.cycleStart.poke(true.B)
            dut.io.bramData.poke(((1 << 16) | (2 << 8) | 3).U)
            dut.clock.step()
            dut.io.cycleStart.poke(false.B)
            for (_ <- 0 until 64) { dut.clock.step() }
            dut.clock.step(2)
            dut.io.cycleMatch.expect(true.B)
            dut.clock.step()

            // Now trigger with mismatch: BRAM has cycle 5, expected is 2
            dut.io.cycleStart.poke(true.B)
            dut.clock.step()
            dut.io.cycleStart.poke(false.B)

            val badRowData = (5 << 16) | (2 << 8) | 3
            for (_ <- 0 until 64) {
                dut.io.bramData.poke(badRowData.U)
                dut.clock.step()
            }
            while (!dut.io.valid.peekBoolean()) { dut.clock.step() }

            dut.io.valid.expect(true.B)
            dut.io.cycleMatch.expect(false.B)
            dut.io.expectedCycle.expect(2.U) // should NOT increment

            for (i <- 0 until 64) {
                dut.io.rawData(i).expect(0.U)
            }
        }
    }
}
