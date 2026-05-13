package ultrasound

import chisel3._
import chisel3.util._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
  * End-to-end integration tests for the Top module.
  */
class TopSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {

    behavior of "Top"

    it should "output correct 595 timing for two consecutive cycles with matching data" in {
        val div = 4
        val srclkLow = 2
        val srclkHigh = 2
        val rclkHigh = 2
        val cycleCycs = 140  // must be >= div*8*(srclkLow+srclkHigh) + rclkHigh = 4*32+2=130

        test(new Top(div, srclkLow, srclkHigh, rclkHigh, cycleCycs)) { dut =>
            val frameCycs = 8 * (srclkLow + srclkHigh)

            // Align to a cycleStart: wait for bramRen to go high
            while (!dut.io.bramRen.peekBoolean()) { dut.clock.step() }

            // ----- Feed Cycle 1 BRAM data (cycle index = 1) -----
            val rowDataCycle1 = (1 << 16) | (2 << 8) | 1  // duty=2, phase=1
            val lut = LutGenerator.generateLut(div, log2Ceil(div))
            val expectedBitmap1 = lut(2)(1)

            for (addr <- 0 until 64) {
                dut.io.bramAddr.expect(addr.U)
                dut.io.bramRen.expect(true.B)
                dut.io.bramData.poke(rowDataCycle1.U)
                dut.clock.step()
            }
            dut.io.bramRen.expect(false.B)

            // Wait until the next cycleStart (end of current 40kHz cycle)
            while (!dut.io.bramRen.peekBoolean()) { dut.clock.step() }

            // Now we are at the start of Cycle 2 output.
            // Verify first few SRCLK cycles of the new cycle.
            for (c <- 0 until 8) {
                for (_ <- 0 until srclkLow) {
                    dut.io.srclk.expect(false.B)
                    dut.clock.step()
                }
                for (_ <- 0 until srclkHigh) {
                    dut.io.srclk.expect(true.B)
                    dut.clock.step()
                }
            }

            // Verify RCLK rises right after frame 0's last SRCLK
            dut.io.rclk.expect(true.B)
            dut.clock.step()
            dut.io.rclk.expect(true.B)
            dut.clock.step()
            dut.io.rclk.expect(false.B)
        }
    }

    it should "output all zeros when BRAM cycle index does not match" in {
        val div = 4
        val srclkLow = 2
        val srclkHigh = 2
        val rclkHigh = 2
        val cycleCycs = 140

        test(new Top(div, srclkLow, srclkHigh, rclkHigh, cycleCycs)) { dut =>
            // Align to cycleStart
            while (!dut.io.bramRen.peekBoolean()) { dut.clock.step() }

            // First cycle with correct index 1 to set expectedCycle = 2
            val rowDataOk = (1 << 16) | (2 << 8) | 1
            for (_ <- 0 until 64) {
                dut.io.bramData.poke(rowDataOk.U)
                dut.clock.step()
            }

            // Wait for next cycleStart
            while (!dut.io.bramRen.peekBoolean()) { dut.clock.step() }

            // Second cycle with wrong index 99 (expected 2)
            val rowDataBad = (99 << 16) | (2 << 8) | 1
            for (_ <- 0 until 64) {
                dut.io.bramData.poke(rowDataBad.U)
                dut.clock.step()
            }

            // Wait for next cycleStart when the bad data will be output
            while (!dut.io.bramRen.peekBoolean()) { dut.clock.step() }

            // After cycleStart, the output should be all zeros for this cycle
            for (_ <- 0 until 20) {
                for (r <- 0 until 8) {
                    dut.io.ser(r).expect(false.B)
                }
                dut.clock.step()
            }
        }
    }
}
