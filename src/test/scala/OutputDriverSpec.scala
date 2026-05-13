package ultrasound

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
  * Tests for OutputDriver 595 timing generator.
  */
class OutputDriverSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {

    behavior of "OutputDriver"

    it should "produce correct SRCLK low/high durations" in {
        val div = 2
        val srclkLow = 5
        val srclkHigh = 5
        val rclkHigh = 5
        val cycleCycs = 250

        test(new OutputDriver(div, srclkLow, srclkHigh, rclkHigh, cycleCycs)) { dut =>
            // Frame 0, column 0: SRCLK low for 5 cycles, then high for 5 cycles
            for (c <- 0 until 8) {
                for (h <- 0 until srclkLow) {
                    dut.io.srclk.expect(false.B)
                    dut.clock.step()
                }
                for (h <- 0 until srclkHigh) {
                    dut.io.srclk.expect(true.B)
                    dut.clock.step()
                }
            }
        }
    }

    it should "raise RCLK right after the last SRCLK of each frame" in {
        val div = 2
        val srclkLow = 5
        val srclkHigh = 5
        val rclkHigh = 5
        val cycleCycs = 250
        val frameCycs = 8 * (srclkLow + srclkHigh)

        test(new OutputDriver(div, srclkLow, srclkHigh, rclkHigh, cycleCycs)) { dut =>
            // Skip to end of frame 0 (8 SRCLK periods = 80 cycles)
            // At cycle 80, RCLK should rise (overlapping frame 1 start)
            for (_ <- 0 until frameCycs) { dut.clock.step() }

            // RCLK should be high for rclkHigh cycles
            for (_ <- 0 until rclkHigh) {
                dut.io.rclk.expect(true.B)
                dut.clock.step()
            }

            // RCLK should drop
            dut.io.rclk.expect(false.B)
        }
    }

    it should "output correct SER data per frame and column" in {
        val div = 2
        val srclkLow = 5
        val srclkHigh = 5
        val rclkHigh = 5
        val cycleCycs = 250

        test(new OutputDriver(div, srclkLow, srclkHigh, rclkHigh, cycleCycs)) { dut =>
            // Prepare frame data: frame 0 = 0xAAAAAAAAAAAAAAAA, frame 1 = 0x5555555555555555
            // Bit pattern: even bits = 1 in frame 0, odd bits = 1 in frame 1
            dut.io.readFrameData.poke(BigInt("AAAAAAAAAAAAAAAA", 16).U(64.W))

            // Wait for frame 0, column 0 start
            dut.clock.step()

            // Frame 0, col 0: ser(r) = bit (r*8 + 0)
            // In 0xAA..., bit 0=0, bit 8=0, bit 16=0...
            for (r <- 0 until 8) {
                dut.io.ser(r).expect(false.B)
            }

            // Skip ahead to frame 1, col 0 (after frame 0 completes)
            val frameCycs = 8 * (srclkLow + srclkHigh)
            // Step until we reach frame 1
            for (_ <- 0 until frameCycs - 1) { dut.clock.step() }

            // Provide frame 1 data when driver asks for it
            dut.io.readFrameIdx.expect(1.U)
            dut.io.readFrameData.poke(BigInt("5555555555555555", 16).U(64.W))
            dut.clock.step()

            // Now in frame 1, col 0
            // In 0x55..., bit 0=1, bit 8=1, bit 16=1...
            for (r <- 0 until 8) {
                dut.io.ser(r).expect(true.B)
            }
        }
    }

    it should "assert cycleStart at the end of each 40kHz cycle" in {
        val div = 2
        val srclkLow = 5
        val srclkHigh = 5
        val rclkHigh = 5
        val cycleCycs = 200  // must be >= div*8*(srclkLow+srclkHigh) + rclkHigh = 165

        test(new OutputDriver(div, srclkLow, srclkHigh, rclkHigh, cycleCycs)) { dut =>
            for (_ <- 0 until cycleCycs - 2) {
                dut.io.cycleStart.expect(false.B)
                dut.clock.step()
            }
            // One cycle before rollover
            dut.io.cycleStart.expect(false.B)
            dut.clock.step()
            // At rollover
            dut.io.cycleStart.expect(true.B)
        }
    }
}
