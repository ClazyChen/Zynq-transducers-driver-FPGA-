package ultrasound

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
  * Tests for DataMapper module.
  */
class DataMapperSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {

    behavior of "DataMapper"

    it should "correctly map 64 rows of raw data using the LUT" in {
        // Use small div to keep LUT tiny and test fast
        val div = 4
        val phaseDutyBits = 2

        test(new DataMapper(div, phaseDutyBits)) { dut =>
            // Prepare test data: all transducers have duty=2, phase=1
            val testDuty = 2
            val testPhase = 1
            val rawValue = (testDuty << 8) | testPhase  // cycle index = 0

            for (i <- 0 until 64) {
                dut.io.rawData(i).poke(rawValue.U)
            }
            dut.io.valid.poke(true.B)

            dut.clock.step()

            // Read back mapped frame data
            val lut = LutGenerator.generateLut(div, phaseDutyBits)
            val expectedBitmap = lut(testDuty)(testPhase)

            for (i <- 0 until 64) {
                val actual = dut.io.frameData(i).peekInt()
                actual shouldBe expectedBitmap
            }

            dut.io.ready.peekBoolean() shouldBe true
        }
    }

    it should "output zero when valid is low" in {
        val div = 4
        val phaseDutyBits = 2

        test(new DataMapper(div, phaseDutyBits)) { dut =>
            dut.io.rawData(0).poke(0x1234.U)
            dut.io.valid.poke(false.B)

            dut.clock.step()

            dut.io.ready.peekBoolean() shouldBe false
        }
    }
}
