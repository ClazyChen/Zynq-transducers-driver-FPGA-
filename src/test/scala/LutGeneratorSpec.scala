package ultrasound

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
  * Tests for the compile-time LUT generator.
  * Since LutGenerator is pure Scala logic, we test it directly without Chisel simulation.
  */
class LutGeneratorSpec extends AnyFlatSpec with Matchers {

    behavior of "LutGenerator"

    it should "generate correct PWM bitmap for DIV=30, duty=5, phase=3" in {
        val div = 30
        val bits = 5
        val lut = LutGenerator.generateLut(div, bits)

        val duty = 5
        val phase = 3
        val bitmap = lut(duty)(phase)

        // For each frame f, output should be HIGH if ((f - phase + div) % div) < duty
        for (f <- 0 until div) {
            val pos = (f - phase + div) % div
            val expected = pos < duty
            val actual = ((bitmap >> f) & 1) == 1
            actual shouldBe expected
        }
    }

    it should "generate all-zeros when duty=0" in {
        val div = 30
        val bits = 5
        val lut = LutGenerator.generateLut(div, bits)

        for (phase <- 0 until (1 << bits)) {
            lut(0)(phase) shouldBe BigInt(0)
        }
    }

    it should "generate all-ones when duty=div" in {
        val div = 30
        val bits = 5
        val lut = LutGenerator.generateLut(div, bits)

        for (phase <- 0 until (1 << bits)) {
            val bitmap = lut(div)(phase)
            val mask = (BigInt(1) << div) - 1
            (bitmap & mask) shouldBe mask
        }
    }

    it should "correctly wrap around phase" in {
        val div = 30
        val bits = 5
        val lut = LutGenerator.generateLut(div, bits)

        val duty = 10
        val phase = 25
        val bitmap = lut(duty)(phase)

        for (f <- 0 until div) {
            val pos = (f - phase + div) % div
            val expected = pos < duty
            val actual = ((bitmap >> f) & 1) == 1
            actual shouldBe expected
        }
    }
}
