package ultrasound

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/**
  * Tests for FrameBuffer A/B ping-pong module.
  */
class FrameBufferSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {

    behavior of "FrameBuffer"

    it should "correctly transpose and store transducer-major data" in {
        val div = 4

        test(new FrameBuffer(div)) { dut =>
            // Write pattern: transducer 0 = 0b0001, transducer 1 = 0b0010, etc.
            // transducerData(i)(f) = (i >> f) & 1
            for (i <- 0 until 64) {
                var bitmap = 0
                for (f <- 0 until div) {
                    if (((i >> f) & 1) != 0) bitmap |= (1 << f)
                }
                dut.io.transducerData(i).poke(bitmap.U)
            }
            dut.io.writeValid.poke(true.B)
            dut.io.cycleStart.poke(false.B)
            dut.clock.step()
            dut.io.writeValid.poke(false.B)

            // Switch active buffer so we can read the written data
            dut.io.cycleStart.poke(true.B)
            dut.clock.step()
            dut.io.cycleStart.poke(false.B)

            // Verify each frame: frame f should have bit i = transducerData(i)(f)
            for (f <- 0 until div) {
                dut.io.readFrameIdx.poke(f.U)
                dut.clock.step()

                val frameData = dut.io.readFrameData.peekInt()
                for (i <- 0 until 64) {
                    val expected = ((i >> f) & 1) == 1
                    val actual = ((frameData >> i) & 1) == 1
                    actual shouldBe expected
                }
            }
        }
    }

    it should "ping-pong between two buffers on successive cycleStarts" in {
        val div = 4

        test(new FrameBuffer(div)) { dut =>
            // Write pattern A (all 0x0F)
            for (i <- 0 until 64) dut.io.transducerData(i).poke(0x0F.U)
            dut.io.writeValid.poke(true.B)
            dut.clock.step()
            dut.io.writeValid.poke(false.B)

            // cycleStart: switch to buffer containing pattern A
            dut.io.cycleStart.poke(true.B)
            dut.clock.step()
            dut.io.cycleStart.poke(false.B)

            dut.io.readFrameIdx.poke(0.U)
            dut.clock.step()
            dut.io.readFrameData.peekInt() shouldBe BigInt("F" * 16, 16) // 64 bits of 1

            // Write pattern B (all 0x00) to inactive buffer
            for (i <- 0 until 64) dut.io.transducerData(i).poke(0x00.U)
            dut.io.writeValid.poke(true.B)
            dut.clock.step()
            dut.io.writeValid.poke(false.B)

            // cycleStart: switch to buffer containing pattern B
            dut.io.cycleStart.poke(true.B)
            dut.clock.step()
            dut.io.cycleStart.poke(false.B)

            dut.io.readFrameIdx.poke(0.U)
            dut.clock.step()
            dut.io.readFrameData.peekInt() shouldBe BigInt(0)
        }
    }
}
