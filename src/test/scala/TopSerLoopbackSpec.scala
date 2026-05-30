package ultrasound

import chisel3._
import chisel3.util.log2Ceil
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import java.nio.file.{Files, Paths}
import java.nio.ByteOrder

/**
  * End-to-end Top test: BRAM -> BramReader -> DataMapper -> FrameBuffer -> OutputDriver -> SER.
  *
  * Reconstructs per-transducer PWM bitmaps from io.ser (column-by-column) and compares to LUT
  * expectations derived from BRAM duty/phase fields — software twin for ILA on ser/srclk/rclk.
  *
  * Optional VCD (slow, production timing):
  *   set TOP_SER_DUMP_VCD=1
  *   sbt "testOnly ultrasound.TopSerLoopbackSpec -- -z VCD"
  */
class TopSerLoopbackSpec extends AnyFlatSpec with ChiselScalatestTester with Matchers {

    val bramDepth = 64
    val phaseDutyMask = 0x1F

    /** Production 40 kHz / DIV=30 timing (matches Top defaults). */
    val production = TimingParams(30, 5, 5, 5, 2500)

    /** Shorter cycle for fast regression (same structure as TopSpec). */
    val fast = TimingParams(4, 2, 2, 2, 140)

    case class TimingParams(
        div: Int,
        srclkLow: Int,
        srclkHigh: Int,
        rclkHigh: Int,
        cycleCycs: Int
    ) {
        val phaseDutyBits: Int = log2Ceil(div)
        val srclkPeriod: Int = srclkLow + srclkHigh
        val frameCycs: Int = 8 * srclkPeriod
    }

    def readBramFile(path: String): Array[BigInt] = {
        val bytes = Files.readAllBytes(Paths.get(path))
        require(bytes.length == 64 * 4, s"Expected 256 bytes (64 uint32), got ${bytes.length}")
        val bb = java.nio.ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        Array.fill(64)(BigInt(bb.getInt() & 0xFFFFFFFFL))
    }

    def dutyPhaseFromRow(row: BigInt): (Int, Int) = {
        val duty = ((row >> 8) & phaseDutyMask).toInt
        val phase = (row & phaseDutyMask).toInt
        (duty, phase)
    }

    def stepUntilCycleStart(dut: Top, timing: TimingParams): Unit = {
        var steps = 0
        var done = false
        val limit = timing.cycleCycs * 2
        while (!done && steps < limit) {
            if (dut.io.cycleStart.peekBoolean()) done = true
            dut.clock.step()
            steps += 1
        }
        require(done, "timeout waiting for cycleStart")
    }

    def sampleSerBitmaps(dut: Top, timing: TimingParams): Array[Int] = {
        val bitmaps = Array.fill(64)(0)
        stepUntilCycleStart(dut, timing)

        for (f <- 0 until timing.div) {
            for (col <- 0 until 8) {
                for (r <- 0 until 8) {
                    val i = r * 8 + col
                    val bit = if (dut.io.ser(r).peekBoolean()) 1 else 0
                    bitmaps(i) |= (bit << f)
                }
                dut.clock.step(timing.srclkPeriod)
            }
        }
        bitmaps
    }

    def feedBramFrame(dut: Top, rows: Array[BigInt], timing: TimingParams): Unit = {
        require(rows.length == 64)
        var steps = 0
        while (!dut.io.bramRen.peekBoolean() && steps < timing.cycleCycs * 2) {
            dut.clock.step()
            steps += 1
        }
        require(dut.io.bramRen.peekBoolean(), "timeout waiting for bramRen")

        for (addr <- 0 until 64) {
            dut.io.bramAddr.expect(addr.U)
            dut.io.bramRen.expect(true.B)
            dut.io.bramData.poke(rows(addr).U(32.W))
            dut.clock.step()
        }
        dut.io.bramRen.expect(false.B)
    }

    def assertBitmapsMatchBram(
        rows: Array[BigInt],
        bitmaps: Array[Int],
        timing: TimingParams
    ): Unit = {
        val lut = LutGenerator.generateLut(timing.div, timing.phaseDutyBits)
        for (i <- 0 until 64) {
            val (duty, phase) = dutyPhaseFromRow(rows(i))
            val expected = lut(duty)(phase).toInt
            withClue(s"transducer $i duty=$duty phase=$phase: ") {
                bitmaps(i) shouldBe expected
            }
        }
    }

    def runTopSerLoopback(
        rows: Array[BigInt],
        timing: TimingParams,
        dumpVcd: Boolean = false
    ): Unit = {
        def run(dut: Top): Unit = {
            feedBramFrame(dut, rows, timing)
            val bitmaps = sampleSerBitmaps(dut, timing)
            assertBitmapsMatchBram(rows, bitmaps, timing)
        }

        if (dumpVcd) {
            test(
                new Top(
                    timing.div,
                    timing.srclkLow,
                    timing.srclkHigh,
                    timing.rclkHigh,
                    timing.cycleCycs,
                    bramDepth
                )
            ).withAnnotations(Seq(WriteVcdAnnotation))(run)
        } else {
            test(
                new Top(
                    timing.div,
                    timing.srclkLow,
                    timing.srclkHigh,
                    timing.rclkHigh,
                    timing.cycleCycs,
                    bramDepth
                )
            )(run)
        }
    }

    behavior of "Top SER loopback"

    it should "reconstruct LUT bitmaps from SER (fast timing smoke test)" in {
        val cycleIdx = 1
        val duty = 2
        val phase = 1
        val row = BigInt(cycleIdx) << 16 | BigInt(duty) << 8 | BigInt(phase)
        val rows = Array.fill(64)(row)
        runTopSerLoopback(rows, fast)
    }

    it should "reconstruct LUT bitmaps from SER (production DIV=30 timing)" in {
        val cycleIdx = 1
        val duty = 10
        val phase = 7
        val row = BigInt(cycleIdx) << 16 | BigInt(duty) << 8 | BigInt(phase)
        val rows = Array.fill(64)(row)
        runTopSerLoopback(rows, production)
    }

    it should "output all-zero SER when cycle index does not match" in {
        test(new Top(fast.div, fast.srclkLow, fast.srclkHigh, fast.rclkHigh, fast.cycleCycs, bramDepth)) { dut =>
            val badRow = (BigInt(99) << 16) | (BigInt(10) << 8) | BigInt(7)
            feedBramFrame(dut, Array.fill(64)(badRow), fast)
            val bitmaps = sampleSerBitmaps(dut, fast)
            for (i <- 0 until 64) bitmaps(i) shouldBe 0
        }
    }

    val outputDir = "host/loopback_output"
    val testCases = Seq(
        "unet_1focus_center",
        "unet_1focus_offset",
        "unet_2foci",
        "unet_3foci",
        "gspat_1focus",
        "gspat_2foci",
    )

    for (name <- testCases) {
        val filePath = s"$outputDir/${name}_bram.bin"
        if (Files.exists(Paths.get(filePath))) {
            it should s"match LUT via SER for BRAM file $name (production timing)" in {
                runTopSerLoopback(readBramFile(filePath), production)
            }
        } else {
            it should s"find BRAM file for SER loopback $name" in { pending }
        }
    }

    it should "emit VCD for unet_1focus_center when TOP_SER_DUMP_VCD=1" in {
        assume(sys.env.getOrElse("TOP_SER_DUMP_VCD", "0") == "1", "set TOP_SER_DUMP_VCD=1 to run")
        val path = s"$outputDir/unet_1focus_center_bram.bin"
        assume(Files.exists(Paths.get(path)), "BRAM file required for VCD golden case")
        runTopSerLoopback(readBramFile(path), production, dumpVcd = true)
        info("VCD: build/chiselsim/<latest>/workdir-verilator/trace.vcd")
    }
}
