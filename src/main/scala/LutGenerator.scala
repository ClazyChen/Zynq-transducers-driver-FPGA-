package ultrasound

import chisel3._
import chisel3.util._

/**
  * Ultrasound frame LUT generator.
  * 
  * Generates a compile-time constant lookup table that maps
  * (duty, phase) -> DIV-bit frame bitmap for a single transducer.
  * 
  * The mapping follows PWM semantics: for frame index f (0 <= f < DIV),
  * output is HIGH iff ((f - phase + DIV) % DIV) < duty.
  * 
  * Parameters:
  *   DIV           - number of frames per 40kHz cycle
  *   phaseDutyBits - bit width of duty/phase indices (e.g. 5 for DIV=30)
  */
object LutGenerator {

    /**
      * Generate the LUT as a Scala Seq[Seq[BigInt]].
      * Each entry is a DIV-bit bitmap (packed into BigInt).
      */
    def generateLut(div: Int, phaseDutyBits: Int): Seq[Seq[BigInt]] = {
        val lutSize = 1 << phaseDutyBits
        (0 until lutSize).map { duty =>
            (0 until lutSize).map { phase =>
                val bits = (0 until div).map { frame =>
                    // Use double div offset to ensure positive value before modulo
                    val pos = (frame - phase + div * 2) % div
                    if (pos < duty) 1 else 0
                }
                // Pack bits: frame 0 is LSB
                bits.zipWithIndex
                    .filter(_._1 == 1)
                    .map(_._2)
                    .foldLeft(BigInt(0))((acc, bitIdx) => acc | (BigInt(1) << bitIdx))
            }
        }
    }

    /**
      * Create a Chisel Vec-of-Vec constant from the Scala LUT.
      */
    def chiselLut(div: Int, phaseDutyBits: Int): Vec[Vec[UInt]] = {
        val lut = generateLut(div, phaseDutyBits)
        VecInit(lut.map(row => VecInit(row.map(_.U(div.W)))))
    }
}
