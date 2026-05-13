package ultrasound

import circt.stage.ChiselStage

/**
  * Standalone object to emit SystemVerilog for the Top module.
  * Run with: sbt "runMain ultrasound.GenerateVerilog"
  */
object GenerateVerilog extends App {
    ChiselStage.emitSystemVerilogFile(
        new Top(
            div = 30,
            srclkLowCycs = 5,
            srclkHighCycs = 5,
            rclkHighCycs = 5,
            cycleCycs = 2500,
            bramDepth = 1024
        ),
        Array("--target-dir", "generated"),
        Array("--preserve-aggregate=1d-vec", "--disable-all-randomization")
    )
}
