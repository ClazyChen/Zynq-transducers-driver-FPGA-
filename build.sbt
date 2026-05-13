ThisBuild / version := "0.1.0"
ThisBuild / scalaVersion := "2.13.14"

lazy val root = (project in file("."))
  .settings(
    name := "zynq-ultrasound-fpga",
    libraryDependencies ++= Seq(
      "org.chipsalliance" %% "chisel" % "6.6.0",
      "edu.berkeley.cs" %% "chiseltest" % "6.0.0" % Test
    ),
    addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % "6.6.0" cross CrossVersion.full),
    scalacOptions ++= Seq(
      "-unchecked",
      "-deprecation",
      "-language:reflectiveCalls",
      "-feature",
      "-Xcheckinit"
    )
  )
