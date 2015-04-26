import sbt.Keys._
import sbt._

object Build extends sbt.Build {

  val sharedSettings = Seq(
    organization := "com.ngneers",
    scalaVersion := "2.11.6",
    version := "0.1-SNAPSHOT",

    scalacOptions ++= Seq(
      "-unchecked", "-deprecation", "-feature", "-Xfatal-warnings",
      "-Xlint", "-Xfuture",
      "-Yinline-warnings", "-Ywarn-adapted-args", "-Ywarn-inaccessible",
      "-Ywarn-nullary-override", "-Ywarn-nullary-unit", "-Yno-adapted-args"
    ),

    resolvers ++= Seq(
      "JBoss repository" at "https://repository.jboss.org/nexus/content/groups/public",
      "Scala-tools Maven2 Repository" at "http://scala-tools.org/repo-releases",
      Resolver.sonatypeRepo("snapshots"),
      Resolver.typesafeRepo("releases"),
      Resolver.mavenLocal
    ),

    libraryDependencies ++= {
      Seq(
        "com.typesafe.scala-logging" %% "scala-logging" % "3.1.0",
        "ch.qos.logback" % "logback-classic" % "1.1.2",
        "org.scalatest" %% "scalatest" % "2.2.4" % "test",
        "org.nd4j" % "nd4j-api"  % "0.0.3.5.5.2",
        "org.nd4j" % "nd4j-jblas" % "0.0.3.5.5.2",
        "com.typesafe" % "config" % "1.2.1"
      )
    },

    autoCompilerPlugins := true,
    /* Testing */
    testOptions in Test += Tests.Argument("-oDF"),
    parallelExecution in Test := false
  )

  lazy val root = (project in file("core"))
    .settings(name := "core")
    .settings(sharedSettings:_*)

}