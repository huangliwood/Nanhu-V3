/***************************************************************************************
* Copyright (c) 2020-2021 Institute of Computing Technology, Chinese Academy of Sciences
* Copyright (c) 2020-2021 Peng Cheng Laboratory
*
* XiangShan is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
*
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
*
* See the Mulan PSL v2 for more details.
***************************************************************************************/

package top

import chisel3._
import xiangshan._
import utils._
import system._
import chisel3.stage.ChiselGeneratorAnnotation
import chipsalliance.rocketchip.config._
import freechips.rocketchip.diplomacy._
import freechips.rocketchip.tilelink._
import freechips.rocketchip.jtag.JTAGIO
import freechips.rocketchip.util.{ElaborationArtefacts, HasRocketChipStageUtils}
import huancun.{HCCacheParamsKey, HuanCun}
import xs.utils.{ResetGen, DFTResetSignals}
import xs.utils.sram.BroadCastBundle
import coupledL3._

abstract class BaseXSSoc()(implicit p: Parameters) extends LazyModule
  with BindingScope
{
  val misc = LazyModule(new SoCMisc())
  lazy val dts = DTS(bindingTree)
  lazy val json = JSON(bindingTree)
}

class XSTop()(implicit p: Parameters) extends BaseXSSoc() with HasSoCParameter
{
  ResourceBinding {
    val width = ResourceInt(2)
    val model = "freechips,rocketchip-unknown"
    Resource(ResourceAnchors.root, "model").bind(ResourceString(model))
    Resource(ResourceAnchors.root, "compat").bind(ResourceString(model + "-dev"))
    Resource(ResourceAnchors.soc, "compat").bind(ResourceString(model + "-soc"))
    Resource(ResourceAnchors.root, "width").bind(width)
    Resource(ResourceAnchors.soc, "width").bind(width)
    Resource(ResourceAnchors.cpus, "width").bind(ResourceInt(1))
    def bindManagers(xbar: TLNexusNode) = {
      ManagerUnification(xbar.edges.in.head.manager.managers).foreach{ manager =>
        manager.resources.foreach(r => r.bind(manager.toResource))
      }
    }
    bindManagers(misc.l3_xbar.asInstanceOf[TLNexusNode])
    bindManagers(misc.peripheralXbar.asInstanceOf[TLNexusNode])
  }

  println(s"FPGASoC cores: $NumCores banks: $L3NBanks block size: $L3BlockSize bus size: $L3OuterBusWidth")

  val core_with_l2 = tiles.zipWithIndex.map({case (coreParams,idx) =>
    LazyModule(new XSTile(s"XSTop_XSTile_")(p.alterPartial({
      case XSCoreParamsKey => coreParams
    })))
  })

  // val l3cacheOpt = soc.L3CacheParamsOpt.map(l3param =>
  //   LazyModule(new HuanCun("XSTop_L3_")(new Config((_, _, _) => {
  //     case HCCacheParamsKey => l3param.copy(enableTopDown = debugOpts.EnableTopDown)
  //   })))
  // )
  val l3cacheOpt = soc.L3CacheParamsOpt.map(l3param =>
    LazyModule(new HuanCun("XSTop_L3_")(new Config((_, _, _) => {
      case HCCacheParamsKey => l3param.copy(enableTopDown = debugOpts.EnableTopDown)
    })))
  )
  // val l3cacheOpt = soc.L3CacheParamsOpt.map(l3param =>
  //   LazyModule(new CoupledL3()(new Config((_, _, _) => {
  //     case L3ParamKey => l3param
  //   })))
  // )

  for (i <- 0 until NumCores) {
    core_with_l2(i).clint_int_sink := misc.clint.intnode
    core_with_l2(i).plic_int_sink :*= misc.plic.intnode
    core_with_l2(i).debug_int_sink := misc.debugModule.debug.dmOuter.dmOuter.intnode
    misc.plic.intnode := IntBuffer() := core_with_l2(i).beu_int_source
    misc.peripheral_ports(i) := core_with_l2(i).uncache
    misc.core_to_l3_ports(i) :=* core_with_l2(i).memory_port
  }

   (core_with_l2.head.l2cache.get.spp_send_node, core_with_l2.last.l2cache.get.spp_send_node) match{
     case(Some(l2_0),Some(l2_1))=>{
       val l3pf_RecvXbar = LazyModule(new coupledL2.prefetch.PrefetchReceiverXbar(NumCores))
       for (i <- 0 until NumCores) {
         println(s"Connecting L2 prefecher_sender_${i} to L3!")
         l3pf_RecvXbar.inNode(i) := core_with_l2(i).l2cache.get.spp_send_node.get
       }
       l3cacheOpt.get.pf_l3recv_node.map(l3Recv => l3Recv := l3pf_RecvXbar.outNode.head)
     }
     case(_,_) => None
   }

  // val core_rst_nodes = if(l3cacheOpt.nonEmpty && l3cacheOpt.get.rst_nodes.nonEmpty){
  //   l3cacheOpt.get.rst_nodes.get
  // } else {
  //   core_with_l2.map(_ => BundleBridgeSource(() => Reset()))
  // }
  val core_rst_nodes = core_with_l2.map(_ => BundleBridgeSource(() => Reset()))

  core_rst_nodes.zip(core_with_l2.map(_.core_reset_sink)).foreach({
    case (source, sink) =>  sink := source
  })

  l3cacheOpt match {
    case Some(l3) =>
      misc.l3_out :*= l3.node :*= TLBuffer.chainNode(2) :*= misc.l3_banked_xbar
    case None =>
  }

  lazy val module = new LazyRawModuleImp(this) {
    ElaborationArtefacts.add("dts", dts)
    ElaborationArtefacts.add("graphml", graphML)
    ElaborationArtefacts.add("json", json)
    ElaborationArtefacts.add("plusArgs", freechips.rocketchip.util.PlusArgArtefacts.serialize_cHeader())

    val dma = IO(Flipped(misc.dma.cloneType))
    val peripheral = IO(misc.peripheral.cloneType)
    val memory = IO(misc.memory.cloneType)

    misc.dma <> dma
    peripheral <> misc.peripheral
    memory <> misc.memory

    val io = IO(new Bundle {
      val clock = Input(Bool())
      val reset = Input(AsyncReset())
      val extIntrs = Input(UInt(NrExtIntr.W))
      val systemjtag = new Bundle {
        val jtag = Flipped(new JTAGIO(hasTRSTn = false))
        val reset = Input(AsyncReset()) // No reset allowed on top
        val mfr_id = Input(UInt(11.W))
        val part_number = Input(UInt(16.W))
        val version = Input(UInt(4.W))
      }
      val debug_reset = Output(Bool())
      val riscv_halt = Output(Vec(NumCores, Bool()))
    })

    val scan_mode = IO(Input(Bool()))
    val dft_lgc_rst_n = IO(Input(AsyncReset()))
    val dft_mode = IO(Input(Bool()))
    val dfx_reset = Wire(new DFTResetSignals())
    dfx_reset.lgc_rst_n := dft_lgc_rst_n
    dfx_reset.mode := dft_mode
    dfx_reset.scan_mode := scan_mode

    val reset_sync = withClockAndReset(io.clock.asClock, io.reset) { ResetGen(2, Some(dfx_reset)) }
    val jtag_reset_sync = withClockAndReset(io.systemjtag.jtag.TCK, io.systemjtag.reset) { ResetGen(2, Some(dfx_reset)) }

    // override LazyRawModuleImp's clock and reset
    childClock := io.clock.asClock
    childReset := reset_sync

    // output
    io.debug_reset := misc.module.debug_module_io.debugIO.ndreset

    // input
    dontTouch(dma)
    dontTouch(io)
    dontTouch(peripheral)
    dontTouch(memory)
    misc.module.ext_intrs := io.extIntrs

    for ((core, i) <- core_with_l2.zipWithIndex) {
      core.moduleInstance.io.hartId := i.U
      core.moduleInstance.io.dfx_reset:= dfx_reset
      io.riscv_halt(i) := core.moduleInstance.io.cpu_halt
    }

    // if(l3cacheOpt.isEmpty || l3cacheOpt.get.rst_nodes.isEmpty){
    //   // tie off core soft reset
    //   for(node <- core_rst_nodes){
    //     node.out.head._1 := false.B.asAsyncReset()
    //   }
    //   if(l3cacheOpt.get.module.dfx_reset.isDefined) {
    //     l3cacheOpt.get.module.dfx_reset.get := dfx_reset
    //   }
    // }
    // if(l3cacheOpt.isEmpty || l3cacheOpt.get.rst_nodes.isEmpty){
      // tie off core soft reset
      for(node <- core_rst_nodes){
        node.out.head._1 := false.B.asAsyncReset()
      }
      // if(l3cacheOpt.get.module.dfx_reset.isDefined) {
      //   l3cacheOpt.get.module.dfx_reset.get := dfx_reset
      // }
    // }

    misc.module.debug_module_io.resetCtrl.hartIsInReset := core_with_l2.map(_.moduleInstance.ireset.asBool)
    misc.module.debug_module_io.clock := io.clock
    misc.module.debug_module_io.reset := misc.module.reset

    misc.module.debug_module_io.debugIO.reset := misc.module.reset
    misc.module.debug_module_io.debugIO.clock := io.clock.asClock
    // TODO: delay 3 cycles?
    misc.module.debug_module_io.debugIO.dmactiveAck := misc.module.debug_module_io.debugIO.dmactive
    // jtag connector
    misc.module.debug_module_io.debugIO.systemjtag.foreach { x =>
      x.jtag        <> io.systemjtag.jtag
      x.reset       := jtag_reset_sync
      x.mfr_id      := io.systemjtag.mfr_id
      x.part_number := io.systemjtag.part_number
      x.version     := io.systemjtag.version
    }

    val mbistBroadCastToTile = if(core_with_l2.head.moduleInstance.dft.isDefined) {
      val res = Some(Wire(new BroadCastBundle))
      core_with_l2.foreach(_.moduleInstance.dft.get := res.get)
      res
    } else {
      None
    }
    val mbistBroadCastToL3 = if(l3cacheOpt.isDefined) {
      // if(l3cacheOpt.get.module.dft.isDefined){
      //   val res = Some(Wire(new BroadCastBundle))
      //   l3cacheOpt.get.module.dft.get := res.get
      //   res
      // } else {
      //   None
      // }
      None
    } else {
      None
    }
    val dft = if(mbistBroadCastToTile.isDefined || mbistBroadCastToL3.isDefined){
      Some(IO(new BroadCastBundle))
    } else {
      None
    }
    if(dft.isDefined){
      if(mbistBroadCastToTile.isDefined){
        mbistBroadCastToTile.get := dft.get
      }
      if(mbistBroadCastToL3.isDefined){
        // mbistBroadCastToL3.get := dft.get
      }
    }

    withClockAndReset(io.clock.asClock, reset_sync) {
      // Modules are reset one by one
      // reset ----> SYNC --> {SoCMisc, L3 Cache, Cores}
      val coreResetChain:Seq[Reset] = core_with_l2.map(_.moduleInstance.ireset)
      val resetChain = Seq(misc.module.reset) ++ l3cacheOpt.map(_.module.reset) ++ coreResetChain
      val resetDftSigs = ResetGen.applyOneLevel(resetChain, reset_sync, !debugOpts.FPGAPlatform)
      resetDftSigs:= dfx_reset
    }
  }
}

object TopMain extends App with HasRocketChipStageUtils {
  override def main(args: Array[String]): Unit = {
    val (config, firrtlOpts) = ArgParser.parse(args)
    val soc = DisableMonitors(p => LazyModule(new XSTop()(p)))(config)
    XiangShanStage.execute(firrtlOpts, Seq(
      ChiselGeneratorAnnotation(() => {
        soc.module
      })
    ))
    ElaborationArtefacts.files.foreach{ case (extension, contents) =>
      writeOutputFile("./build", s"XSTop.${extension}", contents())
    }
  }
}
