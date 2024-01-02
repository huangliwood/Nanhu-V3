package xiangshan.vector.vbackend.vregfile

import org.chipsalliance.cde.config.Parameters
import chisel3._
import chisel3.experimental.prefix
import chisel3.util._
import difftest._
import freechips.rocketchip.diplomacy.{AdapterNode, LazyModule, LazyModuleImp, ValName}
import xiangshan.backend.execute.exu.{ExuConfig, ExuOutputNode, ExuOutwardImpl, ExuType}
import xiangshan.{ExuInput, ExuOutput, FuType, HasXSParameter, MicroOp, Redirect, SrcType, XSBundle}
import xiangshan.backend.regfile.{RegFileNode, ScalarRfReadPort}
import xiangshan.vector.HasVectorParameters
import xs.utils.{SignExt, UIntToMask, ZeroExt}
import VRegfileTopUtil._

object VrfHelper {
  def getElementIdx(segIdx: UInt, sew:UInt, VLEN: Int): UInt = {
    val vlenBytesLen = log2Ceil(VLEN / 8)
    val regElmIdx = MuxCase(0.U, Seq(
      (sew === 0.U) -> segIdx(vlenBytesLen - 1, 0),
      (sew === 1.U) -> segIdx(vlenBytesLen - 2, 0),
      (sew === 2.U) -> segIdx(vlenBytesLen - 3, 0),
      (sew === 3.U) -> segIdx(vlenBytesLen - 4, 0),
    ))
    regElmIdx
  }

  def extractElement(segIdx:UInt, vsrc:UInt, sew:UInt, VLEN: Int, XLEN: Int): UInt = {
    require(vsrc.getWidth == VLEN)
    val elemsIdx = getElementIdx(segIdx, sew, VLEN)
    val res = WireInit(0.U(XLEN.W))
    val vsrcSplit8 = VecInit(Seq.tabulate(VLEN / 8)(idx => vsrc(idx * 8 + 7, idx * 8)))
    val vsrcSplit16 = VecInit(Seq.tabulate(VLEN / 16)(idx => vsrc(idx * 16 + 15, idx * 16)))
    val vsrcSplit32 = VecInit(Seq.tabulate(VLEN / 32)(idx => vsrc(idx * 32 + 31, idx * 32)))
    val vsrcSplit64 = VecInit(Seq.tabulate(VLEN / 64)(idx => vsrc(idx * 64 + 63, idx * 64)))
    res := MuxCase(0.U, Seq(
      (sew === 0.U) -> ZeroExt(vsrcSplit8(elemsIdx(log2Ceil(VLEN / 8) - 1, 0)), XLEN),
      (sew === 1.U) -> ZeroExt(vsrcSplit16(elemsIdx(log2Ceil(VLEN / 16) - 1, 0)), XLEN),
      (sew === 2.U) -> ZeroExt(vsrcSplit32(elemsIdx(log2Ceil(VLEN / 32) - 1, 0)), XLEN),
      (sew === 3.U) -> ZeroExt(vsrcSplit64(elemsIdx(log2Ceil(VLEN / 64) - 1, 0)), XLEN),
    ))
    res
  }
}

class VectorWritebackMergeNode(implicit valName: ValName) extends AdapterNode(ExuOutwardImpl)({p => p.copy(throughVectorRf = true)}, {p => p})

class VectorRfReadPort(implicit p:Parameters) extends XSBundle{
  val addr = Input(UInt(PhyRegIdxWidth.W))
  val data = Output(UInt(VLEN.W))
}

object VRegfileTopUtil{
  def GenLoadVrfMask(in:MicroOp, VLEN:Int):UInt = {
    val width = VLEN / 8
    val vlenBytes = log2Ceil(VLEN / 8)
    val sew = in.vctrl.eew(2)
    val movIdx = VrfHelper.getElementIdx(in.segIdx, sew, VLEN)
    val mask = MuxCase(0.U, Seq(
      (sew === 0.U) -> ("h01".U << Cat(movIdx(vlenBytes - 1, 0), 0.U(0.W))),
      (sew === 1.U) -> ("h03".U << Cat(movIdx(vlenBytes - 2, 0), 0.U(1.W))),
      (sew === 2.U) -> ("h0f".U << Cat(movIdx(vlenBytes - 3, 0), 0.U(2.W))),
      (sew === 3.U) -> ("hff".U << Cat(movIdx(vlenBytes - 4, 0), 0.U(3.W))),
    ))
    Mux(in.loadStoreEnable, mask(width - 1, 0).asUInt, 0.U(width.W))
  }
}

class VRegfileTop(extraVectorRfReadPort: Int)(implicit p:Parameters) extends LazyModule with HasXSParameter with HasVectorParameters{
  val issueNode = new RegFileNode
  val writebackMergeNode = new VectorWritebackMergeNode
  lazy val module = new Impl
  class Impl extends LazyModuleImp(this){
    val rfReadNum:Int = issueNode.in.length

    val io = IO(new Bundle {
      val hartId = Input(UInt(64.W))
      val vectorReads = Vec(extraVectorRfReadPort, new VectorRfReadPort)
      val scalarReads = Vec(rfReadNum, Flipped(new ScalarRfReadPort))
      val moveOldValReqs = Input(Vec(loadUnitNum, Valid(new MoveReq)))
      val vecAllocPregs = Vec(vectorParameters.vRenameWidth, Flipped(ValidIO(UInt(PhyRegIdxWidth.W))))
      val debug_vec_rat = Input(Vec(32, UInt(PhyRegIdxWidth.W)))
      val redirect = Input(Valid(new Redirect))
    })

    private val fromVectorFu = writebackMergeNode.in.map(e => (e._1, e._2._1))
    private val toWritebackNetwork = writebackMergeNode.out.map(e => (e._1, e._2._1))

    private val wbVFUPair = fromVectorFu.zip(toWritebackNetwork).map(e => {
      require(e._1._2.name == e._2._2.name && e._1._2.id == e._2._2.id)
      require(e._1._2.writeVecRf || e._1._2.exuType == ExuType.sta || e._1._2.exuType == ExuType.std)
      (e._1._1, e._2._1, e._1._2)
    })
    require(issueNode.in.length == 1)

    private val wbPairNeedMerge = wbVFUPair.filter(_._3.willTriggerVrfWkp)
    private val wbPairDontNeedMerge = wbVFUPair.filterNot(_._3.willTriggerVrfWkp).filterNot(_._3.exuType == ExuType.sta).filterNot(_._3.exuType == ExuType.std)
    private val wbPairStu = wbVFUPair.filter(p => p._3.exuType == ExuType.sta || p._3.exuType == ExuType.std)

    private val fromRs = issueNode.in.flatMap(i => i._1.zip(i._2._2).map(e => (e._1, e._2, i._2._1)))
    private val toExuMap = issueNode.out.map(i => i._2._2 -> (i._1, i._2._2, i._2._1)).toMap

    private val readPortsNum = fromRs.length * 4 + extraVectorRfReadPort

    private val vrf = Module(new VRegfile(wbPairNeedMerge.length, wbPairDontNeedMerge.length, readPortsNum))
    vrf.io.vecAllocPregs.zip(io.vecAllocPregs).foreach({case(a, b) => a := Pipe(b)})

    println("====================VRF writeback port:====================")
    wbVFUPair.foreach(e => print(e._3))

    println("\n====================VRF writeback port need merged:====================")
    wbPairNeedMerge.foreach(e => print(e._3))

    println("\n====================VRF writeback port not need merged:====================")
    wbPairDontNeedMerge.foreach(e => print(e._3))

    vrf.io.wbWakeup.zip(vrf.io.wakeupMask).zip(wbPairNeedMerge).foreach({case((rfwb, rfwkp),(wbin, wbout, cfg)) =>
      if(cfg.exuType == ExuType.ldu){
        val sew = wbin.bits.uop.vctrl.eew(2)
        val bitsWire = WireInit(wbin.bits)
        bitsWire.data := MuxCase(0.U, Seq(
          (sew === 0.U) -> Cat(Seq.fill(VLEN / 8)(wbin.bits.data(7, 0))),
          (sew === 1.U) -> Cat(Seq.fill(VLEN / 16)(wbin.bits.data(15, 0))),
          (sew === 2.U) -> Cat(Seq.fill(VLEN / 32)(wbin.bits.data(31, 0))),
          (sew === 3.U) -> Cat(Seq.fill(VLEN / 64)(wbin.bits.data(63, 0)))
        ))
        val validReg = RegNext(wbin.valid && !wbin.bits.uop.robIdx.needFlush(io.redirect), false.B)
        val bitsReg = RegEnable(bitsWire, wbin.valid)
        val lmask = GenLoadVrfMask(bitsReg.uop, VLEN)
        rfwb.valid := validReg
        rfwb.bits := bitsReg
        rfwb.bits.wakeupMask := lmask
        rfwb.bits.writeDataMask := Mux(bitsReg.uop.cf.exceptionVec.reduce(_ | _), 0.U((VLEN / 8).W), lmask)
        rfwb.bits.redirectValid := false.B
        rfwb.bits.redirect := DontCare
      } else {
        rfwb.valid := wbin.valid
        rfwb.bits := wbin.bits
        rfwb.bits.writeDataMask := Mux(wbin.bits.uop.ctrl.vdWen, wbin.bits.writeDataMask, 0.U)
        rfwb.bits.wakeupMask := Mux(wbin.bits.uop.ctrl.vdWen, wbin.bits.wakeupMask, 0.U)
      }
      val wbBitsReg = RegEnable(rfwb.bits, rfwb.valid)
      wbout.valid := RegNext(rfwb.valid && !rfwb.bits.uop.robIdx.needFlush(io.redirect), false.B)
      wbout.bits := wbBitsReg
      if(cfg.exuType == ExuType.ldu){
        wbout.bits.wakeupValid := rfwkp.andR
        wbout.bits.uop.uopIdx := wbout.bits.uop.segIdx
      } else {
        wbout.bits.wakeupValid := rfwkp.andR || wbBitsReg.uop.ctrl.fpWen || wbBitsReg.uop.ctrl.rfWen
      }
    })
    vrf.io.wbNoWakeup.zip(wbPairDontNeedMerge).foreach({case(rfwb, (wbin, wbout, _)) =>
      rfwb.valid := wbin.valid && wbin.bits.uop.ctrl.vdWen
      rfwb.bits := wbin.bits
      wbout.valid := RegNext(wbin.valid & !wbin.bits.uop.robIdx.needFlush(io.redirect), false.B)
      wbout.bits := RegEnable(wbin.bits, wbin.valid)
      wbout.bits.wakeupValid := true.B
      wbout.bits.redirectValid := false.B
      wbout.bits.redirect := DontCare
    })
    wbPairStu.foreach({case(wbin, wbout, cfg) =>
      val validReg = RegNext(wbin.valid & !wbin.bits.uop.robIdx.needFlush(io.redirect), false.B)
      val bitsReg = RegEnable(wbin.bits, wbin.valid)
      val redirectValidReg = RegNext(wbin.bits.redirectValid & !wbin.bits.redirect.robIdx.needFlush(io.redirect), false.B)
      val redirectBitsReg = RegEnable(wbin.bits.redirect, wbin.bits.redirectValid)
      wbout.valid := validReg && !bitsReg.uop.robIdx.needFlush(io.redirect)
      wbout.bits := bitsReg
      if (cfg.exuType == ExuType.sta || cfg.exuType == ExuType.std) {
        wbout.bits.uop.uopIdx := wbout.bits.uop.segIdx
      }
      wbout.bits.redirectValid := redirectValidReg && !redirectBitsReg.robIdx.needFlush(io.redirect)
      wbout.bits.redirect := redirectBitsReg
    })
    vrf.io.moveOldValReqs.zip(io.moveOldValReqs).foreach({case(a, b) => a := Pipe(b)})
    vrf.io.readPorts.take(extraVectorRfReadPort).zip(io.vectorReads).foreach({case(rr, ir) =>
      rr.addr := ir.addr
      ir.data := rr.data
    })
    private val lduWbs = toWritebackNetwork.filter(_._2.exuType == ExuType.ldu)
    lduWbs.zipWithIndex.foreach({case(lwb, i) =>
      val preLduWbs = lduWbs.take(i)
      val kill = (preLduWbs.map(_._1).map(l => l.valid && l.bits.uop.pdest === lwb._1.bits.uop.pdest && l.bits.wakeupValid) :+ false.B).reduce(_||_)
      when(kill){
        lwb._1.bits.wakeupValid := false.B
      }
    })

    private var vecReadPortIdx = extraVectorRfReadPort
    private var scalarReadPortIdx = 0
    for (in <- fromRs) {
      val out = toExuMap(in._2)
      val bi = in._1
      val bo = out._1
      val exuInBundle = WireInit(bi.issue.bits)
      exuInBundle.src := DontCare
      io.scalarReads(scalarReadPortIdx).addr := bi.specialPsrc
      val idata = RegEnable(io.scalarReads(scalarReadPortIdx).idata, bi.specialPsrcRen)
      val fdata = RegEnable(io.scalarReads(scalarReadPortIdx).fdata, bi.specialPsrcRen)
      vrf.io.readPorts(vecReadPortIdx).addr := bi.issue.bits.uop.psrc(0)
      vrf.io.readPorts(vecReadPortIdx + 1).addr := bi.issue.bits.uop.psrc(1)
      vrf.io.readPorts(vecReadPortIdx + 2).addr := bi.issue.bits.uop.psrc(2)
      vrf.io.readPorts(vecReadPortIdx + 3).addr := bi.issue.bits.uop.vm

      exuInBundle.src(0) := MuxCase(vrf.io.readPorts(vecReadPortIdx).data, Seq(
        SrcType.isReg(bi.issue.bits.uop.ctrl.srcType(0)) -> idata,
        SrcType.isFp(bi.issue.bits.uop.ctrl.srcType(0)) -> fdata,
        SrcType.isVec(bi.issue.bits.uop.ctrl.srcType(0)) -> vrf.io.readPorts(vecReadPortIdx).data,
        SrcType.isImm(bi.issue.bits.uop.ctrl.srcType(0)) -> SignExt(bi.issue.bits.uop.ctrl.imm(4,0), VLEN)
      ))
      exuInBundle.src(1) := vrf.io.readPorts(vecReadPortIdx + 1).data
      exuInBundle.src(2) := vrf.io.readPorts(vecReadPortIdx + 2).data
      exuInBundle.vm := vrf.io.readPorts(vecReadPortIdx + 3).data

      val issValidReg = RegInit(false.B)
      val issDataReg = Reg(new ExuInput())
      val allowPipe = !issValidReg || bo.issue.ready || (issValidReg && issDataReg.uop.robIdx.needFlush(io.redirect))
      bo.issue.valid := issValidReg
      bo.issue.bits := issDataReg
      when(allowPipe){
        issValidReg := bi.issue.valid
      }
      when(bi.issue.fire) {
        issDataReg := exuInBundle
      }
      bi.issue.ready := allowPipe

      bo.rsIdx := DontCare
      bi.rsFeedback := DontCare
      bo.hold := false.B

      scalarReadPortIdx = scalarReadPortIdx + 1
      vecReadPortIdx = vecReadPortIdx + 4
    }

    if(vrf.io.debug.isDefined) vrf.io.debug.get.zipWithIndex.foreach { case (rp, i) => rp.addr := io.debug_vec_rat(i) }

    if (env.EnableDifftest) {
      val difftestArchVec = DifftestModule(new DiffArchVecRegState)
      difftestArchVec.coreid := io.hartId
      vrf.io.debug.get.zipWithIndex.foreach {
        case (rp, i) => {
          difftestArchVec.value(i*2) := rp.data(VLEN/2-1, 0)
          difftestArchVec.value(i*2+1) := rp.data(VLEN-1, VLEN/2)
        }
      }
    }
  }
}
