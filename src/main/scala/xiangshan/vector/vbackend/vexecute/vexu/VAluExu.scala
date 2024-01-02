package xiangshan.vector.vbackend.vexecute.vexu
import org.chipsalliance.cde.config.Parameters
import chisel3._
import chisel3.util._
import darecreek.exu.vfu.alu.VAlu
import darecreek.exu.vfu.vmask.VMask
import xiangshan.{HasXSParameter, SrcType}
import xiangshan.backend.execute.exu.{BasicExu, BasicExuImpl, ExuConfig, ExuInputNode, ExuOutputNode, ExuType}
import xiangshan.backend.execute.fu.FuConfigs
import xiangshan.vector.{EewType, HasVectorParameters}
import xiangshan.vector.vbackend.vexecute.vfu.s2v.Scalar2Vector
import xiangshan.vector.vbackend.vexecute.vfu.uopToVuop
import xiangshan.backend.execute.fu.fpu.FPU
class VAluExu(id:Int, complexName:String)(implicit p: Parameters) extends BasicExu{
  private val cfg = ExuConfig(
    name = "VAluExu",
    id = id,
    complexName = complexName,
    fuConfigs = Seq(FuConfigs.valuCfg, FuConfigs.vmaskCfg, FuConfigs.s2vCfg),
    exuType = ExuType.valu,
    writebackToRob = false,
    writebackToVms = true
  )
  val issueNode = new ExuInputNode(cfg)
  val writebackNode = new ExuOutputNode(cfg)

  lazy val module = new Impl
  class Impl extends BasicExuImpl(this) with HasXSParameter with HasVectorParameters {
    require(issueNode.in.length == 1)
    require(writebackNode.out.length == 1)
    val io = IO(new Bundle{
      val vstart = Input(UInt(log2Up(VLEN).W))
      val vcsr = Input(UInt(3.W))
      val frm = Input(UInt(3.W))
    })

    def latency = 2
    private val iss = issueNode.in.head._1.issue
    private val wb = writebackNode.out.head._1
    iss.ready := true.B

    private val valu = Module(new VAlu)
    private val vmask = Module(new VMask)
    //private val vred = Module(new Reduction)
    private val s2v = Module(new Scalar2Vector)
    private val uopShiftQueue = Module(new MicroOpShiftQueue(latency))

    private val vuop = uopToVuop(iss.bits.uop, iss.valid, io.vstart, io.vcsr(2,1), io.frm, p)
    private val src0 = iss.bits.src(0)
    private val src1 = iss.bits.src(1)
    private val src2 = iss.bits.src(2)
    private val mask = iss.bits.vm

    uopShiftQueue.io.in.valid := iss.valid && cfg.fuConfigs.map(_.fuType === iss.bits.uop.ctrl.fuType).reduce(_|_) && !iss.bits.uop.robIdx.needFlush(redirectIn)
    uopShiftQueue.io.in.bits := iss.bits.uop
    uopShiftQueue.io.redirect := redirectIn

    valu.io.in.valid := iss.valid && iss.bits.uop.ctrl.fuType === FuConfigs.valuCfg.fuType && !iss.bits.uop.robIdx.needFlush(redirectIn)
    valu.io.in.bits.uop := vuop
    valu.io.in.bits.vs1 := src0
    valu.io.in.bits.vs2 := src1
    valu.io.in.bits.rs1 := src0(XLEN - 1, 0)
    valu.io.in.bits.oldVd := src2
    valu.io.in.bits.mask := mask

    vmask.io.in.valid := iss.valid && iss.bits.uop.ctrl.fuType === FuConfigs.vmaskCfg.fuType && !iss.bits.uop.robIdx.needFlush(redirectIn)
    vmask.io.in.bits.uop := vuop
    vmask.io.in.bits.vs1 := src0
    vmask.io.in.bits.vs2 := src1
    vmask.io.in.bits.rs1 := src0(XLEN - 1, 0)
    vmask.io.in.bits.oldVd := src2
    vmask.io.in.bits.mask := mask

    private val rs1IsFp = iss.bits.uop.ctrl.srcType(0) === SrcType.fp
    private val typeTag = Mux(iss.bits.uop.vCsrInfo.vsew === 2.U, FPU.S, FPU.D)
    s2v.io.in.valid := iss.valid && iss.bits.uop.ctrl.fuType === FuConfigs.s2vCfg.fuType && !iss.bits.uop.robIdx.needFlush(redirectIn)
    s2v.io.in.bits.uop := vuop
    s2v.io.in.bits.vs1 := src0
    s2v.io.in.bits.vs2 := src1
    s2v.io.in.bits.rs1 := Mux(rs1IsFp, FPU.unbox(src0(XLEN - 1, 0), typeTag), src0(XLEN - 1, 0))
    s2v.io.in.bits.oldVd := src2
    s2v.io.in.bits.mask := mask

    private val uopIdx = uopShiftQueue.io.out.bits.uopIdx
    private val uopNum = uopShiftQueue.io.out.bits.uopNum
    private val uopOut = uopShiftQueue.io.out.bits
    private val isNarrow = uopOut.vctrl.isNarrow && !uopOut.vctrl.maskOp
    private val isVcpop = uopOut.vctrl.ff
    private val lowHalf = !uopIdx(0)
    private val highHalf = uopIdx(0)
    private val maskLen = VLEN / 8
    private val halfMaskLen = maskLen / 2
    private def ones(in:Int):UInt = ((1 << in) - 1).U(in.W)

    private val lowHalfMask = Cat(0.U(halfMaskLen.W), ones(halfMaskLen))
    private val highHalfMask = Cat(ones(halfMaskLen), 0.U(halfMaskLen.W))
    private val fullMask = ones(maskLen)
    private val finalMask = MuxCase(fullMask, Seq(
      (isNarrow && lowHalf) -> lowHalfMask,
      (isNarrow && highHalf) -> highHalfMask,
    ))

    private val validSeq = Seq(valu.io.out.valid, vmask.io.out.valid, s2v.io.out.valid)
    assert(PopCount(validSeq) <= 1.U)
    private val dataSeq = Seq(valu.io.out.bits, vmask.io.out.bits, s2v.io.out.bits)
    private val wbData = Mux1H(validSeq, dataSeq)

    wb.valid := uopShiftQueue.io.out.valid && Mux(isVcpop, uopIdx === (uopNum - 1.U), true.B) && validSeq.reduce(_||_)
    wb.bits := DontCare
    wb.bits.uop := uopShiftQueue.io.out.bits
    wb.bits.data := wbData.vd
    wb.bits.vxsat := wbData.vxsat
    wb.bits.wakeupMask := Mux(uopNum === 1.U, fullMask, finalMask)
    wb.bits.writeDataMask := Mux(uopNum === 1.U, fullMask, finalMask)
  }
}
