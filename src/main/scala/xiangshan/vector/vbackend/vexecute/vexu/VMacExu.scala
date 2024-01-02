package xiangshan.vector.vbackend.vexecute.vexu
import org.chipsalliance.cde.config.Parameters
import chisel3._
import chisel3.util._
import darecreek.exu.vfu.mac.VMac
import darecreek.exu.vfu.reduction.Reduction
import xiangshan.HasXSParameter
import xiangshan.backend.execute.exu.{BasicExu, BasicExuImpl, ExuConfig, ExuInputNode, ExuOutputNode, ExuType}
import xiangshan.backend.execute.fu.FuConfigs
import xiangshan.vector.HasVectorParameters
import xiangshan.vector.vbackend.vexecute.vfu.uopToVuop
class VMacExu(id:Int, complexName:String)(implicit p: Parameters) extends BasicExu{
  private val cfg = ExuConfig(
    name = "VMacExu",
    id = id,
    complexName = complexName,
    fuConfigs = Seq(FuConfigs.vmacCfg, FuConfigs.vredCfg),
    exuType = ExuType.vmac,
    writebackToRob = false,
    writebackToVms = true
  )
  val issueNode = new ExuInputNode(cfg)
  val writebackNode = new ExuOutputNode(cfg)

  lazy val module = new Impl
  class Impl extends BasicExuImpl(this) with HasXSParameter with HasVectorParameters {
    require(issueNode.in.length == 1)
    require(writebackNode.out.length == 1)
    val io = IO(new Bundle {
      val vstart = Input(UInt(log2Up(VLEN).W))
      val vcsr = Input(UInt(3.W))
      val frm = Input(UInt(3.W))
    })
    def latency = 3
    private val iss = issueNode.in.head._1.issue
    private val wb = writebackNode.out.head._1
    iss.ready := true.B

    private val vmac = Module(new VMac)
    private val vred = Module(new Reduction)
    private val uopShiftQueue = Module(new MicroOpShiftQueue(latency))

    private val vuop = uopToVuop(iss.bits.uop, iss.valid, io.vstart, io.vcsr(2,1), io.frm, p)
    private val src0 = iss.bits.src(0)
    private val src1 = iss.bits.src(1)
    private val src2 = iss.bits.src(2)
    private val mask = iss.bits.vm

    private val uopIdx = uopShiftQueue.io.out.bits.uopIdx
    private val uopNum = uopShiftQueue.io.out.bits.uopNum
    private val uopOut = uopShiftQueue.io.out.bits
    private val isNarrow = uopOut.vctrl.isNarrow && !uopOut.vctrl.maskOp
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


    uopShiftQueue.io.in.valid := iss.valid && cfg.fuConfigs.map(_.fuType === iss.bits.uop.ctrl.fuType).reduce(_|_) && !iss.bits.uop.robIdx.needFlush(redirectIn)
    uopShiftQueue.io.in.bits := iss.bits.uop
    uopShiftQueue.io.redirect := redirectIn

    vmac.io.in.valid := iss.valid && iss.bits.uop.ctrl.fuType === FuConfigs.vmacCfg.fuType && !iss.bits.uop.robIdx.needFlush(redirectIn)
    vmac.io.in.bits.uop := vuop
    vmac.io.in.bits.vs1 := src0
    vmac.io.in.bits.vs2 := src1
    vmac.io.in.bits.rs1 := src0(XLEN - 1, 0)
    vmac.io.in.bits.oldVd := src2
    vmac.io.in.bits.mask := mask

    vred.io.in.valid := iss.valid && iss.bits.uop.ctrl.fuType === FuConfigs.vredCfg.fuType && !iss.bits.uop.robIdx.needFlush(redirectIn)
    vred.io.in.bits.uop := vuop
    vred.io.in.bits.vs1 := src0
    vred.io.in.bits.vs2 := src1
    vred.io.in.bits.rs1 := src0(XLEN - 1, 0)
    vred.io.in.bits.oldVd := src2
    vred.io.in.bits.mask := mask
    private val isVred = uopOut.ctrl.fuType === FuConfigs.vredCfg.fuType

    private val validSeq = Seq(vmac.io.out.valid, vred.io.out.valid)
    private val dataSeq = Seq(vmac.io.out.bits, vred.io.out.bits)
    private val wbData = Mux1H(validSeq, dataSeq)

    wb.valid := uopShiftQueue.io.out.valid && validSeq.reduce(_||_)
    wb.bits := DontCare
    wb.bits.uop := uopShiftQueue.io.out.bits
    wb.bits.data := wbData.vd
    wb.bits.vxsat := wbData.vxsat
    wb.bits.wakeupMask := Mux(isVred, Mux(uopNum === 1.U, fullMask, finalMask), ((1 << (VLEN / 8)) - 1).U((VLEN / 8).W))
    wb.bits.writeDataMask := Mux(isVred, Mux(uopNum === 1.U, fullMask, finalMask), ((1 << (VLEN / 8)) - 1).U((VLEN / 8).W))
  }
}
