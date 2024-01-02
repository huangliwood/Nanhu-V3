package xiangshan.vector.vbackend.vexecute.vexu
import org.chipsalliance.cde.config.Parameters
import chisel3._
import chisel3.util._
import darecreek.exu.vfu.VFpuOutput
import darecreek.exu.vfu.fp.VFPUWrapper
import xiangshan.HasXSParameter
import xiangshan.backend.execute.exu.{BasicExu, BasicExuImpl, ExuConfig, ExuInputNode, ExuOutputNode, ExuType}
import xiangshan.backend.execute.fu.FuConfigs
import xiangshan.vector.{EewType, HasVectorParameters}
import xiangshan.vector.vbackend.vexecute.vfu.uopToVuop
class VFpExu(id:Int, complexName:String)(implicit p: Parameters) extends BasicExu{
  private val cfg = ExuConfig(
    name = "VFpExu",
    id = id,
    complexName = complexName,
    fuConfigs = Seq(FuConfigs.vfpCfg),
    exuType = ExuType.vfp,
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

    private val iss = issueNode.in.head._1.issue
    private val wb = writebackNode.out.head._1

    private val vfp = Module(new VFPUWrapper)

    private val vuop = uopToVuop(iss.bits.uop, iss.valid, io.vstart, io.vcsr(2,1), io.frm, p)
    private val src0 = iss.bits.src(0)
    private val src1 = iss.bits.src(1)
    private val src2 = iss.bits.src(2)
    private val mask = iss.bits.vm

    iss.ready := vfp.io.in.ready
    vfp.io.in.valid := iss.valid && iss.bits.uop.ctrl.fuType === cfg.fuConfigs.head.fuType && !iss.bits.uop.robIdx.needFlush(redirectIn)
    vfp.io.in.bits.uop := vuop
    vfp.io.in.bits.vs1 := src0
    vfp.io.in.bits.vs2 := src1
    vfp.io.in.bits.rs1 := src0(XLEN - 1, 0)
    vfp.io.in.bits.oldVd := src2
    vfp.io.in.bits.mask := mask
    vfp.io.out.ready := true.B
    vfp.io.redirect := redirectIn

    private val outDelay = Wire(Valid(new VFpuOutput))
    outDelay.valid := RegNext(vfp.io.out.valid & !vfp.io.out.bits.uop.sysUop.robIdx.needFlush(redirectIn), false.B)
    outDelay.bits := RegEnable(vfp.io.out.bits, vfp.io.out.valid)
    
    wb.valid := outDelay.valid
    wb.bits := DontCare
    wb.bits.uop := outDelay.bits.uop.sysUop
    wb.bits.data := outDelay.bits.vd
    wb.bits.fflags := outDelay.bits.fflags

    private val uopOut = outDelay.bits.uop.sysUop
    private val uopIdx = uopOut.uopIdx
    private val uopNum = uopOut.uopNum
    private val isNarrow = uopOut.vctrl.isNarrow && !uopOut.vctrl.maskOp
    private val lowHalf = !uopIdx(0)
    private val highHalf = uopIdx(0)
    private val maskLen = VLEN / 8
    private val halfMaskLen = maskLen / 2

    private def ones(in: Int): UInt = ((1 << in) - 1).U(in.W)

    private val lowHalfMask = Cat(0.U(halfMaskLen.W), ones(halfMaskLen))
    private val highHalfMask = Cat(ones(halfMaskLen), 0.U(halfMaskLen.W))
    private val fullMask = ones(maskLen)
    private val finalMask = MuxCase(fullMask, Seq(
      (isNarrow && lowHalf) -> lowHalfMask,
      (isNarrow && highHalf) -> highHalfMask,
    ))
    wb.bits.wakeupMask := Mux(uopNum === 1.U, fullMask, finalMask)
    wb.bits.writeDataMask := Mux(uopNum === 1.U, fullMask, finalMask)
  }
}
