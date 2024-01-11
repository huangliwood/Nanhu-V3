package xiangshan.vector.vbackend.vexecute

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters

import darecreek.exu.vfu.VUop

import xiangshan.MicroOp
import xiangshan.vector.{EewType, EewVal}

package object vfu {
  def uopToVuop(src:MicroOp, valid:Bool, vstart:UInt, vxrm:UInt, frm:UInt, p:Parameters):VUop = {
    val res = Wire(new VUop()(p))
    res.ctrl.funct6 := src.vctrl.funct6
    res.ctrl.funct3 := src.vctrl.funct3
    res.ctrl.vm := !src.vctrl.vm
    res.ctrl.vs1_imm := src.ctrl.imm(4, 0)
    res.ctrl.widen := src.vctrl.isWidden && src.vctrl.eewType(1) === EewType.sew
    res.ctrl.widen2 := src.vctrl.isWidden && src.vctrl.eewType(1) === EewType.sewm2
    res.ctrl.narrow := src.vctrl.isNarrow && src.vctrl.eewType(2) === EewType.sew
    res.ctrl.narrow_to_1 := src.vctrl.isNarrow && src.vctrl.maskOp
    res.info.ma := src.vCsrInfo.vma(0)
    res.info.ta := src.vCsrInfo.vta(0)
    res.info.vsew := src.vCsrInfo.vsew
    res.info.vlmul := src.vCsrInfo.vlmul
    res.info.vl := src.vCsrInfo.vl
    res.info.vstart := vstart
    res.info.vxrm := vxrm
    res.info.frm := frm
    res.uopIdx := src.uopIdx(2,0)
    res.uopEnd := src.uopIdx(2,0) === (src.uopNum - 1.U)
    res.sysUop := src
    when(valid){assert(src.uopIdx <= 7.U)}
    when(valid){assert(src.uopNum <= 8.U)}
    res
  }
}
