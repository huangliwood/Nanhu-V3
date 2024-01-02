package xiangshan

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import xiangshan.vector.videcode.VLDecode

package object vector {
  object EewType {
    def dc:UInt = 0.U
    def const:UInt = 1.U
    def scalar:UInt = 2.U
    def sew:UInt = 3.U
    def sewm2:UInt = 4.U
    def sewd2:UInt = 5.U
    def sewd4:UInt = 6.U
    def sewd8:UInt = 7.U

    def apply():UInt = UInt(3.W)
  }

  object EewVal {
    def dc: BitPat = BitPat("b111")
    def byte:UInt = "b000".U
    def hword:UInt = "b001".U
    def word:UInt = "b010".U
    def dword:UInt = "b011".U

    def apply():UInt = UInt(3.W)
  }

  object EmulVal {
    def dc: BitPat = BitPat("b111")
    def r1 :BitPat = BitPat("b000")
    def r2: BitPat = BitPat("b001")
    def r4: BitPat = BitPat("b010")
    def r8: BitPat = BitPat("b011")
    def apply():UInt = UInt(3.W)
  }

  object EmulType {
    def const:UInt = "b0".U
    def lmul:UInt = "b1".U

    def apply():UInt = UInt(1.W)
  }

  class VCtrlSignals(implicit p: Parameters) extends XSBundle {
    val eew = Vec(3, EewVal()) //VS1, VS2, VS3/VD EEW
    val eewType = Vec(3, EewType())
    val emul = EmulVal() //Destination EMUL
    val emulType = EmulType()
    val ordered = Bool()
    val isLs = Bool()
    val disable = Bool()
    val ff = Bool()
    val maskOp = Bool()
    val isWidden = Bool()
    val isNarrow = Bool()
    val nf = UInt(4.W)
    val vm = Bool()
    val funct6 = UInt(6.W)
    val funct3 = UInt(3.W)

    private def allSignals: IndexedSeq[UInt] = eew ++ eewType ++ Seq(emul, emulType, ordered, isLs,
      ff, maskOp, isWidden, isNarrow)
    def decode(inst: UInt, table: Iterable[(BitPat, List[BitPat])]): VCtrlSignals = {
      this := DontCare
      val decoder = xiangshan.backend.decode.DecodeLogic(inst, VLDecode.decodeDefault, table, QMC = false)
      allSignals zip decoder foreach { case (s, d) => s := d }
      this
    }
  }
}
