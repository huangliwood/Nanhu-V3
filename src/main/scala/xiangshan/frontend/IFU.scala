package xiangshan.frontend

import chisel3._
import chisel3.util._
import device.RAMHelper
import xiangshan._
import utils._

trait HasIFUConst { this: XSModule =>
  val resetVector = 0x80000000L//TODO: set reset vec
  val groupAlign = log2Up(FetchWidth * 4 * 2)
  def groupPC(pc: UInt): UInt = Cat(pc(VAddrBits-1, groupAlign), 0.U(groupAlign.W))
  // each 1 bit in mask stands for 2 Bytes
  def mask(pc: UInt): UInt = (Fill(PredictWidth * 2, 1.U(1.W)) >> pc(groupAlign - 1, 1))(PredictWidth - 1, 0)
  def snpc(pc: UInt): UInt = pc + (PopCount(mask(pc)) << 1)
}

class IFUIO extends XSBundle
{
  val fetchPacket = DecoupledIO(new FetchPacket)
  val redirect = Flipped(ValidIO(new Redirect))
  val outOfOrderBrInfo = Flipped(ValidIO(new BranchUpdateInfo))
  val inOrderBrInfo = Flipped(ValidIO(new BranchUpdateInfo))
  val icacheReq = DecoupledIO(new FakeIcacheReq)
  val icacheResp = Flipped(DecoupledIO(new FakeIcacheResp))
  val icacheFlush = Output(UInt(2.W))
}


class IFU extends XSModule with HasIFUConst
{
  val io = IO(new IFUIO)
  val bpu = if (EnableBPU) Module(new BPU) else Module(new FakeBPU)
  val pd = Module(new PreDecode)

  val if2_redirect, if3_redirect, if4_redirect = WireInit(false.B)
  val if1_flush, if2_flush, if3_flush, if4_flush = WireInit(false.B)

  if4_flush := io.redirect.valid
  if3_flush := if4_flush || if4_redirect
  if2_flush := if3_flush || if3_redirect
  if1_flush := if2_flush || if2_redirect

  //********************** IF1 ****************************//
  val if1_valid = !reset.asBool
  val if1_npc = WireInit(0.U(VAddrBits.W))
  val if2_ready = WireInit(false.B)
  val if1_fire = if1_valid && (if2_ready || if1_flush) && io.icacheReq.ready

  // val extHist = VecInit(Fill(ExtHistoryLength, RegInit(0.U(1.W))))
  val extHist = RegInit(VecInit(Seq.fill(ExtHistoryLength)(0.U(1.W))))
  val headPtr = RegInit(0.U(log2Up(ExtHistoryLength).W))
  val shiftPtr = WireInit(false.B)
  val newPtr = Wire(UInt(log2Up(ExtHistoryLength).W))
  val ptr = Mux(shiftPtr, newPtr, headPtr)
  when (shiftPtr) { headPtr := newPtr }
  val hist = Wire(Vec(HistoryLength, UInt(1.W)))
  for (i <- 0 until HistoryLength) {
    hist(i) := extHist(ptr + i.U)
  }

  newPtr := headPtr
  shiftPtr := false.B

  //********************** IF2 ****************************//
  val if2_valid = RegEnable(next = if1_valid, init = false.B, enable = if1_fire)
  val if3_ready = WireInit(false.B)
  val if2_fire = if2_valid && if3_ready
  val if2_pc = RegEnable(next = if1_npc, init = resetVector.U, enable = if1_fire)
  val if2_snpc = snpc(if2_pc)
  val if2_histPtr = RegEnable(ptr, if1_fire)
  if2_ready := if2_fire || !if2_valid
  when (if2_flush) { if2_valid := if1_fire }

  when (RegNext(reset.asBool) && !reset.asBool) {
    if1_npc := resetVector.U(VAddrBits.W)
  }.elsewhen (if2_fire) {
    if1_npc := if2_snpc
  }.otherwise {
    if1_npc := RegNext(if1_npc)
  }

  val if2_bp = bpu.io.out(0).bits
  if2_redirect := if2_fire && bpu.io.out(0).valid && if2_bp.redirect && !if2_bp.saveHalfRVI
  when (if2_redirect) { 
    if1_npc := if2_bp.target
  }

  when (if2_fire && (if2_bp.taken || if2_bp.hasNotTakenBrs)) {
    shiftPtr := true.B
    newPtr := headPtr - 1.U
    hist(0) := if2_bp.taken.asUInt
    extHist(newPtr) := if2_bp.taken.asUInt
  }

  //********************** IF3 ****************************//
  val if3_valid = RegEnable(next = if2_valid, init = false.B, enable = if2_fire)
  val if4_ready = WireInit(false.B)
  val if3_fire = if3_valid && if4_ready && io.icacheResp.valid
  val if3_pc = RegEnable(if2_pc, if2_fire)
  val if3_histPtr = RegEnable(if2_histPtr, if2_fire)
  if3_ready := if3_fire || !if3_valid
  when (if3_flush) { if3_valid := false.B }

  val if3_bp = bpu.io.out(1).bits
  val prev_half_valid = RegInit(false.B)
  val prev_half_redirect = RegInit(false.B)
  val prev_half_fetchpc = Reg(UInt(VAddrBits.W))
  val prev_half_idx = Reg(UInt(log2Up(PredictWidth).W))
  val prev_half_tgt = Reg(UInt(VAddrBits.W))
  val prev_half_taken = RegInit(false.B)
  val prev_half_instr = Reg(UInt(16.W))
  when (if3_flush) {
    prev_half_valid := false.B
    prev_half_redirect := false.B
  }.elsewhen (if3_fire && if3_bp.saveHalfRVI) {
    prev_half_valid := true.B
    prev_half_redirect := if3_bp.redirect && bpu.io.out(1).valid
    prev_half_fetchpc := if3_pc
    val idx = Mux(if3_bp.redirect && bpu.io.out(1).valid, if3_bp.jmpIdx, PopCount(mask(if3_pc)) - 1.U)
    prev_half_idx := idx
    prev_half_tgt := if3_bp.target
    prev_half_taken := if3_bp.taken
    prev_half_instr := pd.io.out.instrs(idx)(15, 0)
  }.elsewhen (if3_fire) {
    prev_half_valid := false.B
    prev_half_redirect := false.B
  }

  // if3_redirect := if3_fire && (prev_half_valid && prev_half_taken || bpu.io.out(1).valid && if3_bp.redirect && !if3_bp.saveHalfRVI)
  // when (if3_redirect) {
  //   if1_npc := Mux(prev_half_valid && prev_half_redirect, prev_half_tgt, if3_bp.target)
  // }

  when (bpu.io.out(1).valid && if3_fire) {
    when (prev_half_valid && prev_half_taken) {
      if3_redirect := true.B
      if1_npc := prev_half_tgt
      shiftPtr := true.B
      newPtr := if3_histPtr - 1.U
      hist(0) := 1.U
      extHist(newPtr) := 1.U
    }.elsewhen (if3_bp.redirect && !if3_bp.saveHalfRVI) {
      if3_redirect := true.B
      if1_npc := if3_bp.target
      shiftPtr := true.B
      newPtr := Mux(if3_bp.taken || if3_bp.hasNotTakenBrs, if3_histPtr - 1.U, if3_histPtr)
      hist(0) := Mux(if3_bp.taken || if3_bp.hasNotTakenBrs, if3_bp.taken.asUInt, extHist(if3_histPtr))
      extHist(newPtr) := Mux(if3_bp.taken || if3_bp.hasNotTakenBrs, if3_bp.taken.asUInt, extHist(if3_histPtr))
    }.elsewhen (if3_bp.saveHalfRVI) {
      if3_redirect := true.B
      if1_npc := snpc(if3_pc)
      shiftPtr := true.B
      newPtr := Mux(if3_bp.hasNotTakenBrs, if3_histPtr - 1.U, if3_histPtr)
      hist(0) := Mux(if3_bp.hasNotTakenBrs, 0.U, extHist(if3_histPtr))
      extHist(newPtr) := Mux(if3_bp.hasNotTakenBrs, 0.U, extHist(if3_histPtr))
    }.otherwise {
      if3_redirect := false.B
    }
  }.otherwise {
    if3_redirect := false.B
  }


  //********************** IF4 ****************************//
  val if4_pd = RegEnable(pd.io.out, if3_fire)
  // val if4_icacheResp = RegEnable(io.icacheResp.bits, if3_fire)
  val if4_valid = RegEnable(next = if3_valid, init = false.B, enable = if3_fire)
  val if4_fire = if4_valid && io.fetchPacket.ready
  val if4_pc = RegEnable(if3_pc, if3_fire)
  val if4_histPtr = RegEnable(if3_histPtr, if3_fire)
  if4_ready := if4_fire || !if4_valid
  when (if4_flush) { if4_valid := false.B }

  val if4_bp = bpu.io.out(2).bits

  when (bpu.io.out(2).valid && if4_fire && if4_bp.redirect) {
    when (!if4_bp.saveHalfRVI) {
      if4_redirect := true.B
      if1_npc := if4_bp.target

      shiftPtr := true.B
      newPtr := Mux(if4_bp.taken || if4_bp.hasNotTakenBrs, if4_histPtr - 1.U, if4_histPtr)
      hist(0) := Mux(if4_bp.taken || if4_bp.hasNotTakenBrs, if4_bp.taken.asUInt, extHist(if4_histPtr))
      extHist(newPtr) := Mux(if4_bp.taken || if4_bp.hasNotTakenBrs, if4_bp.taken.asUInt, extHist(if4_histPtr))

    }.otherwise {
      if4_redirect := true.B
      if1_npc := snpc(if4_pc)

      prev_half_valid := true.B
      prev_half_redirect := true.B
      prev_half_fetchpc := if4_pc
      val idx = PopCount(mask(if4_pc)) - 1.U
      prev_half_idx := idx
      prev_half_tgt := if4_bp.target
      prev_half_taken := if4_bp.taken
      prev_half_instr := if4_pd.instrs(idx)(15, 0)

      shiftPtr := true.B
      newPtr := Mux(if4_bp.hasNotTakenBrs, if4_histPtr - 1.U, if4_histPtr)
      hist(0) := Mux(if4_bp.hasNotTakenBrs, 0.U, extHist(if4_histPtr))
      extHist(newPtr) := Mux(if4_bp.hasNotTakenBrs, 0.U, extHist(if4_histPtr))
    }
  }.otherwise {
    if4_redirect := false.B
  }

  when (io.outOfOrderBrInfo.valid) {
    shiftPtr := true.B
    newPtr := io.outOfOrderBrInfo.bits.brInfo.histPtr - 1.U
    hist(0) := io.outOfOrderBrInfo.bits.taken
    extHist(newPtr) := io.outOfOrderBrInfo.bits.taken
  }

  when (io.redirect.valid) {
    if1_npc := io.redirect.bits.target
  }

  io.icacheReq.valid := if1_valid && if2_ready
  io.icacheReq.bits.addr := if1_npc
  io.icacheResp.ready := if3_valid && if4_ready
  io.icacheFlush := Cat(if3_flush, if2_flush)

  val inOrderBrHist = Wire(Vec(HistoryLength, UInt(1.W)))
  (0 until HistoryLength).foreach(i => inOrderBrHist(i) := extHist(i.U + io.inOrderBrInfo.bits.brInfo.histPtr))
  bpu.io.inOrderBrInfo.valid := io.inOrderBrInfo.valid
  bpu.io.inOrderBrInfo.bits := Cat(inOrderBrHist.asUInt, io.inOrderBrInfo.bits.asUInt).asTypeOf(new BranchUpdateInfoWithHist)
  bpu.io.flush := Cat(if4_flush, if3_flush, if2_flush)
  bpu.io.in.valid := if1_fire
  bpu.io.in.bits.pc := if1_npc
  bpu.io.in.bits.hist := hist.asUInt
  bpu.io.in.bits.inMask := mask(if1_npc)
  bpu.io.out(0).ready := if2_fire
  bpu.io.out(1).ready := if3_fire
  bpu.io.out(2).ready := if4_fire
  bpu.io.predecode.valid := if4_valid
  bpu.io.predecode.bits.mask := if4_pd.mask
  bpu.io.predecode.bits.pd := if4_pd.pd
  bpu.io.branchInfo.ready := if4_fire

  pd.io.in := io.icacheResp.bits
  pd.io.prev.valid := prev_half_valid
  pd.io.prev.bits := prev_half_instr

  io.fetchPacket.valid := if4_valid && !io.redirect.valid
  io.fetchPacket.bits.instrs := if4_pd.instrs
  io.fetchPacket.bits.mask := if4_pd.mask & (Fill(PredictWidth, !if4_bp.taken) | (Fill(PredictWidth, 1.U(1.W)) >> (~if4_bp.jmpIdx)))
  io.fetchPacket.bits.pc := if4_pd.pc
  (0 until PredictWidth).foreach(i => io.fetchPacket.bits.pnpc(i) := if4_pd.pc(i) + Mux(if4_pd.pd(i).isRVC, 2.U, 4.U))
  when (if4_bp.taken) {
    io.fetchPacket.bits.pnpc(if4_bp.jmpIdx) := if4_bp.target
  }
  io.fetchPacket.bits.brInfo := bpu.io.branchInfo.bits
  (0 until PredictWidth).foreach(i => io.fetchPacket.bits.brInfo(i).histPtr := if4_histPtr)
  io.fetchPacket.bits.pd := if4_pd.pd

  // debug info
  XSDebug(RegNext(reset.asBool) && !reset.asBool, "Reseting...\n")
  XSDebug(io.icacheFlush(0).asBool, "Flush icache stage2...\n")
  XSDebug(io.icacheFlush(1).asBool, "Flush icache stage3...\n")
  XSDebug(io.redirect.valid, "Rediret from backend! isExcp=%d isMisPred=%d isReplay=%d pc=%x\n",
    io.redirect.bits.isException, io.redirect.bits.isMisPred, io.redirect.bits.isReplay, io.redirect.bits.pc)
  XSDebug(io.redirect.valid, p"Rediret from backend! target=${Hexadecimal(io.redirect.bits.target)} brTag=${io.redirect.bits.brTag}\n")

  XSDebug("[IF1] v=%d     fire=%d            flush=%d pc=%x ptr=%d mask=%b\n", if1_valid, if1_fire, if1_flush, if1_npc, ptr, mask(if1_npc))
  XSDebug("[IF2] v=%d r=%d fire=%d redirect=%d flush=%d pc=%x ptr=%d snpc=%x\n", if2_valid, if2_ready, if2_fire, if2_redirect, if2_flush, if2_pc, if2_histPtr, if2_snpc)
  XSDebug("[IF3] v=%d r=%d fire=%d redirect=%d flush=%d pc=%x ptr=%d\n", if3_valid, if3_ready, if3_fire, if3_redirect, if3_flush, if3_pc, if3_histPtr)
  XSDebug("[IF4] v=%d r=%d fire=%d redirect=%d flush=%d pc=%x ptr=%d\n", if4_valid, if4_ready, if4_fire, if4_redirect, if4_flush, if4_pc, if4_histPtr)

  XSDebug("[IF1][icacheReq] v=%d r=%d addr=%x\n", io.icacheReq.valid, io.icacheReq.ready, io.icacheReq.bits.addr)
  XSDebug("[IF1][ghr] headPtr=%d shiftPtr=%d newPtr=%d ptr=%d\n", headPtr, shiftPtr, newPtr, ptr)
  XSDebug("[IF1][ghr] hist=%b\n", hist.asUInt)
  XSDebug("[IF1][ghr] extHist=%b\n\n", extHist.asUInt)

  XSDebug("[IF2][bp] redirect=%d taken=%d jmpIdx=%d hasNTBrs=%d target=%x saveHalfRVI=%d\n\n", if2_bp.redirect, if2_bp.taken, if2_bp.jmpIdx, if2_bp.hasNotTakenBrs, if2_bp.target, if2_bp.saveHalfRVI)

  XSDebug("[IF3][icacheResp] v=%d r=%d pc=%x mask=%b\n", io.icacheResp.valid, io.icacheResp.ready, io.icacheResp.bits.pc, io.icacheResp.bits.mask)
  XSDebug("[IF3][bp] redirect=%d taken=%d jmpIdx=%d hasNTBrs=%d target=%x saveHalfRVI=%d\n", if3_bp.redirect, if3_bp.taken, if3_bp.jmpIdx, if3_bp.hasNotTakenBrs, if3_bp.target, if3_bp.saveHalfRVI)
  XSDebug("[IF3][prevHalfInstr] v=%d redirect=%d fetchpc=%x idx=%d tgt=%x taken=%d instr=%x\n\n",
    prev_half_valid, prev_half_redirect, prev_half_fetchpc, prev_half_idx, prev_half_tgt, prev_half_taken, prev_half_instr)

  XSDebug("[IF4][predecode] mask=%b\n", if4_pd.mask)
  XSDebug("[IF4][bp] redirect=%d taken=%d jmpIdx=%d hasNTBrs=%d target=%x saveHalfRVI=%d\n", if4_bp.redirect, if4_bp.taken, if4_bp.jmpIdx, if4_bp.hasNotTakenBrs, if4_bp.target, if4_bp.saveHalfRVI)
  XSDebug(io.fetchPacket.fire(), "[IF4][fetchPacket] v=%d r=%d mask=%b\n", io.fetchPacket.valid, io.fetchPacket.ready, io.fetchPacket.bits.mask)
  for (i <- 0 until PredictWidth) {
    XSDebug(io.fetchPacket.fire(), "[IF4][fetchPacket] %b %x pc=%x pnpc=%x pd: rvc=%d brType=%b call=%d ret=%d\n",
      io.fetchPacket.bits.mask(i),
      io.fetchPacket.bits.instrs(i),
      io.fetchPacket.bits.pc(i),
      io.fetchPacket.bits.pnpc(i),
      io.fetchPacket.bits.pd(i).isRVC,
      io.fetchPacket.bits.pd(i).brType,
      io.fetchPacket.bits.pd(i).isCall,
      io.fetchPacket.bits.pd(i).isRet
    )
  }
}