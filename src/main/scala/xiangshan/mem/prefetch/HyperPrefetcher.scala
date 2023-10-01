package xiangshan.mem.prefetch

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tilelink._
import xiangshan.{XSBundle, XSModule}
import xiangshan.cache.HasDCacheParameters
import xs.utils.perf.HasPerfLogging

case class HyperPrefetchParams(
  fTableEntries: Int = 32,
  pTableQueueEntries: Int = 2,
  fTableQueueEntries: Int = 256
) extends PrefetcherParams {
   val hasPrefetchBit:  Boolean = true
   val inflightEntries: Int = 32
}

trait HasHyperPrefetcherParams extends HasDCacheParameters {
  val hyperPrefetchParams = HyperPrefetchParams()

  val fullAddressBits = PAddrBits
  val pageOffsetBits = log2Up(4096)
  val offsetBits = log2Up(dcacheParameters.blockBytes)

  val pageAddrBits = fullAddressBits - pageOffsetBits
  val blkOffsetBits = pageOffsetBits - offsetBits

  val fTableEntries = hyperPrefetchParams.fTableEntries
  val fTagBits = pageAddrBits - log2Up(fTableEntries)
  val pTableQueueEntries = hyperPrefetchParams.pTableQueueEntries
  val fTableQueueEntries = hyperPrefetchParams.fTableQueueEntries
}

abstract class PrefetchBranchV2Module(implicit val p: Parameters) extends Module with HasHyperPrefetcherParams
abstract class PrefetchBranchV2Bundle(implicit val p: Parameters) extends Bundle with HasHyperPrefetcherParams

class FilterV2(implicit p: Parameters) extends PrefetchBranchV2Module {
  val io = IO(new Bundle() {
    val req = Flipped(DecoupledIO(new PrefetchReq))
    val resp = DecoupledIO(new PrefetchReq)
    val spp2llc = Input(Bool())
    val hint2llc = ValidIO(new PrefetchReq)
  })

  def idx(addr:      UInt) = addr(log2Up(fTableEntries) - 1, 0)
  def tag(addr:      UInt) = addr(pageAddrBits - 1, log2Up(fTableEntries))

  def fTableEntry() = new Bundle {
    val valid = Bool()
    val tag = UInt(fTagBits.W)
    val bitMap = Vec(64, Bool())
  }
  val dupNums = 4

  val req_dups = RegInit(VecInit(Seq.fill(dupNums)(0.U.asTypeOf(new PrefetchReq))))
  val req_dups_valid = RegInit(VecInit(Seq.fill(dupNums)(false.B)))
  val req_hint2llc = RegNext(io.spp2llc,false.B)
  req_dups.foreach(_ := io.req.bits)
  req_dups_valid.foreach( _ := io.req.valid)
  val dupOffsetBits = log2Up(fTableEntries/dupNums)
  val dupBits = log2Up(dupNums)
  val fTable = RegInit(VecInit(Seq.fill(fTableEntries)(0.U.asTypeOf(fTableEntry()))))
  val q = Module(new Queue(UInt(fullAddressBits.W), fTableQueueEntries, flow = false, pipe = true))

  val hit = WireInit(VecInit.fill(dupNums)(false.B))
  val readResult = WireInit(VecInit.fill(dupNums)(0.U.asTypeOf(fTableEntry())))
  val hitForMap = WireInit(VecInit.fill(dupNums)(false.B))

  for(i <- 0 until(dupNums)) {
    when(req_dups(i).addr(dupOffsetBits-1+dupBits,dupOffsetBits-1) === i.U(dupBits.W)) {
      val oldAddr = req_dups(i).addr
      val pageAddr = oldAddr(fullAddressBits - 1, pageOffsetBits)
      val blkOffset = oldAddr(pageOffsetBits - 1, offsetBits)

      //read fTable
      readResult(i) := fTable(idx(pageAddr))
      hit(i) := readResult(i).valid
      hitForMap(i) := hit(i) && readResult(i).bitMap(blkOffset)

      val wData = WireInit(0.U.asTypeOf(fTableEntry()))
      val newBitMap = readResult(i).bitMap.zipWithIndex.map { case (b, i) => Mux(i.asUInt === blkOffset, true.B, false.B) }

      wData.valid := true.B
      wData.tag := tag(pageAddr)
      wData.bitMap := newBitMap
      when(req_dups_valid(i)) {
        when(hit(i)) {
          fTable(idx(pageAddr)).bitMap(blkOffset) := true.B
        }.otherwise {
          fTable(idx(pageAddr)) := wData
        }
      }
    }
  }
  io.resp.valid := req_dups_valid(0) && (!hitForMap.asUInt.orR)
  io.resp.bits.addr := req_dups(0).addr

  io.hint2llc.valid := req_dups_valid(1) && req_hint2llc
  io.hint2llc.bits.addr := req_dups(1).addr

  q.io.enq.valid := req_dups_valid(2) && !hitForMap.asUInt.orR && !req_hint2llc // if spp2llc , don't enq
  q.io.enq.bits := req_dups(2).addr

  val isqFull = q.io.count === fTableQueueEntries.U
  q.io.deq.ready := isqFull;dontTouch(q.io.deq.ready)

  val evictAddr = q.io.deq.bits
  val evictPageAddr = evictAddr(fullAddressBits - 1, pageOffsetBits)
  val evictBlkOffset = evictAddr(pageOffsetBits - 1, offsetBits)
  val evictBlkAddr = evictAddr(fullAddressBits - 1, offsetBits)
  val readEvict = WireInit(VecInit.fill(dupNums)(0.U.asTypeOf(fTableEntry())))
  val hitEvict =  WireInit(VecInit.fill(dupNums)(false.B))
  for(i <- 0 until(dupNums)) {
    when(req_dups(i).addr(dupOffsetBits-1+dupBits,dupOffsetBits-1) === i.U(dupBits.W)) {
      val oldAddr = req_dups(i).addr
      val blkAddr = oldAddr(fullAddressBits - 1, offsetBits)
      val conflict = req_dups_valid.reduce(_ || _) && blkAddr === evictBlkAddr
      readEvict(i) := fTable(idx(evictPageAddr))
      hitEvict(i) := q.io.deq.fire && readEvict(i).valid && tag(evictPageAddr) === readEvict(i).tag && readEvict(i).bitMap(evictBlkOffset) && !conflict
      when(hitEvict(i)) {
        fTable(idx(evictPageAddr)).bitMap(evictBlkOffset) := false.B
      }
    }
  }

  /*
  val evictAddr = io.evict.bits.addr
  val evictPageAddr = evictAddr(fullAddressBits - 1, pageOffsetBits)
  val evictBlkOffset = evictAddr(pageOffsetBits - 1, offsetBits)
  val evictBlkAddr = evictAddr(fullAddressBits - 1, offsetBits)
  val readEvict = Wire(fTableEntry())
  val hitEvict = Wire(Bool())
  val conflict = io.req.fire && blkAddr === evictBlkAddr
  readEvict := fTable(idx(evictPageAddr))
  hitEvict := io.evict.valid && readEvict.valid && tag(evictPageAddr) === readEvict.tag && readEvict.bitMap(evictBlkOffset) && !conflict
  when(hitEvict) {
    fTable(idx(evictPageAddr)).bitMap(evictBlkOffset) := false.B
  }*/

  io.req.ready := true.B
}

class PrefetchTrain(implicit p:Parameters) extends XSBundle {
  val addr = UInt(PAddrBits.W)
}

class PrefetchReq(implicit p:Parameters) extends XSBundle {
  val addr = UInt(PAddrBits.W)
}

class PrefetchQueue(inflightEntries:Int = 16)(implicit p: Parameters) extends XSModule with HasPerfLogging{
  val io = IO(new Bundle {
    val enq = Flipped(DecoupledIO(new PrefetchReq))
    val deq = DecoupledIO(new PrefetchReq)
    val used = Output(UInt(6.W))
  })
  /*  Here we implement a queue that
   *  1. is pipelined  2. flows
   *  3. always has the latest reqs, which means the queue is always ready for enq and deserting the eldest ones
   */
  val queue = RegInit(VecInit(Seq.fill(inflightEntries)(0.U.asTypeOf(new PrefetchReq))))
  val valids = RegInit(VecInit(Seq.fill(inflightEntries)(false.B)))
  val idxWidth = log2Up(inflightEntries)
  val head = RegInit(0.U(idxWidth.W))
  val tail = RegInit(0.U(idxWidth.W))
  val empty = head === tail && !valids.last
  val full = head === tail && valids.last

  when(!empty && io.deq.ready) {
    valids(head) := false.B
    head := head + 1.U
  }

  when(io.enq.valid) {
    val exist = queue.zipWithIndex.map{case (x, i) => Mux(valids(i) && x.addr === io.enq.bits.addr, true.B, false.B)}.reduce(_ || _)
    when(!exist) {
      queue(tail) := io.enq.bits
      valids(tail) := !empty || !io.deq.ready // true.B
      tail := tail + (!empty || !io.deq.ready).asUInt
      when(full && !io.deq.ready) {
        head := head + 1.U
      }
    }
  }

  io.enq.ready := true.B
  io.deq.valid := !empty || io.enq.valid
  io.deq.bits := Mux(empty, io.enq.bits, queue(head))

  io.used := PopCount(valids.asUInt)

  // The reqs that are discarded = enq - deq
  XSPerfAccumulate("prefetch_queue_enq", io.enq.fire)
  XSPerfAccumulate("prefetch_queue_deq", io.deq.fire)
  XSPerfHistogram("prefetch_queue_entry", PopCount(valids.asUInt),
    true.B, 0, inflightEntries, 1)
}

//Only used for hybrid spp and bop
class HyperPrefetcher(parentName:String="")(implicit p: Parameters) extends BasePrefecher
  with HasSMSModuleHelper
  with HasPerfLogging{
//  val io = IO(new Bundle() {
//    val train = Flipped(DecoupledIO(new PrefetchTrain))
//    val req = DecoupledIO(new PrefetchReq)
////    val resp = Flipped(DecoupledIO(new PrefetchResp))
////    val evict = Flipped(DecoupledIO(new PrefetchEvict))
//    val hint2llc = ValidIO(new PrefetchReq)
//    val db_degree = Flipped(ValidIO(UInt(2.W)))
//    val queue_used = Input(UInt(6.W))
//  })
  val sms_ctrl = IO(new Bundle() {
    val io_agt_en = Input(Bool())
    val io_stride_en = Input(Bool())
    val io_pht_en = Input(Bool())
    val io_act_threshold = Input(UInt(REGION_OFFSET.W))
    val io_act_stride = Input(UInt(6.W))
  })
  // sms
  val sms = Module(new SMSPrefetcher(parentName =parentName + "sms_"))
  sms.io_agt_en := sms_ctrl.io_agt_en
  sms.io_stride_en := sms_ctrl.io_stride_en
  sms.io_pht_en := sms_ctrl.io_pht_en
  sms.io_act_threshold := sms_ctrl.io_act_threshold
  sms.io_act_stride := sms_ctrl.io_act_stride
  sms.io.enable := io.enable
  sms.io.ld_in <> io.ld_in
  io.tlb_req <> sms.io.tlb_req

  val q_sms = Module(new PrefetchQueue(8))
  q_sms.io.enq.valid := sms.io.pf_addr.valid
  q_sms.io.enq.bits.addr := sms.io.pf_addr.bits

  val fTable = Module(new FilterV2)

  //spp
  val spp = Module(new SppDev2Prefetch())
//  val bop = Module(new BestOffsetPrefetch())

  val q_spp = Module(new PrefetchQueue(32))

  q_spp.io.enq <> spp.io.req
  q_spp.io.deq.ready := true.B

  spp.io.train.valid := io.ld_in(0).valid
  spp.io.train.bits.addr := io.ld_in(0).bits.paddr

//  val train_for_bop = RegInit(0.U.asTypeOf(new PrefetchTrain))
//  val train_for_bop_valid = RegInit(false.B)

//  spp.io.resp.bits := 0.U
//  spp.io.resp.valid := false.B

  spp.io.req.ready := true.B

  q_spp.io.deq.ready := fTable.io.req.ready
  fTable.io.req.valid := q_spp.io.deq.valid
  fTable.io.req.bits.addr := q_spp.io.deq.bits.addr


  fTable.io.resp.ready := false.B
  fTable.io.spp2llc := false.B
  fTable.io.hint2llc := DontCare

  val pftArb = Module(new Arbiter(new PrefetchReq(), 2))
  pftArb.io.in(0) <> q_sms.io.deq
  pftArb.io.in(1) <> q_spp.io.deq

  io.pf_addr.valid := pftArb.io.out.valid
  io.pf_addr.bits := pftArb.io.out.bits.addr
  pftArb.io.out.ready := true.B

//  io.req <> fTable.io.resp
//  io.hint2llc := fTable.io.hint2llc;dontTouch(io.hint2llc)
//  fTable.io.evict.valid := io.evict.valid
//  fTable.io.evict.bits := io.evict.bits
//  io.evict.ready := fTable.io.evict.ready

//  io.train.ready := true.B
//  spp.io.db_degree.valid := io.db_degree.valid
//  spp.io.db_degree.bits := io.db_degree.bits
//  spp.io.queue_used := io.queue_used
  XSPerfAccumulate("sms_send2_queue", q_sms.io.deq.fire)
  XSPerfAccumulate("spp_send2_queue", fTable.io.resp.fire && q_spp.io.deq.fire)
//  XSPerfAccumulate("prefetcher_has_evict", io.evict.fire)
}