package xiangshan.mem.prefetch
import org.chipsalliance.cde.config.Parameters
//import chipsalliance.rocketchip.config.Parameters
import chisel3.util._
import chisel3._
import chisel3.experimental.BundleLiterals._
import org.antlr.v4.runtime.CharStreams
import chisel3.util.experimental.{BoringUtils, loadMemoryFromFileInline}
import xs.utils.sram.SRAMTemplate
import org.chipsalliance.cde.config.Parameters
import chisel3._
import chisel3.util._
import xiangshan.cache.HasDCacheParameters
import xs.utils.{CircularShift,RegNextN,ReplacementPolicy,HasCircularQueuePtrHelper,CircularQueuePtr}
import xs.utils.{ParallelOperation,QPtrMatchMatrix,ZeroExt,OneHot}
import xs.utils.perf.HasPerfLogging

import java.io.File


case class SppDev2Prarameters() extends PrefetcherParams {
  val hasPrefetchBit: Boolean = true
  val inflightEntries: Int = 16
  val prefetchName="SppDev2 Prefetch"
}
trait hasSppDev2Parameters extends HasDCacheParameters{
  //l2 generic request bundle
  //    lazy val fullTagBits = fullAddressBits - setBits - offsetBits
  //    39-11-6=22
  lazy val prefetchQueue_size = 16

  val fullAddressBits = PAddrBits
  val pageOffsetBits = log2Up(4096)
  val offsetBits = log2Up(64)
  val setBits = log2Up(128)
  val fullTagBits = fullAddressBits - setBits - offsetBits

  val pageAddrBits = fullAddressBits - pageOffsetBits // 36 - 12 = 24
  val pageTagBits = pageAddrBits
  val blkOffsetBits = offsetBits
  val sigBits = 12
  val sigElemBits = 4
  val blkAddrBits = fullAddressBits - blkOffsetBits //36-6 = 30
  //st
  class sTableParm{
    val entry_nums: Int= 1024
    val tag_width: Int= 11
    val offset_width: Int= 9
    val delta_width: Int= 7
    val data_width = tag_width+sigBits+delta_width+1
    val nways = entry_nums
    val entryBits = log2Up(entry_nums)
  }
  val stParam = new sTableParm
  //pt
  class pTableParm{
    val entry_nums:Int=4096
    val entryBits = log2Up(entry_nums)
    val way:Int=4
    val c_sig_width: Int=4
    val c_delta_width: Int=4
    val delta_width: Int=7
    val u_data_width = c_sig_width+c_delta_width*way
    val s_data_width = delta_width*way
    val data_width = u_data_width+s_data_width
    val c_delta_initValue: Int = 1
    val confidence_width:Int = 8
    val max_lookup_depth = 8
    val enq_num = way
    val deq_num = 1
    val pfThreshold = 50
  }
  val ptParam = new pTableParm
  //prefetchFilter
  val pf_tag_width: Int= 6
  //globalHistoryReg
  val confidence_width: Int= 8
  val ghr_delta_width: Int= 7
  //accCounter
  val acc_width: Int= 10
  //   def acc_totalMax
  class ghrParam{
    val entry_nums:Int=8
    val entry_bits:Int=log2Up(entry_nums)
    val confidence_width: Int=8
    val last_offset_width: Int=6
    val delta_width: Int=7
  }
  class filterParam{
    val filter_size:Int = 16
    val entry_nums:Int = filter_size
    val tagBits: Int = fullAddressBits - pageOffsetBits
    val region_size = 8192
    val region_addrBits:Int = log2Up(region_size)
    val region_offsetBits:Int = region_addrBits-offsetBits
    val filterBits:Int = 128

  }
  val ghrParam=new ghrParam
  val pffParam = new filterParam

  //extensive function
  val hasBpOpt = Some(true)
}

abstract class SppDev2Bundle(implicit val p: Parameters) extends Bundle with hasSppDev2Parameters
abstract class SppDev2Module(implicit val p: Parameters) extends Module
  with HasPerfLogging
  with hasSppDev2Parameters
  with HasCircularQueuePtrHelper{
  def makeSign(old_sig: UInt, delta: SInt):UInt=((old_sig << sigElemBits) ^ (delta.asUInt))

  def get_blockAddr(x: UInt): UInt = {
    x(x.getWidth - 1, blkOffsetBits)
  }

  def get_blockOff(x: UInt): UInt = {
    x(blkOffsetBits - 1, 0)
  }
  def get_pageAddr(x: UInt): UInt = {
    x(x.getWidth - 1, pageOffsetBits)
  }

  def parseFullAddress(x: UInt): (UInt, UInt, UInt) = {
    val offset = x // TODO: check address mapping
    val set = offset >> offsetBits
    val tag = set >> setBits
    (tag(fullTagBits - 1, 0), set(setBits - 1, 0), offset(offsetBits - 1, 0))
  }

  def getPPN(x: UInt): UInt = {
    x(x.getWidth - 1, pageOffsetBits)
  }

  def generatePipeControl(lastFire: Bool, thisFire: Bool, thisFlush: Bool, lastFlush: Bool): Bool = {
    val valid = RegInit(false.B)
    when(thisFlush) {
      valid := false.B
    }
      .elsewhen(lastFire && !lastFlush) {
        valid := true.B
      }
      .elsewhen(thisFire) {
        valid := false.B
      }
    valid
  }
  class sppCounter[T<:Data](gen:T,init:Option[T],need_overflow:Boolean,name:String="Counter"){
    val initCounter=if(init.nonEmpty) RegInit(init.get).suggestName(name+"_Counter") else null
    val Counter=Reg(gen).suggestName(name+"_Counter")
    var value=if(init.nonEmpty) initCounter.asInstanceOf[T] else Counter.asInstanceOf[T]
    val saturated=RegInit(false.B)
    def apply:T={
      if(init.nonEmpty){
        println("Use received init value ")
        initCounter.asInstanceOf[T]
      }else{
        println("Use specialezd init value")
        Counter
      }
    }
    def reset(x:T)={
      value := x
      saturated := false.B
    }
    def add(x:UInt):Bool={
      val tmp=value.asUInt + x
      saturated:=Mux(tmp === ((1<<gen.getWidth)-1).U,true.B,false.B)
      when(saturated){
        if (need_overflow) {
          value := tmp
        }
      }.otherwise{
        value := tmp
      }
      Mux(saturated,true.B,false.B)
    }
  }
  object sppCounter{
    def apply[T<:UInt](gen:T,init:Option[T]=None,need_overflow:Boolean = false,name:String):sppCounter[T]={
      new sppCounter[T](gen,init,need_overflow,name)
    }

  }
  class ramCounter[T<:Data](gen:T, need_shift:Bool=false.B, name:String="Counter"){
    val value=WireInit(gen).suggestName(name+"_ram_Counter")
    private val value_next=WireInit(gen)
    val saturated=WireInit(false.B)
    def get_shiftValue:UInt={
      Mux(need_shift,(value.asUInt >> 1.U).asUInt,value.asUInt)
    }
    def add(x:Int=1):UInt={
      saturated:=Mux(value.asUInt===((1<<gen.getWidth)-1).U,true.B,false.B)//(Fill(gen.asUInt.getWidth,1.U)),true.B,false.B)
      value_next:=Mux(need_shift,(value.asUInt >> 1.U).asUInt + x.U(gen.getWidth.W),value.asUInt+x.U(gen.getWidth.W))
      value_next.asUInt
    }
    def set(x:Int=1):UInt={
      x.U(gen.asUInt.getWidth.W)
    }
  }
  class Table_RAM[T<:Data](num:Int, way:Int=1, gen:T)(implicit p:Parameters) extends SppDev2Module{
    val io = IO(new Bundle {
      val we = Input(Bool())
      val wr = Output(Bool())
      val windex = Input(UInt(log2Up(num).W))
      val wdata = Input(gen)
      val re = Input(Bool())
      val rr = Output(Bool())
      val rindex = Input(UInt(log2Up(num).W))
      val rdata = Output(gen)
    })

    val sram = Module(new SRAMTemplate(gen, set = num, way = way, holdRead = true, bypassWrite = false, shouldReset = true, singlePort = true))
    dontTouch(sram.io)
    sram.io.r.req.valid := io.re
    sram.io.r.req.bits.setIdx := io.rindex
    sram.io.w.req.valid := io.we
    sram.io.w.req.bits.setIdx := io.windex
    sram.io.w.req.bits.data(0) := io.wdata
    io.rdata := sram.io.r.resp.data(0)
    io.rr := sram.io.r.req.ready
    io.wr := sram.io.w.req.ready

    def initRAM: SppDev2Module.this.Table_RAM[T] = {
      this.io.re := false.B
      this.io.we := false.B
      this.io.windex := 0.U
      this.io.wdata := 0.U.asTypeOf(gen)
      this
    }
    def fullConnect(rdata: Data, rindex: Data, rvalid: Data) = {
      this.io.rindex := rindex
      rdata := this.io.rdata
    }

    def readRAM(en:Bool=false.B,rindex: UInt) = {
      this.io.re := en
      this.io.rindex := rindex
    }
    def writeRAM(en:Bool=false.B,windex: UInt, wdata: Data) = {
      this.io.we := en && !this.io.re
      this.io.windex := windex
      this.io.wdata := wdata
    }
//    assert(!RegNext(io.we && io.re), "single port SRAM should not read and write at the same time")
  }
  object ParallelMax {
    def apply[T <: Data](x: Seq[(T, UInt)]): UInt = {
      ParallelOperation(x, (a: (T, UInt), b: (T, UInt)) => {
        val maxV = WireInit(a._1.asUInt)
        val maxIndex = WireInit(a._2)
        when(a._1.asUInt < b._1.asUInt) {
          maxV := b._1
          maxIndex := b._2
        }
        (maxV.asTypeOf(x.head._1), maxIndex)
      })._2
    }
  }
  object ParallelMin {
    //width=(x,y) minV[x-1:0] minIndex[y-1:0]
    def apply(x: Seq[(UInt, UInt)], width:Int=2): UInt = {
      ParallelOperation(x, (a: (UInt, UInt), b: (UInt, UInt)) => {
        var minV = WireInit(0.U(a._1.getWidth.W))
        var minIndex = WireInit(0.U(width.W))
        minV := a._1
        minIndex := a._2
        when(a._1 > b._1) {
          minV := b._1
          minIndex := b._2
        }
        (minV, minIndex)
      })._2
    }
  }

  class PrefetchQueue_Ptr extends CircularQueuePtr[PrefetchQueue_Ptr](prefetchQueue_size)
    with HasCircularQueuePtrHelper{}

  def hash1(addr: UInt) = addr(log2Up(stParam.entry_nums) - 1, 0)

  def hash2(addr: UInt) = addr(2 * log2Up(stParam.entry_nums) - 1, log2Up(stParam.entry_nums))

  def idx(addr: UInt) = hash1(addr) ^ hash2(addr)
}

trait hasFifoParameters{
  val entries: Int = 16
  val entriesBits : Int = log2Up(entries)
  val deq_num : Int = 1
  val enq_num : Int = 4
  val beatBytes : Int = 1
  val dataBits : Int = 36
}
class multiPoint_fifo[T<:Data](gen:T,deq_num:Int=1,enq_num:Int=4,entries:Int=16)(implicit val p: Parameters) extends Module
with hasFifoParameters
with HasCircularQueuePtrHelper {
    val io = IO(new Bundle() {
      val enq = new Bundle() {
        val canAccept = Output(Bool())
        val req = Vec(enq_num, Flipped(DecoupledIO(gen)))
      }
      val deq = Vec(deq_num,DecoupledIO(gen))
    })
    class FIFO_Ptr extends CircularQueuePtr[FIFO_Ptr](entries) with HasCircularQueuePtrHelper {}
  
    // head: first valid entry
    val headPtr = RegInit(VecInit((0 until deq_num).map(_.U.asTypeOf(new FIFO_Ptr))))
    val headPtrOH = RegInit(1.U(entries.W))
    val headPtrOHShift = CircularShift(headPtrOH)
    val headPtrOHVec = VecInit.tabulate(deq_num+1)(headPtrOHShift.left);dontTouch(headPtrOHVec)
    val headPtrNext = Wire(Vec(deq_num, new FIFO_Ptr))
    // tail: first invalid entry
    val tailPtr = RegInit(VecInit((0 until enq_num).map(_.U.asTypeOf(new FIFO_Ptr))))
    val tailPtrOH = RegInit(1.U(entries.W))
    val tailPtrOHShift = CircularShift(tailPtrOH)
    val tailPtrOHVec = VecInit.tabulate(enq_num+1)(tailPtrOHShift.left) //todo: this is important,need a extra bits for selecting

    val allowEnqueue = RegInit(true.B)
    val currentValidCounter = distanceBetween(tailPtr(0), headPtr(0)) //todo can optimize bits?
    val numDeq = Mux(currentValidCounter > deq_num.U, deq_num.U, currentValidCounter)
    val num_acqDeq = PopCount(io.deq.map(_.ready))
    val numEnq = Mux(io.enq.canAccept, PopCount(io.enq.req.map(_.valid)), 0.U)
  
    val enqOffset = VecInit((0 until enq_num).map(i => PopCount(io.enq.req.map(_.valid).take(i))))
    val enqIndexOH = VecInit((0 until enq_num).map(i => tailPtrOHVec(enqOffset(i))))
    dontTouch(enqOffset)
    dontTouch(enqIndexOH)
  
    //data array
    //val dataModule = Module(new SyncDataModuleTemplate(gen, entries, numRead = deq_num, numWrite = enq_num)) //fixme : modify to mem
    val dataModule = Mem(entries, gen)
//    dontTouch(dataModule)
    val wen=WireInit(VecInit(Seq.fill(enq_num)(false.B)))
    val waddr=WireInit(VecInit(Seq.fill(enq_num)(0.U.asTypeOf(new FIFO_Ptr))))
    for (i <- 0 until enq_num) {
      wen(i) := allowEnqueue && io.enq.req(i).valid
      waddr(i) := tailPtr(enqOffset(i))
      when(wen(i)){
        dataModule(waddr(i).value) := io.enq.req(i).bits
      }
    }
    for (i <- 0 until enq_num) {
        for(j <- i+1 until enq_num){
            assert(!(wen(i) && wen(j) && waddr(i)===waddr(j)),"write conflict")
        }
    }


  val s_invalid :: s_valid :: Nil = Enum(2)
  val stateEntries = RegInit(VecInit(Seq.fill(entries)(s_invalid)));dontTouch(stateEntries)
  val isTrueEmpty = !VecInit(stateEntries.map(_ === s_valid)).asUInt.orR


  for (i <- 0 until entries) {
    val enq_updateVec = WireInit(VecInit(io.enq.req.map(_.valid).zip(enqIndexOH).map{ case (v, oh) => v && oh(i) }).asUInt)//todo understand
    dontTouch(enq_updateVec)
    when(enq_updateVec.orR && allowEnqueue) {
      stateEntries(i) := s_valid
    }
  }
  //update pointer enqueue, no matter how many enq pointer, all pointers must be updated
  val update_pointer = io.enq.req.map(_.fire).reduce(_||_)
  for (i <- 0 until enq_num) {
    tailPtr(i) := Mux(update_pointer,tailPtr(i) + numEnq,tailPtr(i))
  }
  tailPtrOH := tailPtrOHVec(numEnq)
  // dequeue
  for (i <- 0 until entries) {
    val deq_updateVec = WireInit(VecInit(io.deq.map(_.fire).zip(headPtrOHVec).map{ case (v, oh) => v && oh(i) }).asUInt)
    dontTouch(deq_updateVec)
    when(deq_updateVec.orR) {
      stateEntries(i) := s_invalid
    }
  }
  val maxDeqNum=Mux(num_acqDeq < numDeq,num_acqDeq,numDeq)
  for (i <- 0 until deq_num) {
    headPtrNext(i) := Mux(io.deq.map(_.fire).reduce(_||_),headPtr(i) + maxDeqNum,headPtr(i))
  }
  headPtr := headPtrNext
  headPtrOH := headPtrOHVec(numDeq)//ParallelPriorityMux(validMask.asUInt, headPtrOHVec)
  
  //set output valid and data bits
  val nextStepData = Wire(Vec(deq_num, gen))
  val ptrMatch = new QPtrMatchMatrix(headPtr, tailPtr)
  val deqNext = Wire(Vec(deq_num, gen))
  deqNext.zipWithIndex.map({ case (d, index) => d := nextStepData(index) }) //ParallelPriorityMux(validMask.asUInt, nextStepData.drop(index).take(deq_num + 1))})//todo:why?
  
  for (i <- 0 until deq_num) {
    io.deq(i).bits:=deqNext(i).asUInt
    io.deq(i).valid := Mux1H(headPtrOHVec(i), stateEntries) === s_valid
  }
  io.deq.map(x=>{dontTouch(x.bits)})

  for (i <- 0 until  deq_num) {
    val enqMatchVec = VecInit(ptrMatch(i))
    val enqBypassEnVec = io.enq.req.map(_.valid).zip(enqOffset).map { case (v, o) => v && enqMatchVec(o) }
    val enqBypassEn = io.enq.canAccept && (VecInit(enqBypassEnVec).asUInt.orR)
    val enqBypassData = Mux1H(enqBypassEnVec, io.enq.req.map(_.bits))
    when(io.deq(i).fire){
      nextStepData(i) := dataModule(headPtr(i).value)
    }.otherwise(
      nextStepData(i) := 0.U
    )

  }
  
  allowEnqueue := Mux(currentValidCounter > (entries - enq_num).U, false.B, numEnq <= (entries - enq_num).U - currentValidCounter)
  io.enq.req.foreach(_.ready:=allowEnqueue)
  io.enq.canAccept := allowEnqueue
}



class sppPrefetchIO(implicit p: Parameters) extends SppDev2Bundle {

}
class st2pt_dataPack(implicit p:Parameters) extends SppDev2Bundle{
  val blkAddr = UInt(blkAddrBits.W)
  val delta = SInt(stParam.delta_width.W)
  val old_sig = UInt(sigBits.W)
  val new_sig = UInt(sigBits.W)
}
class st2ghr_datapack(implicit p:Parameters) extends SppDev2Bundle{
  val req = DecoupledIO(new Bundle{
    val last_blkOffset = UInt(stParam.offset_width.W)
    val delta = SInt(ptParam.delta_width.W)
  })
  val resp = Flipped(ValidIO(new Bundle() {
    val sig=UInt(sigBits.W)
  }))
}
class pt2ghr_datapack(implicit p:Parameters)extends SppDev2Bundle{
  val req = DecoupledIO(new Bundle() {
    val signature = UInt(sigBits.W)
    val confidence = UInt(ghrParam.confidence_width.W)
    val delta = SInt(ghrParam.delta_width.W)
    val last_blkOffset = UInt(ghrParam.last_offset_width.W)
  })
  val resp = Flipped(ValidIO(new Bundle(){
    val sig = UInt(sigBits.W)
  }))
}

class BreakPointReq(implicit p: Parameters) extends SppDev2Bundle{
  val pageAddr = UInt(pageAddrBits.W)
  val parent_sig = Vec(1,UInt(sigBits.W))
  val prePredicted_blkOffset = UInt(blkOffsetBits.W)
}

case class STable[T<:Data]()(implicit p: Parameters) extends SppDev2Module {s=>
  val io=IO(new Bundle{
    val in = Flipped(DecoupledIO(new PrefetchTrain()))
    val ghr_access =new st2ghr_datapack()
    val pt_access = DecoupledIO(new st2pt_dataPack)
    val pt_bp_update = hasBpOpt.map(_ => Flipped(ValidIO(new BreakPointReq())))
  })

  class signatureTable_Entry_Ram extends SppDev2Bundle {
    val signature = UInt(sigBits.W)
    val last_blkOffset = UInt(stParam.offset_width.W)
  }
  class signatureTable_Entry extends SppDev2Bundle {
    val valid = Bool //FIXME : can remove this bit?
    val tag = UInt(stParam.tag_width.W)
    val ram = new signatureTable_Entry_Ram
  }


  object signatureTable_Entry {
    def apply:signatureTable_Entry={
      val entry=WireInit(0.U.asTypeOf(new signatureTable_Entry))
      entry
    }
    def apply(UInt_data: Bits, SInt_data: Bits)(implicit p: Parameters): signatureTable_Entry = {
      val entry=new signatureTable_Entry
      entry
    }
    def apply(valid:Bool,tag:UInt,last_blkOffset:SInt,signature:UInt,lru:UInt): signatureTable_Entry ={
      val UInt_data=Cat(valid,tag,signature,lru)
      val SInt_data=last_blkOffset
      val entry=new signatureTable_Entry
      entry
    }
    //  override def cloneType=(new signatureTable_Entry(valid,tag,last_blkOffset,signature,lru)).asInstanceOf[this.type]
  }

  val evict_index = WireInit(0.U(stParam.entryBits.W));dontTouch(evict_index)
  val has_freespace = WireInit(0.U(false.B)); dontTouch(has_freespace)
  //RAM data
  val dataRAM=Module(new SRAMTemplate(new signatureTable_Entry_Ram, set = stParam.entry_nums, way = 1, holdRead = true, bypassWrite = false, shouldReset = true, singlePort = true))

  //REG data
  private val tag_reg:Vec[UInt] = RegInit(VecInit(Seq.fill(stParam.entry_nums)(0.U(pageTagBits.W))));dontTouch(tag_reg)
  private val valid_reg:Vec[Bool] = RegInit(VecInit(Seq.fill(stParam.entry_nums)(false.B)));dontTouch(valid_reg)

  private val entry_r = signatureTable_Entry.apply;dontTouch(entry_r)
  private val entry_w = signatureTable_Entry.apply;dontTouch(entry_w)

  val need_query_ghr = RegInit(false.B)
  def lookup_cam[T<:Data](en:Bool,context: T,ram:Vec[T],matched:Bool,matched_index:UInt,name:String="cam",nums:Int=stParam.entry_nums)={
    def lookup_tagTable(context: T, ram: Vec[T]): UInt={ //Tuple2[Vec[Bool],Vec[UInt]]={
      val res = WireInit(VecInit(Seq.fill(nums)(false.B))).suggestName(name)
      ram.zipWithIndex.foreach({ x =>
        when(x._1.asUInt === context.asUInt) {
          res(x._2) := true.B
        }.otherwise {
          res(x._2) := false.B
        }
      })
      res.asUInt
    }
    val resBits = WireInit(0.U(nums.W));dontTouch(resBits)
    resBits := lookup_tagTable(context,ram)

    matched := resBits.orR && en
    def grantFirst(x:UInt):UInt = x & ~((x - 1.U)(x.getWidth-1,0).asUInt)
    matched_index := OHToUInt(grantFirst(resBits))
  }

  val replacement = ReplacementPolicy.fromString("plru", stParam.entry_nums)

  /** pipeline control signal */
  val s0_ready = WireInit(false.B)
  val s0_fire = WireInit(false.B)
  // --------------------------------------------------------------------------------
  // stage 0
  // --------------------------------------------------------------------------------
  // read sig tag

  val s0_valid = io.in.valid
  val s0_req = io.in.bits
  s0_fire := io.in.fire
  //CAM lookup
  val s0_pageAddr = WireInit(get_pageAddr(io.in.bits.addr));dontTouch(s0_pageAddr)
  val s0_hit = WireInit(false.B)

  val replacer = ReplacementPolicy.fromString("plru",stParam.entry_nums)
  val s0_replace_index = replacer.way
  val s0_matched_index = WireInit(0.U(stParam.entryBits.W))
  val s0_write_index = Mux(s0_hit, s0_matched_index, s0_replace_index)

  //readTable
  dataRAM.io.r.req.valid := s0_fire
  dataRAM.io.r.req.bits.setIdx := s0_matched_index
  entry_r.ram := dataRAM.io.r.resp.data(0)
  io.in.ready := dataRAM.io.r.req.ready

  when(s0_fire){
    replacer.access(s0_write_index)
  }

  lookup_cam(en = s0_fire, s0_pageAddr, tag_reg, s0_hit, s0_matched_index, "tag")
  lookup_cam(en = s0_fire, false.B, valid_reg, has_freespace, evict_index, "valid")

  // --------------------------------------------------------------------------------
  // stage 1
  // --------------------------------------------------------------------------------
  //
  val s1_valid = generatePipeControl(lastFire = s0_fire, thisFire = true.B, thisFlush = false.B, lastFlush = false.B)
  val s1_req = RegEnable(io.in.bits, s0_fire)

  val s1_matched_index = RegEnable(s0_matched_index,s0_fire)
  val s1_write_index =RegEnable(s0_write_index,s0_fire)
  val s1_hit = RegEnable(s0_hit,s0_fire)

  val (tag,set,tmp) = parseFullAddress(s1_req.addr)

  val old_sig=entry_r.ram.signature
  val old_last_offset = entry_r.ram.last_blkOffset
  val s1_delta_hit = set === old_last_offset
  val s1_delta = WireInit(0.S(stParam.offset_width.W))

  entry_w.tag := tag
  entry_w.ram.last_blkOffset := set
  //gen signature
  when(s1_hit){
    s1_delta := (entry_w.ram.last_blkOffset - entry_r.ram.last_blkOffset).asSInt
    entry_w.ram.signature := makeSign(old_sig, s1_delta)
  }.otherwise{
    s1_delta := (entry_w.ram.last_blkOffset - 0.U).asSInt
    entry_w.ram.signature := makeSign(0.U, s1_delta)
  }

  //update Table
  val bp_hit = WireInit(false.B)
  val bp_prePredicted_blkOff = WireInit(0.U(blkOffsetBits.W))
  val bp_matched_sig = WireInit(0.U(sigBits.W))
  val s1_origin_blkOff = WireInit(get_blockOff(get_blockAddr(s1_req.addr)));dontTouch(s1_origin_blkOff)

  dataRAM.io.w.req.valid := s1_valid
  dataRAM.io.w.req.bits.setIdx := s1_write_index
  dataRAM.io.w.req.bits.data(0) := entry_w.ram
  valid_reg(s1_write_index) := true.B
  tag_reg(s1_write_index) := get_pageAddr(s1_req.addr)

  io.pt_access.valid := s1_valid
  io.pt_access.bits.blkAddr := Mux(bp_hit,(get_blockAddr(s1_req.addr)>> blkOffsetBits << blkOffsetBits) + bp_prePredicted_blkOff,get_blockAddr(s1_req.addr))
  io.pt_access.bits.old_sig := Mux(!s1_hit,0.U,entry_r.ram.signature)
  io.pt_access.bits.new_sig := Mux(bp_hit,io.pt_access.bits.old_sig,entry_w.ram.signature)
  io.pt_access.bits.delta := s1_delta(stParam.delta_width-1,0).asSInt

  io.ghr_access.req.valid := s1_valid && !s1_hit
  io.ghr_access.req.bits.delta := ZeroExt(entry_w.ram.last_blkOffset, stParam.offset_width + 1).asSInt - ZeroExt(entry_r.ram.last_blkOffset, stParam.offset_width + 1).asSInt
  io.ghr_access.req.bits.last_blkOffset := entry_w.ram.last_blkOffset

  hasBpOpt.map({_ =>

    //breakpoint Recovery, reg realize
    def breakPointEntry() = new Bundle() {
      val valid = Bool()
      val tag = UInt(pageTagBits.W)
      val parent_sig = Vec(1,UInt(sigBits.W))
      val prePredicted_blkOffset = UInt(blkOffsetBits.W)
    }
    val io_bp_update = io.pt_bp_update.get
    val bpTable = RegInit(VecInit(Seq.fill(256)(0.U.asTypeOf(breakPointEntry()))));dontTouch(bpTable)
    val bp_page = io_bp_update.bits.pageAddr

    // write
    when(io_bp_update.valid) {
      bpTable(idx(bp_page)).valid := io_bp_update.valid
      bpTable(idx(bp_page)).tag := bp_page
      bpTable(idx(bp_page)).parent_sig.zip(io_bp_update.bits.parent_sig).foreach(x=> x._1 := x._2)
      bpTable(idx(bp_page)).prePredicted_blkOffset := io_bp_update.bits.prePredicted_blkOffset
    }
    // bp access
    val s1_bp_access_index = idx(get_pageAddr(s1_req.addr))(4,0)
    // read
    val rotate_sig = VecInit(Seq.fill(4)(0.U(sigBits.W)));dontTouch(rotate_sig)
    for(i <- 0 until(4)){
      rotate_sig(i) := CircularShift(bpTable(s1_bp_access_index).parent_sig.head).left(3*i)
    }
    bp_hit := bpTable(s1_bp_access_index).tag === get_pageAddr(s1_req.addr) && rotate_sig.map(_ === io.pt_access.bits.old_sig).reduce(_ || _)
    val bp_matched_index = WireInit(0.U(2.W))
//    lookup_cam(en = true.B, io.pt_access.bits.new_sig, rotate_sig, bp_hit, bp_matched_index, "bp_sigTag",4)
    bp_prePredicted_blkOff := bpTable(s1_bp_access_index).prePredicted_blkOffset
    bp_matched_sig := rotate_sig(bp_matched_index)
    dontTouch(bp_hit)
  })

  //perf
  XSPerfAccumulate("spp_st_hit",s1_valid && s1_hit)
  XSPerfAccumulate("spp_st_delta_hit",s1_valid && s1_delta_hit)
  XSPerfAccumulate("spp_st_bp_hit",bp_hit)
  XSPerfAccumulate("spp_st_total_access",s0_fire)
  def toPrintable:Printable = { p"${entry_w.valid}:${entry_w.tag }:${entry_w.ram.last_blkOffset}:${entry_w.ram.signature}:${replacer}"}
}

class sppPrefetchReq(implicit p:Parameters) extends PrefetchReq{
//  val hint2llc = Bool()
}

case class PTable()(implicit p:Parameters) extends SppDev2Module {P=>
  val io=IO(new Bundle{
    val in=Flipped(DecoupledIO(new st2pt_dataPack))
    val do_prefetch=DecoupledIO(new sppPrefetchReq)
    val ghr_access=new pt2ghr_datapack()
    val st2pt_bp = hasBpOpt.map(_ => Flipped(ValidIO(new BreakPointReq)))
    val pt2st_bp = hasBpOpt.map(_ => ValidIO(new BreakPointReq))
  })

  class lookahead_res_one extends SppDev2Bundle {
    val prefetch_delta = SInt(offsetBits.W)
    val valid = UInt(1.W)
    val confidence = UInt(ptParam.confidence_width.W)
  }
  class pt_data_Entry_ram(implicit p:Parameters) extends SppDev2Bundle{
    val c_delta = Vec(ptParam.way, UInt(ptParam.c_delta_width.W))
    val c_sig = UInt(ptParam.c_sig_width.W)
    val delta=Vec(ptParam.way,SInt(ptParam.delta_width.W))
  }
  class patternTable_Entry(implicit p:Parameters) extends SppDev2Bundle{
    val index=UInt(sigBits.W)
    val delta=SInt(ptParam.delta_width.W)
    val dataRam=new pt_data_Entry_ram
    def apply(x:pt_data_Entry_ram):patternTable_Entry={
      WireInit({
        val tmp=Wire(new patternTable_Entry)
        tmp.index:=0.U
        tmp.delta:=0.U
        tmp.dataRam:=x
        tmp
      })
    }
    def apply():patternTable_Entry={
      WireInit(0.U.asTypeOf(new patternTable_Entry()))
    }
  }
  class lookahead_Entry(implicit p:Parameters) extends SppDev2Bundle{
    val index = UInt(sigBits.W)
    val delta = SInt(ptParam.delta_width.W)
  }
  def readTable[T<:Data](en:Bool=false.B,ram:Table_RAM[T],index:UInt)={
      ram.readRAM(en,index)
  }
  def updateTable[T<:Data](en:Bool=false.B,entry:patternTable_Entry,ram:Table_RAM[T])={
        ram.writeRAM(en,entry.index,entry.dataRam)
  }
  /*--------------------
  stParam.sig <---> ptParam.index
  ---------------------*/
  def init_r_entry[T <: Data](entry: patternTable_Entry, ram:Table_RAM[T]) = {
    val ram = WireInit(0.S(ptParam.s_data_width.W)).asTypeOf(new pt_data_Entry_ram())
    val rvalid = WireInit(false.B)
    dataRAM.fullConnect(ram,entry.index,rvalid)
    read_entry.index := DontCare
    read_entry.delta := DontCare
    entry.dataRam.c_delta := ram.c_delta
    entry.dataRam.c_sig := ram.c_sig
    entry.dataRam.delta := ram.delta
  }

  //RAM data
  val dataRAM=Module(new Table_RAM[Data](ptParam.entry_nums,1,new pt_data_Entry_ram()).suggestName("pt_dataRAM"))
  dataRAM.initRAM
  //REG data
  val tag_reg=RegInit(0.U(stParam.entry_nums.W))

  val read_entry=new patternTable_Entry().apply();dontTouch(read_entry)
  val write_entry=new patternTable_Entry().apply();dontTouch(write_entry)
  val lookahead_entry_reg=RegInit(0.U.asTypeOf(new lookahead_Entry()))

  init_r_entry(read_entry,dataRAM)

  def find_highest_confidence_delta(entry:patternTable_Entry)={
    val maxIndex=WireInit(0.U(log2Up(ptParam.way).W)).suggestName("maxIndex");dontTouch(maxIndex)
    maxIndex:=ParallelMax.apply(entry.dataRam.c_delta.zipWithIndex.map(x=>(x._1,x._2.asUInt(ptParam.entryBits.W))))
    maxIndex
  }
  def query_matched_delta(query_delta:SInt,entry:patternTable_Entry):Vec[Bool]={
    var res=VecInit(Seq.fill(ptParam.way)(false.B)).suggestName("")
    entry.dataRam.delta.zipWithIndex.foreach({case(x,index)=>
      when(x===query_delta){
        res(index):=true.B
      }.otherwise{
        res(index):=false.B
      }})
    res
  }
  def query_lowest_confidence_delta(entry:patternTable_Entry):UInt={
    val minIndex = WireInit(0.U(log2Up(ptParam.c_delta_width).W)).suggestName("minIndex");dontTouch(minIndex)
    minIndex := ParallelMin.apply(entry.dataRam.c_delta.zipWithIndex.map(x=>(x._1,x._2.asUInt)))
    minIndex
  }

  val lookahead_res: Vec[lookahead_res_one] = WireInit(VecInit(Seq.fill(ptParam.way)(0.U.asTypeOf(new lookahead_res_one))));dontTouch(lookahead_res)
  val lookahead_continue = WireInit(false.B);dontTouch(lookahead_continue)
  val lookahead_continue_prev = RegNext(lookahead_continue)
  val is_cross_pageBoundary = WireInit(false.B);dontTouch(is_cross_pageBoundary)
  // val entry=WireInit((new patternTable_Entry).Lit(_.c_delta->VecInit(Seq.fill(ptParam.way)(0.U)),_.c_sig->0.U,_.delta->VecInit(Seq.fill(ptParam.way)(0.S))))

    // pttable state
    val s_idle :: s_update_r :: s_update_w :: s_first_lookahead ::s_lookahead :: s_makeSign :: Nil = Enum(6)
    val state_reg = RegInit(s_idle)
    //        val popstate_reg = RegInit(s_idle)
    dontTouch(state_reg)
    switch(state_reg){
      is(s_idle){
        when(io.in.fire && io.in.bits.old_sig=/=0.U){
          state_reg := s_update_r
        }.elsewhen(io.in.fire){
          state_reg := s_first_lookahead
        }.otherwise{state_reg:=s_idle;}
      }
      is(s_update_r){
        state_reg := s_update_w
      }
      is(s_update_w){
        state_reg := s_first_lookahead
      }
      is(s_first_lookahead){
        state_reg := s_makeSign
      }
      is(s_lookahead) {
        state_reg := s_makeSign
      }
      is(s_makeSign){
        when(lookahead_continue) {
          state_reg := s_lookahead
        }.otherwise {
          state_reg := s_idle
        }
      }
    }

    val matched_delta_index = OneHot.OH1ToUInt(query_matched_delta(read_entry.delta, read_entry).asUInt)-1.U
    val lowest_delta_index = query_lowest_confidence_delta(read_entry)
    val highest_delta_index = find_highest_confidence_delta(read_entry)
    val is_all_saturated = WireInit(false.B)
    val c_delta: Seq[ramCounter[UInt]] = Seq.tabulate(ptParam.way)(x=>new ramCounter[UInt](read_entry.dataRam.c_delta(x), is_all_saturated, s"update_delta_${x}"))
    val c_sig: ramCounter[UInt] = new ramCounter[UInt](read_entry.dataRam.c_sig, is_all_saturated, "update_sig")
    val lookahead_counter = sppCounter.apply(UInt(log2Up(ptParam.max_lookup_depth).W), None,false, "lookahead_depth_c");dontTouch(lookahead_counter.value)

    is_all_saturated := c_delta.map(_.saturated).reduce(_ || _) || c_sig.saturated
    val hit = (state_reg===s_update_w) && read_entry.delta =/= 0.S && query_matched_delta(read_entry.delta, read_entry).reduce(_ || _)
    //calculate confidecnce for each way counter
    val parent_conf=RegInit(0.U(ptParam.confidence_width.W))
    when(state_reg===s_idle){
      parent_conf := 0.U
    }.elsewhen(state_reg===s_first_lookahead){
      parent_conf := 100.U
    }.elsewhen(state_reg===s_makeSign){
      parent_conf := parent_conf * (lookahead_res(highest_delta_index).confidence - 1.U) /100.U
    }

    for(i <- 0 until ptParam.way){
      when(c_delta(i).value === 0.U) {
        lookahead_res(i).confidence := 0.U
      }.elsewhen(c_delta(i).value === 1.U && c_sig.value === 1.U) {
        lookahead_res(i).confidence := 100.U //TODO: first allocated entry is needed to set high confidence?
      }.otherwise {
        lookahead_res(i).confidence := parent_conf * c_delta(i).value * 100.U / (c_sig.value * 100.U)
      }
    }

  //lookahead operation
    // crtl from looahead_res
    for (i <- 0 until ptParam.way) {
      lookahead_res(i).valid := lookahead_res(i).confidence >= ptParam.pfThreshold.U
      lookahead_res(i).prefetch_delta := read_entry.dataRam.delta(i)
    }
//    when(state_reg === s_makeSign) {
//
//    }
    val path_accum_delta=RegInit(0.S((ptParam.delta_width+1).W));dontTouch(path_accum_delta)
    lookahead_continue := (state_reg === s_first_lookahead || state_reg === s_lookahead || state_reg === s_makeSign) && lookahead_res.map(_.valid.asBool).reduce(_ || _) &&
      path_accum_delta < 64.S && !lookahead_counter.saturated
    when(state_reg === s_makeSign){
      lookahead_counter.add(1.U)
    }.elsewhen(state_reg === s_idle) {
      lookahead_counter.reset(0.U)
    }

    val lookahead_sel_delta = WireInit(read_entry.dataRam.delta(highest_delta_index));dontTouch(lookahead_sel_delta)
    val lookahead_sel_sig = WireInit(makeSign(lookahead_entry_reg.index, lookahead_sel_delta));dontTouch(lookahead_sel_sig) //todo: whether need to lose precision?
    //readTable
    when(state_reg === s_update_r || state_reg === s_update_w) {
      read_entry.index := io.in.bits.old_sig
    }.elsewhen(state_reg === s_first_lookahead){
      read_entry.index := io.in.bits.new_sig
    }.otherwise{
      read_entry.index := lookahead_sel_sig
    }

    //match delta
    when(state_reg === s_update_r || state_reg === s_update_w || state_reg === s_first_lookahead){
      read_entry.delta := io.in.bits.delta
    }.otherwise{
      read_entry.delta := lookahead_entry_reg.delta
    }

    when(state_reg === s_first_lookahead || state_reg === s_lookahead) {
      lookahead_entry_reg.index := read_entry.index //index:old signature
      lookahead_entry_reg.delta := lookahead_sel_delta
    }
    readTable(en = (state_reg===s_update_r || state_reg===s_first_lookahead || state_reg === s_lookahead), ram = dataRAM, index = read_entry.index)

    //updateTable --- delta entry update
    val replace_delta_index = Mux(hit&&state_reg===s_update_w,matched_delta_index,lowest_delta_index)
    write_entry.index := RegEnable(io.in.bits.old_sig, io.in.fire)
    write_entry.delta := RegEnable(io.in.bits.delta, io.in.fire)
    write_entry.dataRam.c_sig := c_sig.add(1)
    for (i <- 0 until ptParam.way) {
      write_entry.dataRam.delta(i) := Mux(i.U===replace_delta_index, write_entry.delta, read_entry.dataRam.delta(i))
      write_entry.dataRam.c_delta(i) := Mux(i.U===replace_delta_index, Mux(hit,c_delta(i).add(1),c_delta(i).set(1)),c_delta(i).get_shiftValue)
    }
    updateTable(state_reg === s_update_w, entry = write_entry, ram = dataRAM)

    //solve cross page boundary

    when(state_reg === s_idle){
      path_accum_delta := 0.S
    }.elsewhen(state_reg === s_makeSign) {
      path_accum_delta := path_accum_delta + lookahead_res(highest_delta_index).prefetch_delta
    }.otherwise{
      path_accum_delta := path_accum_delta
    }

    val trigger_blkAddr_reg = RegEnable(io.in.bits.blkAddr, io.in.fire)
    val trigger_blkOff = get_blockOff(trigger_blkAddr_reg)
    val trigger_pageAddr = get_pageAddr(Cat(trigger_blkAddr_reg,0.U(blkOffsetBits.W)))

    val prePredicted_blkOffset=WireInit(0.U((offsetBits+1).W))
    prePredicted_blkOffset:= (path_accum_delta + ZeroExt(trigger_blkOff,offsetBits+1).asSInt).asUInt

    is_cross_pageBoundary := state_reg===s_makeSign && prePredicted_blkOffset > 64.U

    io.ghr_access.req.valid:=is_cross_pageBoundary
    io.ghr_access.req.bits.confidence := lookahead_res(highest_delta_index).confidence
    io.ghr_access.req.bits.signature := lookahead_entry_reg.index
    io.ghr_access.req.bits.delta := lookahead_entry_reg.delta
    io.ghr_access.req.bits.last_blkOffset := get_blockOff(io.in.bits.blkAddr)

    //add nextline prefetch
    val do_next_line = WireInit(false.B);
    dontTouch(do_next_line)
    do_next_line := false.B//state_reg === s_makeSign && RegNext(state_reg, 0.U) === s_first_lookahead && !lookahead_res.map(_.valid).reduce(_ | _)

    //put predicted addr into prefetch queue
    val prefetch_fifo = Module(new multiPoint_fifo[UInt](gen = {UInt(ptParam.delta_width.W)},enq_num = ptParam.enq_num, entries = prefetchQueue_size))

    for(i <- 0 until  ptParam.enq_num){
      when(do_next_line) {
          prefetch_fifo.io.enq.req(i).valid := true.B
          prefetch_fifo.io.enq.req(i).bits := i.U
      }.otherwise {
          prefetch_fifo.io.enq.req(i).valid := state_reg === s_makeSign && lookahead_res(i).valid.asBool
          prefetch_fifo.io.enq.req(i).bits := (lookahead_res(i).prefetch_delta + path_accum_delta).asUInt
      }
      // prefetch_fifo.io.enq.req(i).valid := state_reg === s_makeSign && lookahead_res(i).valid.asBool
      // prefetch_fifo.io.enq.req(i).bits := (lookahead_res(i).prefetch_delta + path_accum_delta).asUInt
    }

    //sendout prefetch
    prefetch_fifo.io.deq(0).ready := io.do_prefetch.ready
    val prefetch_blkAddr = WireInit(0.U(blkAddrBits.W))
    val prefetch_addr = WireInit(0.U(fullAddressBits.W))
    prefetch_addr := prefetch_blkAddr<<offsetBits
    io.do_prefetch.valid := prefetch_fifo.io.deq(0).valid
    if(hasBpOpt.nonEmpty){
      prefetch_blkAddr := (trigger_blkAddr_reg.asSInt + (prefetch_fifo.io.deq(0).bits.asUInt.asSInt)).asUInt
    }else{
      prefetch_blkAddr := (trigger_blkAddr_reg.asSInt + prefetch_fifo.io.deq(0).bits.asUInt.asSInt).asUInt
    }
    
    val do_prefetch_pageAddr = WireInit(get_pageAddr(prefetch_addr));dontTouch(do_prefetch_pageAddr)

    io.do_prefetch.bits.addr := prefetch_addr
    //io.do_prefetch.bits.hint2llc := false.B //TODO: support multi-level

    io.in.ready:=state_reg===s_idle


    hasBpOpt.map({_ =>
      val bp_update = WireInit(false.B)
      val io_pt2st_bp = io.pt2st_bp.get
      val bp_valid = RegNext(io_pt2st_bp.valid)
      val bp_parent_sig = RegNext(io_pt2st_bp.bits.parent_sig)
      val bp_block = RegNext(io_pt2st_bp.bits.prePredicted_blkOffset)
      val enbp = WireInit(true.B)

      bp_update := RegNextN(lookahead_continue_prev && !lookahead_continue && lookahead_counter.value > 4.U, 2)


      io_pt2st_bp.valid := enbp && bp_update
      when(io_pt2st_bp.valid) {
        io_pt2st_bp.bits.pageAddr := trigger_pageAddr
        io_pt2st_bp.bits.prePredicted_blkOffset := prePredicted_blkOffset
//        io_pt2st_bp.bits.parent_sig.head := lookahead_sel_sig
        for(i <-0 until(io_pt2st_bp.bits.parent_sig.length)) {
          io_pt2st_bp.bits.parent_sig(i) := RegNextN(lookahead_sel_sig,i,Some(0.U))
        }

      }.otherwise {
        io.pt2st_bp.get.bits := 0.U.asTypeOf(io.pt2st_bp.get.bits.cloneType)
      }

    })

    //perf
    XSPerfAccumulate(s"spp_pt_do_nextline", do_next_line)
    XSPerfAccumulate(s"spp_pt_write",dataRAM.io.we && dataRAM.io.wr)
    XSPerfAccumulate(s"spp_pt_read",dataRAM.io.re && dataRAM.io.rr)
    for (i <- 0 until ptParam.entry_nums) {
      XSPerfAccumulate("spp_pt_touched_entry_" + i.toString,
      (state_reg === s_update_r  || state_reg === s_first_lookahead || state_reg === s_lookahead) &&
        (read_entry.index(ptParam.entryBits - 1, 0) === i.U(ptParam.entryBits.W).asUInt)
      )
    }
    for (i <- 0 until (4)) {
      XSPerfAccumulate(s"spp_pt_way_${i.toString}_firstlook_token", state_reg === s_first_lookahead && lookahead_res(i).valid.asBool)
    }
    for (off <- (-63 until 64 by 1).toList) {
      if (off < 0) {
        XSPerfAccumulate("spp_pt_pfDelta_neg_" + (-off).toString, prefetch_fifo.io.deq(0).valid && prefetch_fifo.io.deq.head.bits === off.S(ptParam.delta_width.W).asUInt)
      } else {
        XSPerfAccumulate("spp_pt_pfDelta_pos_" + off.toString, prefetch_fifo.io.deq(0).valid && prefetch_fifo.io.deq.head.bits.asSInt === off.S(ptParam.delta_width.W))
      }
    }

}

case class Global_history_reg(entry_num:Int)(implicit p:Parameters) extends SppDev2Module{
  val io =IO(new Bundle() {
    val st_access = Flipped(new st2ghr_datapack())
    val pt_access = Flipped(new pt2ghr_datapack())
  })
  class ram_entry(implicit p: Parameters) extends SppDev2Bundle{
    val signature = UInt(sigBits.W)
    val confidence = UInt(ghrParam.confidence_width.W)
    val last_blkOffset = UInt(ghrParam.last_offset_width.W)
    val delta = SInt(ghrParam.delta_width.W)
  }
  class GHR_Entry(implicit p: Parameters) extends SppDev2Bundle {
    val index = UInt(ghrParam.entry_bits.W)
    val ram = new ram_entry()
  }
  val read_ghr_entry=WireInit(0.U.asTypeOf(new GHR_Entry()))
  val write_ghr_entry=WireInit(0.U.asTypeOf(new GHR_Entry()))
  val ghr_entry=(new GHR_Entry())
  val ghr_reg=RegInit(VecInit(Seq.fill(ghrParam.entry_nums)(0.U.asTypeOf(new GHR_Entry().ram))));dontTouch(ghr_reg) //todo: update to RAM if table configs largger
  // --------------------------------------------------------------------------------
  // lru replacemeny
  // --------------------------------------------------------------------------------
  val replacer_wen = RegInit(false.B)
  val replacer = ReplacementPolicy.fromString("plru",ghrParam.entry_nums)


  val replace_index=RegInit(0.U(ghrParam.entry_bits.W))
  val matched_index=WireInit(0.U(ghrParam.entry_bits.W))
  val hit=WireInit(false.B)
  def find_lowest_entry(entry: Vec[ram_entry]): UInt = {
    val minIndex = WireInit(0.U(ghrParam.entry_bits.W)).suggestName("minIndex");
    minIndex := ParallelMin.apply(entry.zipWithIndex.map(x => (x._1.confidence, x._2.asUInt)),width =ghrParam.entry_bits)
    minIndex
  }
  def query_matched_entry(query_delta: SInt,query_last_offset:UInt, entry: Vec[ram_entry],matched_index:UInt): Vec[Bool] = {
    var res = VecInit(Seq.fill(ghrParam.entry_nums)(false.B)).suggestName("ghr_res")
    entry.zipWithIndex.foreach({ case (ghr, index) =>
      var delta=ghr.delta.asUInt//(ghrParam.delta + ghrParam.last_blkOffset.asSInt - 64.S).asUInt & 0x3f.U
      when(delta.orR && delta === query_delta.asUInt){
        matched_index:=index.U
        res(index) := true.B
      }.otherwise {
        matched_index:=0.U
        res(index) := false.B
      }

    })
    res
  }

  val query_delta = WireInit(0.S(ghrParam.delta_width.W))
  val query_last_offset = WireInit(0.U(ghrParam.last_offset_width.W))
  when(io.pt_access.req.fire){
    query_delta:=io.pt_access.req.bits.delta
    query_last_offset:=io.pt_access.req.bits.last_blkOffset
  }.elsewhen(io.st_access.req.fire){
    query_delta := io.st_access.req.bits.delta
    query_last_offset := io.st_access.req.bits.last_blkOffset
  }
  read_ghr_entry.index := matched_index
  read_ghr_entry.ram := ghr_reg(matched_index)
  hit := query_matched_entry(query_delta,query_last_offset,ghr_reg,matched_index).reduce(_||_) || query_delta === write_ghr_entry.ram.delta

  val s0_write_index = WireInit(0.U(ghrParam.entry_bits.W))
  s0_write_index := Mux(hit,matched_index,replacer.way)
  replacer.access(s0_write_index)

  write_ghr_entry.index := RegEnable(s0_write_index,io.pt_access.req.fire)
  write_ghr_entry.ram.signature := RegEnable(io.pt_access.req.bits.signature,io.pt_access.req.fire)
  write_ghr_entry.ram.delta := RegEnable(io.pt_access.req.bits.delta,io.pt_access.req.fire)
  write_ghr_entry.ram.last_blkOffset := RegEnable(io.pt_access.req.bits.last_blkOffset,io.pt_access.req.fire)
  write_ghr_entry.ram.confidence := RegEnable(Mux(io.pt_access.req.bits.confidence>read_ghr_entry.ram.confidence,
    io.pt_access.req.bits.confidence,read_ghr_entry.ram.confidence),io.pt_access.req.fire)

  val wen=RegNext((hit&&io.st_access.req.fire)||io.pt_access.req.fire)
  for (i <- 0 until ghrParam.entry_nums) {
    ghr_reg(i):=Mux(i.U === write_ghr_entry.index && wen, write_ghr_entry.ram,ghr_reg(i))
  }
  io.st_access.req.ready := true.B
  io.pt_access.req.ready := ~io.st_access.req.fire

  io.st_access.resp.valid := hit && io.st_access.req.fire
  io.st_access.resp.bits.sig := read_ghr_entry.ram.signature

  io.pt_access.resp.valid := hit && io.pt_access.req.fire
  io.pt_access.resp.bits.sig := read_ghr_entry.ram.signature

  XSPerfAccumulate("ghr_pf_crosPageBoundary_nums", io.pt_access.req.valid)
}

case class PrefetchFilter2()(implicit p: Parameters) extends SppDev2Module {
  val io = IO(new Bundle() {
    val pt_req = Flipped(DecoupledIO(new sppPrefetchReq))
    val req = DecoupledIO(new sppPrefetchReq)
  })
  def getPF_blkOffset(x:UInt) = x(pffParam.region_addrBits-1,offsetBits)

  class PrefetchFilterEntry_ram()(implicit p: Parameters) extends SppDev2Bundle {
    val tag = UInt(pffParam.tagBits.W)
//    val region_addr = UInt(pffParam.region_addrBits.W)
//    val region_bits = UInt(16.W)
//    val filter_bits = UInt(16.W)
    val filterVec = Vec(pffParam.filterBits, Bool())
  }

  def lookup_cam[T<:Data](nums:Int =1,en: Bool, context: T, ram: Vec[T], matched: Bool, matched_index: UInt, name: String = "cam") = {
    def lookup_tagTable(context: T, ram: Vec[T]): UInt = {
      val res = WireInit(VecInit(Seq.fill(nums)(false.B))).suggestName(name)
      ram.zipWithIndex.foreach({ x =>
        when(x._1.asUInt === context.asUInt) {
          res(x._2) := true.B
        }.otherwise {
          res(x._2) := false.B
        }
      })
      res.asUInt
    }

    val resBits = WireInit(0.U(nums.W));
    dontTouch(resBits)
    resBits := lookup_tagTable(context, ram)

    matched := resBits.orR && en

    def grantFirst(x: UInt): UInt = x & ~((x - 1.U)(x.getWidth - 1, 0).asUInt)

    matched_index := OHToUInt(grantFirst(resBits))
  }

  private val entries_tag = RegInit(VecInit(Seq.fill(pffParam.filter_size)(0.U(pffParam.tagBits.W))))
  private val entries_filter = RegInit(VecInit(Seq.fill(pffParam.filter_size)(0.U(pffParam.filterBits.W))))
  private val valid_reg = RegInit(VecInit(Seq.fill(pffParam.filter_size)(false.B)))

  val replacement = ReplacementPolicy.fromString("plru", pffParam.filter_size)

  val prev_valid = RegNext(io.pt_req.valid, false.B)
  val prev_pt_req = RegEnable(io.pt_req.bits, io.pt_req.valid)

  /** pipeline control signal */
  val s0_ready, s1_ready = WireInit(false.B)
  val s0_fire, s1_fire = WireInit(false.B)
  // --------------------------------------------------------------------------------
  // stage 0
  // --------------------------------------------------------------------------------
  // entries lookup
  val s0_valid = WireInit(false.B)
  val s0_can_go = WireInit(false.B)
  val s0_req = io.pt_req.bits
  val s0_match_prev = prev_valid && (s0_req.addr === prev_pt_req.addr)

  val s0_tag_hit = WireInit(false.B);dontTouch(s0_tag_hit)
  val s0_entry_hit = WireInit(false.B);dontTouch(s0_entry_hit)
  val s0_filter_hit = WireInit(false.B); dontTouch(s0_filter_hit)

  val s0_hit = s0_entry_hit && s0_filter_hit
  val s0_update_way = WireInit(0.U(log2Up(pffParam.filter_size).W))

//  val s0_replace_index = WireInit(0.U(log2Up(pffParam.filter_size).W)); dontTouch(s0_replace_index)
//  val s0_has_freespace = WireInit(0.U(false.B)); dontTouch(s0_has_freespace)


  s0_valid := io.pt_req.valid && s0_can_go

//  val s0_regionVec = WireInit(VecInit(Seq.fill(pffParam.filterBits)(false.B)));dontTouch(s0_regionVec)
  val s0_pf_offset = getPF_blkOffset(s0_req.addr)
  val s0_pf_offsetOH = UIntToOH(s0_pf_offset,pffParam.filterBits)
  val s0_achored_fVec = WireInit(0.U(pffParam.filterBits.W))
  val s0_matchTag = WireInit(s0_req.addr)
  lookup_cam(pffParam.filter_size,s0_fire, s0_matchTag, entries_tag, s0_tag_hit, s0_update_way, "entry_tag")

  when(s0_fire && s0_tag_hit){
    s0_achored_fVec := entries_filter(s0_update_way)
    s0_filter_hit := s0_achored_fVec(s0_pf_offset)
  }

  val s0_replace_way = replacement.way

  s0_entry_hit := s0_tag_hit && valid_reg(s0_update_way)
  val s0_access_way = Mux(s0_entry_hit, s0_update_way, s0_replace_way)

  when(s0_valid){
    replacement.access(s0_access_way)
  }

  io.pt_req.ready := s0_ready
  io.req.valid := s0_valid && !s0_filter_hit
  io.req.bits.addr := s0_req.addr
  //io.req.bits.hint2llc := s0_req.hint2llc
  val s1_s0_way_conflict = RegInit(false.B)
  s0_fire := s0_valid && s0_ready
  s0_ready := s1_ready && s0_can_go
  s0_can_go := true.B//!s1_s0_way_conflict //FIXME: improve to RAM,now default can read and write at the same cycle
  // --------------------------------------------------------------------------------
  // stage 1
  // --------------------------------------------------------------------------------
  // update or alloc
  val s1_valid=generatePipeControl(lastFire = s0_fire, thisFire = s1_fire, thisFlush = false.B, lastFlush = false.B)
  val s1_req = RegEnable(s0_req, s0_valid)
  val s1_access_way = RegNext(s0_access_way)
  val s1_pf_offset = RegNext(s0_pf_offset)
  val s1_achored_fVec = RegNext(s0_achored_fVec)
  val s1_hit = RegNext(s0_entry_hit && s0_filter_hit)
  val s1_w_entry = WireInit(0.U.asTypeOf(new PrefetchFilterEntry_ram()))

  s1_s0_way_conflict := s1_valid && (s0_access_way === s1_access_way)

  s1_ready := true.B//!s0_valid
  s1_fire := s1_valid && s1_ready

  val alloc = WireInit(s0_valid && !s0_entry_hit )
  val update = WireInit(s0_valid && s0_entry_hit && !s0_filter_hit)
  when(update) {
    val update_fvec = WireInit(0.U(pffParam.filterBits.W))
    update_fvec := s0_achored_fVec | UIntToOH(s0_pf_offset,pffParam.filterBits)
    entries_filter(s0_access_way) := update_fvec
  }
  when(alloc) {
    entries_tag(s0_access_way) := s0_req.addr
    entries_filter(s0_access_way) := UIntToOH(s0_pf_offset,pffParam.filterBits)
    valid_reg(s0_access_way) := true.B
  }

   XSPerfAccumulate("spp_filter_recv_req", io.pt_req.valid)
   XSPerfAccumulate("spp_filter_hit", s1_valid && s1_hit)
   XSPerfAccumulate("spp_filter_l2_req", io.req.valid)
}

class SppDev2Prefetch()(implicit p: Parameters) extends SppDev2Module {
  val io = IO(new Bundle() {
    val train = Flipped(DecoupledIO(new PrefetchTrain))
//    val tlb_req = new L2ToL1TlbIO(nRespDups= 1)
    val req = DecoupledIO(new sppPrefetchReq)
//    val resp = Flipped(DecoupledIO(new PrefetchResp))
  })

  io.train.ready := true.B
  //    io.L2.req := DontCare
//  io.resp.ready := true.B
  val base_addr =RegEnable(io.train.bits.addr,io.train.valid)

  val is_crossedPage = WireInit(false.B)
  is_crossedPage := io.req.valid && (get_pageAddr(io.train.bits.addr) =/= get_pageAddr(io.req.bits.addr))

  //signature table
  val ST=Module(STable())
  ST.io.in <> io.train
  // pattern table
  val PT=Module(PTable());dontTouch(PT.io.do_prefetch);
  PT.io.in <> ST.io.pt_access
  hasBpOpt.map({_ =>
    PT.io.st2pt_bp.get := DontCare
    ST.io.pt_bp_update.get <> PT.io.pt2st_bp.get
  })

  //global history registor
  val GHR=Module(Global_history_reg(ghrParam.entry_nums))

  val FT=Module(PrefetchFilter2())
  GHR.io.st_access<>ST.io.ghr_access
  GHR.io.pt_access<>PT.io.ghr_access
  PT.io.ghr_access<>GHR.io.pt_access
  FT.io.pt_req <> PT.io.do_prefetch
  io.req <> FT.io.req
  FT.io.req.ready := true.B

  // --------------------------------------------------------------------------------
  // tlb
  // --------------------------------------------------------------------------------
//  io.tlb_req.req.valid := false.B
//  io.tlb_req.req.bits := 0.U.asTypeOf(io.tlb_req.req.bits.cloneType)
//  io.tlb_req.req_kill := false.B
//  io.tlb_req.resp.ready := true.B
//  dontTouch(io.tlb_req)

  dontTouch(ST.io)
  dontTouch(PT.io)
  dontTouch(GHR.io)
  dontTouch(FT.io)
  //perf
  // val perf = IO(new PerfInfoIO())
  // BoringUtils.addSource(perf.XSPERF_CLEAN, "XSPERF_CLEAN")
  // BoringUtils.addSource(perf.XSPERF_DUMP, "XSPERF_DUMP")

  XSPerfAccumulate("spp_crossedPage_nums", is_crossedPage)
  //  val trainDB = ChiselDB.createTable("L2Prefetcher_train_input", new PrefetchTrain())
  //  val reqDB = ChiselDB.createTable("L2Prefetcher_req", new PrefetchReq())
  //  trainDB.log(io.train.bits, io.train.valid, "L2Prefetcher", clock, reset)
  //  reqDB.log(io.req.bits, io.req.valid, "L2Prefetcher_req_output", clock, reset)
  //  FileRegisters.write(fileDir = s"./test_run_dir/SppDev2PrefetchTest")
}


