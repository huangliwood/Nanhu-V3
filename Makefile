#***************************************************************************************
# Copyright (c) 2020-2021 Institute of Computing Technology, Chinese Academy of Sciences
# Copyright (c) 2020-2021 Peng Cheng Laboratory
#
# XiangShan is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

TOP = XSTop
SIM_TOP   = SimTop
FPGATOP = top.TopMain
BUILD_DIR ?= ./build
TOP_V = $(BUILD_DIR)/$(TOP).sv
SCALA_FILE = $(shell find ./src/main/scala -name '*.scala')
TEST_FILE = $(shell find ./src/test/scala -name '*.scala')
MEM_GEN = ./scripts/vlsi_mem_gen

SIMTOP  = top.SimTop
IMAGE  ?= temp
CONFIG ?= DefaultConfig
NUM_CORES ?= 1
ABS_WORK_DIR := $(shell pwd)
# VCS sim options
RUN_BIN_DIR ?= $(ABS_WORK_DIR)/ready-to-run
RUN_BIN ?= coremark-2-iteration
CONSIDER_FSDB ?= 1
MFC ?= 0

ifdef FLASH
	RUN_OPTS := +flash=$(RUN_BIN_DIR)/$(RUN_BIN).bin
else
	RUN_OPTS := +workload=$(RUN_BIN_DIR)/$(RUN_BIN).bin
endif
ifeq ($(CONSIDER_FSDB),1)
	RUN_OPTS += +dump-wave=fsdb
endif
RUN_OPTS += +diff=$(ABS_WORK_DIR)/ready-to-run/riscv64-nemu-interpreter-so
# RUN_OPTS += +no-diff
RUN_OPTS += -fgp=num_threads:4,num_fsdb_threads:4
RUN_OPTS += -assert finish_maxfail=30 -assert global_finish_maxfail=10000
# co-simulation with DRAMsim3
ifeq ($(WITH_DRAMSIM3),1)
ifndef DRAMSIM3_HOME
$(error DRAMSIM3_HOME is not set)
endif
override SIM_ARGS += --with-dramsim3
endif

# top-down
ifeq ($(ENABLE_TOPDOWN),1)
override SIM_ARGS += --enable-topdown
endif

# emu for the release version
RELEASE_ARGS = --disable-all --fpga-platform
DEBUG_ARGS   = --enable-difftest

ifeq ($(VCS),1)
RELEASE_ARGS += --emission-options disableRegisterRandomization -X sverilog --output-file $(TOP)
DEBUG_ARGS += --emission-options disableRegisterRandomization -X sverilog --output-file $(SIM_TOP)
else
RELEASE_ARGS += --emission-options disableRegisterRandomization -E verilog --output-file $(TOP)
DEBUG_ARGS += -E verilog --output-file $(SIM_TOP)
endif

ifeq ($(RELEASE),1)
override SIM_ARGS += $(RELEASE_ARGS)
else
override SIM_ARGS += $(DEBUG_ARGS)
endif

.DEFAULT_GOAL = verilog

help:
	mill -i XiangShan.test.runMain $(SIMTOP) --help

$(TOP_V): $(SCALA_FILE)
	mkdir -p $(@D)
	time -o $(@D)/time.log mill -i XiangShan.runMain $(FPGATOP) -td $(@D) \
		--config $(CONFIG) --full-stacktrace --num-cores $(NUM_CORES) \
		$(RELEASE_ARGS) | tee build/make.log
	sed -e 's/\(peripheral\|memory\)_0_\(aw\|ar\|w\|r\|b\)_bits_/m_\1_\2_/g' \
	-e 's/\(dma\)_0_\(aw\|ar\|w\|r\|b\)_bits_/s_\1_\2_/g' $@ > $(BUILD_DIR)/tmp.v
	sed -e 's/\(peripheral\|memory\)_0_\(aw\|ar\|w\|r\|b\)_/m_\1_\2_/g' \
	-e 's/\(dma\)_0_\(aw\|ar\|w\|r\|b\)_\(ready\|valid\)/s_\1_\2_\3/g' $(BUILD_DIR)/tmp.v > $(BUILD_DIR)/tmp1.v
	rm $@ $(BUILD_DIR)/tmp.v
	mv $(BUILD_DIR)/tmp1.v $@
	@git log -n 1 >> .__head__
	@git diff >> .__diff__
	@sed -i 's/^/\/\// ' .__head__
	@sed -i 's/^/\/\//' .__diff__
	@cat .__head__ .__diff__ $@ > .__out__
	@mv .__out__ $@
	@rm .__head__ .__diff__

verilog: $(TOP_V)

SIM_TOP_V = $(BUILD_DIR)/$(SIM_TOP).sv
$(SIM_TOP_V): $(SCALA_FILE) $(TEST_FILE)
	mkdir -p $(@D)
	@echo "\n[mill] Generating Verilog files..." > $(@D)/time.log
	@date -R | tee -a $(@D)/time.log
	time -o $(@D)/time.log mill -i XiangShan.test.runMain $(SIMTOP) -td $(@D) \
		--config $(CONFIG) --full-stacktrace --num-cores $(NUM_CORES) \
		$(SIM_ARGS) | tee build/make.log

ifeq ($(RELEASE),1)
ifeq ($(VCS), 1)
	mv $(BUILD_DIR)/$(TOP).sv $(BUILD_DIR)/$(SIM_TOP).sv
else
	mv $(BUILD_DIR)/$(TOP).v $(BUILD_DIR)/$(SIM_TOP).v
endif
endif

ifneq ($(VCS), 1)
	mv $(BUILD_DIR)/$(SIM_TOP).v $(SIM_TOP_V)
	sed -i -e 's/$$fatal/xs_assert(`__LINE__)/g' $(SIM_TOP_V)
endif
	python3 scripts/assertion_alter.py -o $(SIM_TOP_V) $(SIM_TOP_V)

	@git log -n 1 >> .__head__
	@git diff >> .__diff__
	@sed -i 's/^/\/\// ' .__head__
	@sed -i 's/^/\/\//' .__diff__
	@cat .__head__ .__diff__ $@ > .__out__
	@mv .__out__ $@
	@rm .__head__ .__diff__

FILELIST := $(ABS_WORK_DIR)/build/cpu_flist.f

sim-verilog: $(SIM_TOP_V)
	find $(ABS_WORK_DIR)/build -name "*.v" > $(FILELIST)
	find $(ABS_WORK_DIR)/build -name "*.sv" >> $(FILELIST)

clean:
	$(MAKE) -C ./difftest clean
	rm -rf ./build

init:
	git submodule update --init
	cd rocket-chip && git submodule update --init api-config-chipsalliance hardfloat

bump:
	git submodule foreach "git fetch origin&&git checkout master&&git reset --hard origin/master"

bsp:
	mill -i mill.bsp.BSP/install

idea:
	mill -i mill.scalalib.GenIdea/idea

# verilator simulation
emu:
	$(MAKE) -C ./difftest emu SIM_TOP=SimTop DESIGN_DIR=$(NOOP_HOME) NUM_CORES=$(NUM_CORES)

emu_rtl:
	$(MAKE) -C ./difftest emu_rtl SIM_TOP=SimTop DESIGN_DIR=$(NOOP_HOME) NUM_CORES=$(NUM_CORES) 

EMU_RUN_OPTS_EXTRA ?=
EMU_RUN_OPTS = -i $(RUN_BIN_DIR)/$(RUN_BIN).bin
EMU_RUN_OPTS += --diff $(ABS_WORK_DIR)/ready-to-run/riscv64-nemu-interpreter-so
EMU_RUN_OPTS += --wave-path $(ABS_WORK_DIR)/sim/emu/$(RUN_BIN)/tb_top.vcd
EMU_RUN_OPTS += $(EMU_RUN_OPTS_EXTRA)
emu_rtl-run:
	$(shell if [ ! -e $(ABS_WORK_DIR)/sim/emu/$(RUN_BIN) ];then mkdir -p $(ABS_WORK_DIR)/sim/emu/$(RUN_BIN); fi)
	touch sim/emu/$(RUN_BIN)/sim.log
	$(shell if [ -e $(ABS_WORK_DIR)/sim/emu/$(RUN_BIN)/emu ];then rm -f $(ABS_WORK_DIR)/sim/emu/$(RUN_BIN)/emu; fi)
	ln -s $(ABS_WORK_DIR)/sim/emu/comp/emu $(ABS_WORK_DIR)/sim/emu/$(RUN_BIN)/emu
	cd sim/emu/$(RUN_BIN) && (./emu $(EMU_RUN_OPTS) 2> assert.log | tee sim.log)

# vcs simulation
simv:
	$(MAKE) -C ./difftest simv_rtl SIM_TOP=SimTop DESIGN_DIR=$(NOOP_HOME) NUM_CORES=$(NUM_CORES)

simv_rtl:
	$(MAKE) -C ./difftest simv_rtl SIM_TOP=SimTop DESIGN_DIR=$(NOOP_HOME) NUM_CORES=$(NUM_CORES) CONSIDER_FSDB=$(CONSIDER_FSDB) VCS=1

simv_rtl-run:
	$(shell if [ ! -e $(ABS_WORK_DIR)/sim/rtl/$(RUN_BIN) ];then mkdir -p $(ABS_WORK_DIR)/sim/rtl/$(RUN_BIN); fi)
	touch sim/rtl/$(RUN_BIN)/sim.log
	$(shell if [ -e $(ABS_WORK_DIR)/sim/rtl/$(RUN_BIN)/simv ];then rm -f $(ABS_WORK_DIR)/sim/rtl/$(RUN_BIN)/simv; fi)
	$(shell if [ -e $(ABS_WORK_DIR)/sim/rtl/$(RUN_BIN)/simv.daidir ];then rm -rf $(ABS_WORK_DIR)/sim/rtl/$(RUN_BIN)/simv.daidir; fi)
	ln -s $(ABS_WORK_DIR)/sim/rtl/comp/simv $(ABS_WORK_DIR)/sim/rtl/$(RUN_BIN)/simv
	ln -s $(ABS_WORK_DIR)/sim/rtl/comp/simv.daidir $(ABS_WORK_DIR)/sim/rtl/$(RUN_BIN)/simv.daidir
	cd sim/rtl/$(RUN_BIN) && (./simv $(RUN_OPTS) 2> assert.log | tee sim.log)

verdi_rtl:
	cd sim/rtl/$(RUN_BIN) && verdi -sv -2001 +verilog2001ext+v +systemverilogext+v -ssf tb_top.vf -dbdir simv.daidir -f sim_flist.f

.PHONY: verilog sim-verilog emu clean help init bump bsp $(REF_SO)

