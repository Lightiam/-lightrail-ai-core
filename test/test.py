# SPDX-FileCopyrightText: (c) 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    # Single clock for the entire test - 10 us period (100 kHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # ----------------------------------------------------------------
    # Initial reset
    # ----------------------------------------------------------------
    dut._log.info("Reset")
    dut.ena.value    = 1
    dut.ui_in.value  = 0
    dut.uio_in.value = 0
    dut.rst_n.value  = 0
    await ClockCycles(dut.clk, 10)
    assert dut.uo_out.value  == 0, f"acc_a should be 0 in reset, got {dut.uo_out.value}"
    assert dut.uio_out.value == 0, f"acc_b should be 0 in reset, got {dut.uio_out.value}"
    dut.rst_n.value = 1

    # ----------------------------------------------------------------
    # Test 1 - Lane A MAC: weight=3, input=2, product=6
    # ui_in[7:4]=weight_a, ui_in[3:0]=input_a
    # ----------------------------------------------------------------
    dut._log.info("Test 1: Lane A MAC")
    dut.ui_in.value  = 0x32   # weight_a=3, input_a=2
    dut.uio_in.value = 0x00
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 6,  f"Expected acc_a=6,  got {dut.uo_out.value}"
    assert dut.uio_out.value == 0,  f"Expected acc_b=0,  got {dut.uio_out.value}"
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 12, f"Expected acc_a=12, got {dut.uo_out.value}"
    assert dut.uio_out.value == 0,  f"Expected acc_b=0,  got {dut.uio_out.value}"

    # ----------------------------------------------------------------
    # Test 2 - Lane B MAC: weight=5, input=3, product=15
    # uio_in[7:4]=weight_b, uio_in[3:0]=input_b
    # ----------------------------------------------------------------
    dut._log.info("Test 2: Lane B MAC")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    dut.ui_in.value  = 0x00
    dut.uio_in.value = 0x53   # weight_b=5, input_b=3
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 0,  f"Expected acc_a=0,  got {dut.uo_out.value}"
    assert dut.uio_out.value == 15, f"Expected acc_b=15, got {dut.uio_out.value}"
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 0,  f"Expected acc_a=0,  got {dut.uo_out.value}"
    assert dut.uio_out.value == 30, f"Expected acc_b=30, got {dut.uio_out.value}"

    # ----------------------------------------------------------------
    # Test 3 - SIMD parallel: LaneA 4*3=12, LaneB 2*7=14
    # ----------------------------------------------------------------
    dut._log.info("Test 3: SIMD parallel MAC")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    dut.ui_in.value  = 0x43   # weight_a=4, input_a=3
    dut.uio_in.value = 0x27   # weight_b=2, input_b=7
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 12, f"Expected acc_a=12, got {dut.uo_out.value}"
    assert dut.uio_out.value == 14, f"Expected acc_b=14, got {dut.uio_out.value}"
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 24, f"Expected acc_a=24, got {dut.uo_out.value}"
    assert dut.uio_out.value == 28, f"Expected acc_b=28, got {dut.uio_out.value}"

    # ----------------------------------------------------------------
    # Test 4 - 2-element dot product A=[3,5].W=[2,4] = 6+20 = 26
    # ----------------------------------------------------------------
    dut._log.info("Test 4: 2-element dot product")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    dut.ui_in.value  = 0x23   # Lane A: w=2, a=3 -> 6
    dut.uio_in.value = 0x45   # Lane B: w=4, a=5 -> 20
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 6,  f"Expected acc_a=6,  got {dut.uo_out.value}"
    assert dut.uio_out.value == 20, f"Expected acc_b=20, got {dut.uio_out.value}"
    dot = int(dut.uo_out.value) + int(dut.uio_out.value)
    assert dot == 26, f"Expected dot_product=26, got {dot}"

    # ----------------------------------------------------------------
    # Test 5 - 4-element dot product A=[1,2,3,4].W=[4,3,2,1] = 20
    # Cycle 1: LaneA=4*1=4, LaneB=3*2=6
    # Cycle 2: LaneA=2*3=6, LaneB=1*4=4  ->  acc_a=10, acc_b=10
    # ----------------------------------------------------------------
    dut._log.info("Test 5: 4-element dot product")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    dut.ui_in.value  = 0x41   # w0=4, a0=1 -> 4
    dut.uio_in.value = 0x32   # w1=3, a1=2 -> 6
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 4, f"Expected acc_a=4, got {dut.uo_out.value}"
    assert dut.uio_out.value == 6, f"Expected acc_b=6, got {dut.uio_out.value}"
    dut.ui_in.value  = 0x23   # w2=2, a2=3 -> 6
    dut.uio_in.value = 0x14   # w3=1, a3=4 -> 4
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 10, f"Expected acc_a=10, got {dut.uo_out.value}"
    assert dut.uio_out.value == 10, f"Expected acc_b=10, got {dut.uio_out.value}"
    dot = int(dut.uo_out.value) + int(dut.uio_out.value)
    assert dot == 20, f"Expected dot_product=20, got {dot}"

    # ----------------------------------------------------------------
    # Test 6 - Overflow wrap: 15*15=225; 225+225=450 -> mod256=194
    # ----------------------------------------------------------------
    dut._log.info("Test 6: Accumulator overflow wrap")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    dut.ui_in.value  = 0xFF   # weight_a=15, input_a=15 -> 225
    dut.uio_in.value = 0x00
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 225, f"Expected 225, got {dut.uo_out.value}"
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 194, f"Expected 194 (wrap), got {dut.uo_out.value}"

    # ----------------------------------------------------------------
    # Test 7 - ena gate: accumulation must freeze when ena=0
    # ----------------------------------------------------------------
    dut._log.info("Test 7: ena gate")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    dut.ui_in.value  = 0x34   # weight_a=3, input_a=4 -> 12
    dut.uio_in.value = 0x00
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 12, f"Expected acc_a=12, got {dut.uo_out.value}"
    dut.ena.value = 0
    await ClockCycles(dut.clk, 5)
    assert dut.uo_out.value == 12, f"Expected frozen acc_a=12, got {dut.uo_out.value}"
    dut.ena.value = 1
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 24, f"Expected acc_a=24 after re-enable, got {dut.uo_out.value}"

    dut._log.info("All tests PASSED")
