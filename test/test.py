# SPDX-FileCopyrightText: (c) 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut._log.info("Reset")
    dut.ena.value    = 1
    dut.ui_in.value  = 0
    dut.uio_in.value = 0
    dut.rst_n.value  = 0
    await ClockCycles(dut.clk, 10)
    assert dut.uo_out.value  == 0
    assert dut.uio_out.value == 0
    dut.rst_n.value = 1

    # Test 1 - Lane A: weight=3, input=2 -> 6, then 12
    dut._log.info("Test 1: Lane A MAC")
    dut.ui_in.value  = 0x32
    dut.uio_in.value = 0x00
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 6
    assert dut.uio_out.value == 0
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 12
    assert dut.uio_out.value == 0

    # Test 2 - Lane B: weight=5, input=3 -> 15, then 30
    dut._log.info("Test 2: Lane B MAC")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value  = 1
    dut.ui_in.value  = 0x00
    dut.uio_in.value = 0x53
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 0
    assert dut.uio_out.value == 15
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 0
    assert dut.uio_out.value == 30

    # Test 3 - SIMD: LaneA 4*3=12, LaneB 2*7=14
    dut._log.info("Test 3: SIMD parallel MAC")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value  = 1
    dut.ui_in.value  = 0x43
    dut.uio_in.value = 0x27
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 12
    assert dut.uio_out.value == 14
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 24
    assert dut.uio_out.value == 28

    # Test 4 - 2-element dot product: [3,5].[2,4] = 6+20 = 26
    dut._log.info("Test 4: 2-element dot product")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value  = 1
    dut.ui_in.value  = 0x23
    dut.uio_in.value = 0x45
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 6
    assert dut.uio_out.value == 20
    assert int(dut.uo_out.value) + int(dut.uio_out.value) == 26

    # Test 5 - 4-element dot product: [1,2,3,4].[4,3,2,1] = 20
    dut._log.info("Test 5: 4-element dot product")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value  = 1
    dut.ui_in.value  = 0x41
    dut.uio_in.value = 0x32
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 4
    assert dut.uio_out.value == 6
    dut.ui_in.value  = 0x23
    dut.uio_in.value = 0x14
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value  == 10
    assert dut.uio_out.value == 10
    assert int(dut.uo_out.value) + int(dut.uio_out.value) == 20

    # Test 6 - Overflow: 15*15=225, 225+225 mod 256 = 194
    dut._log.info("Test 6: Overflow wrap")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value  = 1
    dut.ui_in.value  = 0xFF
    dut.uio_in.value = 0x00
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 225
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 194

    # Test 7 - ena gate
    dut._log.info("Test 7: ena gate")
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value  = 1
    dut.ui_in.value  = 0x34
    dut.uio_in.value = 0x00
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 12
    dut.ena.value = 0
    await ClockCycles(dut.clk, 5)
    assert dut.uo_out.value == 12
    dut.ena.value = 1
    await ClockCycles(dut.clk, 1)
    assert dut.uo_out.value == 24

    dut._log.info("All tests PASSED")
