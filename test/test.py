# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

    dut._log.info("Test project behavior")

    # Test the MAC (Multiply-Accumulate) Core
    # Weight = 3 (0b0011), Input = 2 (0b0010) => ui_in = 0x32
    dut.ui_in.value = 0x32
    dut.uio_in.value = 0

    # Wait for one clock cycle to calculate and accumulate
    await ClockCycles(dut.clk, 1)

    # 3 * 2 = 6. Accumulator starts at 0, so 0 + 6 = 6
    assert dut.uo_out.value == 6
    
    # Wait for another clock cycle
    await ClockCycles(dut.clk, 1)
    
    # Accumulator should now be 6 + 6 = 12
    assert dut.uo_out.value == 12

    # Keep testing the module by changing the input values, waiting for
    # one or more clock cycles, and asserting the expected output values.
