# SPDX-FileCopyrightText: (c) 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

# LightRail SIMD Dual-Lane Tensor MAC Core - cocotb testbench
#
# Pin mapping:
#   ui_in[3:0]   = input_a  (Lane A activation, 4-bit)
#   ui_in[7:4]   = weight_a (Lane A weight,     4-bit)
#   uio_in[3:0]  = input_b  (Lane B activation, 4-bit)
#   uio_in[7:4]  = weight_b (Lane B weight,     4-bit)
#   uo_out[7:0]  = acc_a    (Lane A accumulator, 8-bit)
#   uio_out[7:0] = acc_b    (Lane B accumulator, 8-bit)
#
# All 8 test scenarios run inside ONE @cocotb.test() so the clock is
# started exactly once and there are no zombie clock coroutines.

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


def pack(weight, activation):
    """Pack a 4-bit weight (upper nibble) and 4-bit activation (lower nibble)."""
    return ((weight & 0xF) << 4) | (activation & 0xF)


async def do_reset(dut):
    """Assert rst_n low for 5 cycles then release; leave ena=1."""
    dut.ena.value    = 1
    dut.ui_in.value  = 0
    dut.uio_in.value = 0
    dut.rst_n.value  = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value  = 1


@cocotb.test()
async def test_tensor_mac_core(dut):
    """LightRail SIMD Tensor MAC Core - full verification suite (8 scenarios)."""
    dut._log.info("Start: LightRail SIMD Tensor MAC Core")

    # Start a single clock for the entire test suite (10 us period = 100 kHz).
    # Well within the 50 MHz synthesis target; avoids multi-clock coroutine bugs.
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())

    # -------------------------------------------------------------------
    # Scenario 1: Reset clears both accumulators
    # -------------------------------------------------------------------
    dut._log.info("[1] Reset")
    dut.ena.value    = 1
    dut.ui_in.value  = pack(15, 15)   # max inputs - must not appear in output
    dut.uio_in.value = pack(15, 15)
    dut.rst_n.value  = 0
    await ClockCycles(dut.clk, 5)
    assert int(dut.uo_out.value)  == 0, \
        f"[1] acc_a should be 0 during reset, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 0, \
        f"[1] acc_b should be 0 during reset, got {int(dut.uio_out.value)}"
    dut._log.info("[1] Reset PASS")

    # -------------------------------------------------------------------
    # Scenario 2: Lane A isolated MAC  (weight=3, input=2 -> 6)
    # -------------------------------------------------------------------
    dut._log.info("[2] Lane A isolated MAC")
    await do_reset(dut)
    dut.ui_in.value  = pack(3, 2)   # weight_a=3, input_a=2 -> product=6
    dut.uio_in.value = pack(0, 0)   # Lane B idle

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 6,  \
        f"[2] Cycle1: expected acc_a=6,  got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 0,  \
        f"[2] Cycle1: expected acc_b=0,  got {int(dut.uio_out.value)}"

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 12, \
        f"[2] Cycle2: expected acc_a=12, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 0,  \
        f"[2] Cycle2: expected acc_b=0,  got {int(dut.uio_out.value)}"
    dut._log.info("[2] Lane A isolated MAC PASS")

    # -------------------------------------------------------------------
    # Scenario 3: Lane B isolated MAC  (weight=5, input=3 -> 15)
    # -------------------------------------------------------------------
    dut._log.info("[3] Lane B isolated MAC")
    await do_reset(dut)
    dut.ui_in.value  = pack(0, 0)   # Lane A idle
    dut.uio_in.value = pack(5, 3)   # weight_b=5, input_b=3 -> product=15

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 0,  \
        f"[3] Cycle1: expected acc_a=0,  got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 15, \
        f"[3] Cycle1: expected acc_b=15, got {int(dut.uio_out.value)}"

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 0,  \
        f"[3] Cycle2: expected acc_a=0,  got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 30, \
        f"[3] Cycle2: expected acc_b=30, got {int(dut.uio_out.value)}"
    dut._log.info("[3] Lane B isolated MAC PASS")

    # -------------------------------------------------------------------
    # Scenario 4: SIMD parallel MAC (both lanes fire simultaneously)
    #   Lane A: 4*3=12,  Lane B: 2*7=14
    # -------------------------------------------------------------------
    dut._log.info("[4] SIMD parallel MAC")
    await do_reset(dut)
    dut.ui_in.value  = pack(4, 3)   # weight_a=4, input_a=3 -> 12
    dut.uio_in.value = pack(2, 7)   # weight_b=2, input_b=7 -> 14

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 12, \
        f"[4] Cycle1: expected acc_a=12, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 14, \
        f"[4] Cycle1: expected acc_b=14, got {int(dut.uio_out.value)}"

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 24, \
        f"[4] Cycle2: expected acc_a=24, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 28, \
        f"[4] Cycle2: expected acc_b=28, got {int(dut.uio_out.value)}"

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 36, \
        f"[4] Cycle3: expected acc_a=36, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 42, \
        f"[4] Cycle3: expected acc_b=42, got {int(dut.uio_out.value)}"
    dut._log.info("[4] SIMD parallel MAC PASS")

    # -------------------------------------------------------------------
    # Scenario 5: 2-element dot product (single SIMD cycle)
    #   A=[3,5], W=[2,4]:  dp = 3*2 + 5*4 = 6 + 20 = 26
    # -------------------------------------------------------------------
    dut._log.info("[5] 2-element dot product")
    await do_reset(dut)
    dut.ui_in.value  = pack(2, 3)   # Lane A: w=2, a=3 -> 6
    dut.uio_in.value = pack(4, 5)   # Lane B: w=4, a=5 -> 20
    await ClockCycles(dut.clk, 1)

    acc_a = int(dut.uo_out.value)
    acc_b = int(dut.uio_out.value)
    assert acc_a == 6,            f"[5] Expected acc_a=6,   got {acc_a}"
    assert acc_b == 20,           f"[5] Expected acc_b=20,  got {acc_b}"
    assert acc_a + acc_b == 26,   f"[5] Expected dot=26,    got {acc_a + acc_b}"
    dut._log.info(f"[5] 2-element dot product={acc_a + acc_b} PASS")

    # -------------------------------------------------------------------
    # Scenario 6: 4-element dot product (two SIMD cycles)
    #   A=[1,2,3,4], W=[4,3,2,1]:  dp = 4+6+6+4 = 20
    # -------------------------------------------------------------------
    dut._log.info("[6] 4-element dot product")
    await do_reset(dut)

    # Cycle 1: dimensions 0 and 1
    dut.ui_in.value  = pack(4, 1)   # w0=4, a0=1 -> 4
    dut.uio_in.value = pack(3, 2)   # w1=3, a1=2 -> 6
    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 4, \
        f"[6] Cycle1: expected acc_a=4, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 6, \
        f"[6] Cycle1: expected acc_b=6, got {int(dut.uio_out.value)}"

    # Cycle 2: dimensions 2 and 3
    dut.ui_in.value  = pack(2, 3)   # w2=2, a2=3 -> 6
    dut.uio_in.value = pack(1, 4)   # w3=1, a3=4 -> 4
    await ClockCycles(dut.clk, 1)

    acc_a = int(dut.uo_out.value)
    acc_b = int(dut.uio_out.value)
    assert acc_a == 10,           f"[6] Expected acc_a=10 (4+6), got {acc_a}"
    assert acc_b == 10,           f"[6] Expected acc_b=10 (6+4), got {acc_b}"
    assert acc_a + acc_b == 20,   f"[6] Expected dot=20,         got {acc_a + acc_b}"
    dut._log.info(f"[6] 4-element dot product={acc_a + acc_b} PASS")

    # -------------------------------------------------------------------
    # Scenario 7: Accumulator overflow (mod-256 wrap)
    #   15*15=225; 225+225=450 -> 450 mod 256 = 194
    # -------------------------------------------------------------------
    dut._log.info("[7] Accumulator overflow wrap")
    await do_reset(dut)
    dut.ui_in.value  = pack(15, 15)  # max 4-bit: 15*15=225
    dut.uio_in.value = pack(0, 0)

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value) == 225, \
        f"[7] Cycle1: expected acc_a=225, got {int(dut.uo_out.value)}"

    await ClockCycles(dut.clk, 1)
    expected_wrap = (225 * 2) & 0xFF   # = 194
    assert int(dut.uo_out.value) == expected_wrap, \
        f"[7] Cycle2: expected acc_a={expected_wrap} (wrap), got {int(dut.uo_out.value)}"
    dut._log.info(f"[7] overflow wrap={expected_wrap} PASS")

    # -------------------------------------------------------------------
    # Scenario 8: ena gate - accumulation must freeze when ena=0
    # -------------------------------------------------------------------
    dut._log.info("[8] ena gate")
    await do_reset(dut)
    dut.ui_in.value  = pack(3, 4)   # 3*4=12
    dut.uio_in.value = pack(0, 0)

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value) == 12, \
        f"[8] Setup: expected acc_a=12, got {int(dut.uo_out.value)}"

    dut.ena.value = 0               # disable - accumulator must freeze
    await ClockCycles(dut.clk, 5)
    assert int(dut.uo_out.value) == 12, \
        f"[8] ena=0: expected frozen acc_a=12, got {int(dut.uo_out.value)}"

    dut.ena.value = 1               # re-enable - must resume accumulating
    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value) == 24, \
        f"[8] Re-enable: expected acc_a=24, got {int(dut.uo_out.value)}"
    dut._log.info("[8] ena gate PASS")

    dut._log.info("All 8 scenarios PASSED - LightRail SIMD Tensor MAC Core verified")
