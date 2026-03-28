# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

"""
cocotb testbench for tt_um_lightrail_ai_core
============================================
SIMD Dual-Lane Tensor MAC Core verification.

Pin mapping under test
----------------------
  ui_in[3:0]  = input_a   (Lane A activation, 4-bit)
  ui_in[7:4]  = weight_a  (Lane A weight,     4-bit)
  uio_in[3:0] = input_b   (Lane B activation, 4-bit)
  uio_in[7:4] = weight_b  (Lane B weight,     4-bit)
  uo_out[7:0] = acc_a     (Lane A accumulator, 8-bit)
  uio_out[7:0]= acc_b     (Lane B accumulator, 8-bit)

Test suite
----------
  1. test_reset                 - Both accumulators cleared on rst_n
  2. test_single_mac_lane_a     - Lane A isolated MAC and accumulation
  3. test_single_mac_lane_b     - Lane B isolated MAC via uio_in
  4. test_simd_parallel_mac     - Both lanes fire simultaneously (SIMD)
  5. test_dot_product_2element  - 2-element dot product in 1 cycle
  6. test_dot_product_4element  - 4-element dot product in 2 cycles
  7. test_accumulator_overflow  - 8-bit wrap-around (mod-256) behaviour
  8. test_ena_gate              - ena=0 freezes accumulators
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


def pack_lane(weight: int, activation: int) -> int:
    """Pack a 4-bit weight and 4-bit activation into an 8-bit bus byte.

    Bit layout: [7:4] = weight (upper nibble), [3:0] = activation (lower nibble)
    """
    return ((weight & 0xF) << 4) | (activation & 0xF)


async def _reset(dut, cycles: int = 5) -> None:
    """Drive rst_n low for *cycles* clocks, then release."""
    dut.ena.value    = 1
    dut.ui_in.value  = 0
    dut.uio_in.value = 0
    dut.rst_n.value  = 0
    await ClockCycles(dut.clk, cycles)
    dut.rst_n.value  = 1


# ---------------------------------------------------------------------------
# Test 1 — Reset
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_reset(dut):
    """Both accumulators must be zero immediately after reset."""
    dut._log.info("test_reset: start")
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())

    # Apply non-zero inputs *before* releasing reset to ensure reset wins
    dut.ena.value    = 1
    dut.ui_in.value  = pack_lane(15, 15)
    dut.uio_in.value = pack_lane(15, 15)
    dut.rst_n.value  = 0
    await ClockCycles(dut.clk, 5)

    assert int(dut.uo_out.value)  == 0, f"acc_a should be 0 in reset, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 0, f"acc_b should be 0 in reset, got {int(dut.uio_out.value)}"
    dut._log.info("test_reset: PASS")


# ---------------------------------------------------------------------------
# Test 2 — Lane A isolated MAC
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_single_mac_lane_a(dut):
    """Lane A: weight=3, input=2 → product=6; accumulates 6 each cycle."""
    dut._log.info("test_single_mac_lane_a: start")
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())
    await _reset(dut)

    # Lane A: weight=3, input=2 → product=6.  Lane B idle.
    dut.ui_in.value  = pack_lane(3, 2)   # 0x32
    dut.uio_in.value = pack_lane(0, 0)   # 0x00

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 6,  f"Cycle 1: expected acc_a=6,  got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 0,  f"Cycle 1: expected acc_b=0,  got {int(dut.uio_out.value)}"

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 12, f"Cycle 2: expected acc_a=12, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 0,  f"Cycle 2: expected acc_b=0,  got {int(dut.uio_out.value)}"
    dut._log.info("test_single_mac_lane_a: PASS")


# ---------------------------------------------------------------------------
# Test 3 — Lane B isolated MAC
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_single_mac_lane_b(dut):
    """Lane B: weight=5, input=3 → product=15; Lane A idle."""
    dut._log.info("test_single_mac_lane_b: start")
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())
    await _reset(dut)

    dut.ui_in.value  = pack_lane(0, 0)   # Lane A idle
    dut.uio_in.value = pack_lane(5, 3)   # weight_b=5, input_b=3 → 15

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 0,  f"Expected acc_a=0,  got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 15, f"Expected acc_b=15, got {int(dut.uio_out.value)}"

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 0,  f"Expected acc_a=0,  got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 30, f"Expected acc_b=30, got {int(dut.uio_out.value)}"
    dut._log.info("test_single_mac_lane_b: PASS")


# ---------------------------------------------------------------------------
# Test 4 — SIMD parallel MAC (both lanes fire simultaneously)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_simd_parallel_mac(dut):
    """Both lanes accumulate independently and in parallel each cycle."""
    dut._log.info("test_simd_parallel_mac: start")
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())
    await _reset(dut)

    # Lane A: weight=4, input=3 → product=12
    # Lane B: weight=2, input=7 → product=14
    dut.ui_in.value  = pack_lane(4, 3)
    dut.uio_in.value = pack_lane(2, 7)

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 12, f"Cycle 1: expected acc_a=12, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 14, f"Cycle 1: expected acc_b=14, got {int(dut.uio_out.value)}"

    # Second cycle with same inputs: acc_a=24, acc_b=28
    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 24, f"Cycle 2: expected acc_a=24, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 28, f"Cycle 2: expected acc_b=28, got {int(dut.uio_out.value)}"

    # Third cycle: acc_a=36, acc_b=42
    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value)  == 36, f"Cycle 3: expected acc_a=36, got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 42, f"Cycle 3: expected acc_b=42, got {int(dut.uio_out.value)}"
    dut._log.info("test_simd_parallel_mac: PASS")


# ---------------------------------------------------------------------------
# Test 5 — 2-element dot product (single cycle)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_dot_product_2element(dut):
    """
    Dot product of 2-element vectors A=[3,5] and W=[2,4]:
      dp = 3*2 + 5*4 = 6 + 20 = 26

    Both dimensions fit in one SIMD cycle:
      Lane A: w=2, a=3 → acc_a = 6
      Lane B: w=4, a=5 → acc_b = 20
      dot_product = acc_a + acc_b = 26
    """
    dut._log.info("test_dot_product_2element: start")
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())
    await _reset(dut)

    dut.ui_in.value  = pack_lane(2, 3)   # Lane A: w=2, a=3
    dut.uio_in.value = pack_lane(4, 5)   # Lane B: w=4, a=5
    await ClockCycles(dut.clk, 1)

    acc_a = int(dut.uo_out.value)
    acc_b = int(dut.uio_out.value)
    dot   = acc_a + acc_b

    assert acc_a == 6,  f"Expected acc_a=6,  got {acc_a}"
    assert acc_b == 20, f"Expected acc_b=20, got {acc_b}"
    assert dot   == 26, f"Expected dot=26,   got {dot}"
    dut._log.info(f"test_dot_product_2element: acc_a={acc_a}, acc_b={acc_b}, dot={dot} — PASS")


# ---------------------------------------------------------------------------
# Test 6 — 4-element dot product (two cycles)
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_dot_product_4element(dut):
    """
    Dot product of 4-element vectors A=[1,2,3,4] and W=[4,3,2,1]:
      dp = 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20

    2-cycle SIMD schedule:
      Cycle 1 — Lane A: (w=4,a=1)=4   Lane B: (w=3,a=2)=6
      Cycle 2 — Lane A: (w=2,a=3)=6   Lane B: (w=1,a=4)=4
      Result:   acc_a = 4+6 = 10      acc_b = 6+4 = 10
      dot_product = 10 + 10 = 20
    """
    dut._log.info("test_dot_product_4element: start")
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())
    await _reset(dut)

    # Cycle 1: dimensions 0 and 1
    dut.ui_in.value  = pack_lane(4, 1)   # w0=4, a0=1 → 4
    dut.uio_in.value = pack_lane(3, 2)   # w1=3, a1=2 → 6
    await ClockCycles(dut.clk, 1)

    assert int(dut.uo_out.value)  == 4,  f"After cycle 1: expected acc_a=4,  got {int(dut.uo_out.value)}"
    assert int(dut.uio_out.value) == 6,  f"After cycle 1: expected acc_b=6,  got {int(dut.uio_out.value)}"

    # Cycle 2: dimensions 2 and 3
    dut.ui_in.value  = pack_lane(2, 3)   # w2=2, a2=3 → 6
    dut.uio_in.value = pack_lane(1, 4)   # w3=1, a3=4 → 4
    await ClockCycles(dut.clk, 1)

    acc_a = int(dut.uo_out.value)
    acc_b = int(dut.uio_out.value)
    dot   = acc_a + acc_b

    assert acc_a == 10, f"Expected acc_a=10 (4+6),  got {acc_a}"
    assert acc_b == 10, f"Expected acc_b=10 (6+4),  got {acc_b}"
    assert dot   == 20, f"Expected dot_product=20,  got {dot}"
    dut._log.info(f"test_dot_product_4element: acc_a={acc_a}, acc_b={acc_b}, dot={dot} — PASS")


# ---------------------------------------------------------------------------
# Test 7 — Accumulator overflow / wrap-around
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_accumulator_overflow(dut):
    """
    Verify 8-bit mod-256 wrap-around:
      weight=15, input=15 → product=225 each cycle.
      Cycle 1: acc_a = 225
      Cycle 2: acc_a = (225+225) mod 256 = 450 mod 256 = 194
    """
    dut._log.info("test_accumulator_overflow: start")
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())
    await _reset(dut)

    dut.ui_in.value  = pack_lane(15, 15)  # max 4-bit: 15*15=225
    dut.uio_in.value = pack_lane(0, 0)

    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value) == 225, \
        f"Cycle 1: expected acc_a=225, got {int(dut.uo_out.value)}"

    await ClockCycles(dut.clk, 1)
    expected = (225 * 2) & 0xFF   # 194
    assert int(dut.uo_out.value) == expected, \
        f"Cycle 2: expected acc_a={expected} (wrap), got {int(dut.uo_out.value)}"
    dut._log.info(f"test_accumulator_overflow: wrap={expected} — PASS")


# ---------------------------------------------------------------------------
# Test 8 — ena gate: accumulation freezes when ena=0
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_ena_gate(dut):
    """When ena=0, both accumulators must hold their current value."""
    dut._log.info("test_ena_gate: start")
    cocotb.start_soon(Clock(dut.clk, 10, units="us").start())
    await _reset(dut)

    # Accumulate one step on Lane A: 3*4=12
    dut.ui_in.value  = pack_lane(3, 4)
    dut.uio_in.value = pack_lane(0, 0)
    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value) == 12, f"Setup: expected acc_a=12, got {int(dut.uo_out.value)}"

    # Disable the core — accumulator must remain frozen for 5 cycles
    dut.ena.value = 0
    await ClockCycles(dut.clk, 5)
    assert int(dut.uo_out.value) == 12, \
        f"ena=0: accumulator should be frozen at 12, got {int(dut.uo_out.value)}"

    # Re-enable — must resume accumulating
    dut.ena.value = 1
    await ClockCycles(dut.clk, 1)
    assert int(dut.uo_out.value) == 24, \
        f"After re-enable: expected acc_a=24 (12+12), got {int(dut.uo_out.value)}"
    dut._log.info("test_ena_gate: PASS")
