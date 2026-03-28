# SPDX-FileCopyrightText: (c) 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

# LightRail AI Gen3 – LR-P8A Photonic Inference Core – cocotb test suite

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer


def s8(v):
    """Interpret an unsigned 8-bit cocotb value as a Python signed integer."""
    v = int(v)
    return v - 256 if v >= 128 else v


def set_inputs(dut, adc=0, weight=0, ch=0, relu=0, clr=0):
    """Drive ui_in and uio_in from named photonic-core fields."""
    w4 = weight & 0xF                          # 4-bit 2's-complement weight
    dut.ui_in.value  = (w4 << 4) | (adc & 0xF)
    dut.uio_in.value = (ch & 0x3) | ((relu & 1) << 2) | ((clr & 1) << 3)


async def reset(dut):
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    await Timer(1, units="ns")
    dut.rst_n.value = 1


@cocotb.test()
async def test_project(dut):
    dut._log.info("LightRail AI Gen3 – LR-P8A Photonic Inference Core")

    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    # ── Reset ────────────────────────────────────────────────────────────────
    dut._log.info("Reset: all accumulators must clear to 0")
    dut.ena.value = 1
    set_inputs(dut)
    await reset(dut)
    assert dut.uo_out.value  == 0, "uo_out after reset"
    assert dut.uio_out.value == 0, "uio_out after reset"

    # ── Test 1: CH-A positive weight accumulation ────────────────────────────
    dut._log.info("Test 1: CH-A  weight=+3, adc=4  →  product=12 per cycle")
    set_inputs(dut, adc=4, weight=3, ch=0)
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 12, f"cycle 1: {int(dut.uo_out.value)}"
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 24, f"cycle 2: {int(dut.uo_out.value)}"
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 36, f"cycle 3: {int(dut.uo_out.value)}"

    # ── Test 2: Channel independence ─────────────────────────────────────────
    dut._log.info("Test 2: CH-B  weight=+2, adc=5  →  CH-A must stay frozen at 36")
    set_inputs(dut, adc=5, weight=2, ch=1)
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    # CH-B = 10; read CH-A combinatorially
    set_inputs(dut, adc=0, weight=0, ch=0)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 36, f"CH-A frozen: {int(dut.uo_out.value)}"
    # Read CH-B
    set_inputs(dut, adc=5, weight=2, ch=1)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 10, f"CH-B after 1 cycle: {int(dut.uo_out.value)}"

    # ── Test 3: Negative weight ──────────────────────────────────────────────
    dut._log.info("Test 3: CH-C  weight=−2, adc=5  →  product=−10 per cycle")
    set_inputs(dut, adc=0, weight=0, ch=2, clr=1)
    await ClockCycles(dut.clk, 1)        # clear CH-C
    await Timer(1, units="ns")
    set_inputs(dut, adc=5, weight=-2, ch=2)
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    assert s8(dut.uo_out.value) == -10, f"CH-C cycle 1: {s8(dut.uo_out.value)}"
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    assert s8(dut.uo_out.value) == -20, f"CH-C cycle 2: {s8(dut.uo_out.value)}"

    # ── Test 4: ReLU activation ──────────────────────────────────────────────
    dut._log.info("Test 4: ReLU – negative acc → 0, positive acc passes through")
    # CH-C = -20, relu_en=1 → uo_out must be 0
    set_inputs(dut, adc=0, weight=0, ch=2, relu=1)
    await ClockCycles(dut.clk, 1)        # product=0, acc stays -20
    await Timer(1, units="ns")
    assert dut.uo_out.value == 0, f"ReLU of −20: {int(dut.uo_out.value)}"
    # CH-A = 36 (positive), relu_en=1 → must pass through as 36
    set_inputs(dut, adc=0, weight=0, ch=0, relu=1)
    await Timer(1, units="ns")           # combinatorial read only
    assert dut.uo_out.value == 36, f"ReLU of +36: {int(dut.uo_out.value)}"

    # ── Test 5: Positive saturation ──────────────────────────────────────────
    dut._log.info("Test 5: CH-D  weight=+7, adc=15  →  clamp at +127")
    set_inputs(dut, adc=0, weight=0, ch=3, clr=1)
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    set_inputs(dut, adc=15, weight=7, ch=3)   # product = 105
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 105, f"CH-D cycle 1: {int(dut.uo_out.value)}"
    await ClockCycles(dut.clk, 1)             # 105 + 105 = 210 > 127 → 127
    await Timer(1, units="ns")
    assert dut.uo_out.value == 127, f"CH-D saturated: {int(dut.uo_out.value)}"
    assert (int(dut.uio_out.value) & 0x08) != 0, "sat_d flag must be set"

    # ── Test 6: Negative saturation ──────────────────────────────────────────
    dut._log.info("Test 6: CH-D  weight=−8, adc=15  →  clamp at −128")
    set_inputs(dut, adc=0, weight=0, ch=3, clr=1)
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    set_inputs(dut, adc=15, weight=-8, ch=3)  # product = -120
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    assert s8(dut.uo_out.value) == -120, f"CH-D cycle 1: {s8(dut.uo_out.value)}"
    await ClockCycles(dut.clk, 1)             # -120 + (-120) = -240 < -128 → -128
    await Timer(1, units="ns")
    assert s8(dut.uo_out.value) == -128, f"CH-D saturated: {s8(dut.uo_out.value)}"
    uio = int(dut.uio_out.value)
    assert (uio & 0x08) != 0, "sat_d flag must be set"
    assert (uio & 0x80) != 0, "neg_d flag must be set"

    # ── Test 7: ch_clear zeroes one channel, leaves others intact ────────────
    dut._log.info("Test 7: ch_clear – zero CH-A, CH-B must be unaffected")
    # CH-A=36, CH-B=10 from earlier tests
    set_inputs(dut, adc=0, weight=0, ch=0, clr=1)
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    set_inputs(dut, adc=0, weight=0, ch=0)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 0,  f"CH-A after clear: {int(dut.uo_out.value)}"
    set_inputs(dut, adc=0, weight=0, ch=1)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 10, f"CH-B untouched: {int(dut.uo_out.value)}"

    # ── Test 8: ena gate freezes accumulators ────────────────────────────────
    dut._log.info("Test 8: ena=0 must freeze all channels")
    set_inputs(dut, adc=5, weight=1, ch=1)   # would add 5 to CH-B
    dut.ena.value = 0
    await ClockCycles(dut.clk, 3)
    await Timer(1, units="ns")
    assert dut.uo_out.value == 10, f"CH-B frozen: {int(dut.uo_out.value)}"
    dut.ena.value = 1
    await ClockCycles(dut.clk, 1)            # 10 + 5 = 15
    await Timer(1, units="ns")
    assert dut.uo_out.value == 15, f"CH-B after resume: {int(dut.uo_out.value)}"

    # ── Test 9: sign flags in uio_out[7:4] ──────────────────────────────────
    dut._log.info("Test 9: neg_a flag – acc[A] goes negative → uio_out[4] set")
    await reset(dut)
    set_inputs(dut, adc=1, weight=-1, ch=0)  # product = -1, acc[A] → -1
    await ClockCycles(dut.clk, 1)
    await Timer(1, units="ns")
    uio = int(dut.uio_out.value)
    assert (uio & 0x10) != 0, f"neg_a should be set, uio_out=0x{uio:02X}"
    assert (uio & 0x20) == 0, f"neg_b should not be set, uio_out=0x{uio:02X}"

    # ── Test 10: 4-element photonic dot product across all channels ───────────
    dut._log.info("Test 10: 4-channel photonic dot product")
    # [w_A,w_B,w_C,w_D] = [3,2,4,1]  ×  [adc_A,adc_B,adc_C,adc_D] = [4,5,3,6]
    # Expected results: A=12, B=10, C=12, D=6
    await reset(dut)
    pairs = [(0, 3, 4), (1, 2, 5), (2, 4, 3), (3, 1, 6)]
    for ch, w, a in pairs:
        set_inputs(dut, adc=a, weight=w, ch=ch)
        await ClockCycles(dut.clk, 1)
        await Timer(1, units="ns")
    expected = {0: 12, 1: 10, 2: 12, 3: 6}
    for ch, exp in expected.items():
        set_inputs(dut, adc=0, weight=0, ch=ch)
        await Timer(1, units="ns")
        assert dut.uo_out.value == exp, f"CH-{chr(65+ch)} dot product: expected {exp}, got {int(dut.uo_out.value)}"

    dut._log.info("All tests PASSED – LightRail AI Gen3 LR-P8A core verified")
