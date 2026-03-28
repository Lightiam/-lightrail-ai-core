<!---
This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

The **LightRail AI Gen3 LR-P8A** is the digital post-processor tile for a photonic matrix-vector
multiplication (MVM) accelerator.  In a photonic AI chip, a Mach-Zehnder Interferometer (MZI)
mesh performs weighted summation of optical signals at the speed of light.  Photodetectors convert
each optical output to an electrical current, and an ADC digitises it.  This tile receives those
ADC readings and completes the inference step electronically.

### Architecture

Four independent inference channels **A, B, C, D** (corresponding to the four fiber-optic input
bundles on the LR-P8A PCIe card) each contain:

- A **signed 4-bit MZI weight** (−8 to +7, 2's-complement) representing the programmed optical
  transmittance coefficient of the waveguide.
- An **unsigned 4-bit ADC input** (0–15) from the photodetector.
- A signed multiply that produces the **optical product** (−120 to +105, fits in 8 bits signed).
- A **saturating 8-bit signed accumulator** that clamps to ±127/−128 instead of wrapping,
  matching the bounded optical power envelope of real photonic systems.
- An optional **ReLU nonlinearity** (`uo_out = max(0, acc)`) for neural-network activation.

Only one channel is active per clock cycle (selected by `ch_sel`), so four dot-product partial
sums can be built up over consecutive cycles without disturbing the other channels.

### Pin map

| Signal | Pins | Direction | Description |
|---|---|---|---|
| `adc_in[3:0]` | `ui[3:0]` | in | Photodetector ADC result (unsigned) |
| `weight[3:0]` | `ui[7:4]` | in | MZI weight (signed 2's-complement) |
| `ch_sel[1:0]` | `uio[1:0]` | in | Channel select: 0=A 1=B 2=C 3=D |
| `relu_en` | `uio[2]` | in | 1 = apply ReLU to output |
| `ch_clear` | `uio[3]` | in | 1 = zero selected channel this cycle |
| `ch_out[7:0]` | `uo[7:0]` | out | Selected accumulator (post-ReLU if enabled) |
| `neg_a…neg_d` | `uio[4:7]` | out | Sign flags (1 = accumulator is negative) |

Saturation flags `{sat_d, sat_c, sat_b, sat_a}` appear on `uio_out[3:0]` and are observable
in simulation; they indicate which channel accumulators are clamped at ±127/−128.

### Arithmetic details

```
product  = signed(weight)  ×  unsigned(adc_in)     // 9-bit signed, range −120…+105
sum      = acc[ch_sel]  +  product                  // 9-bit signed intermediate
sat      = clamp(sum, −128, +127)                   // saturating result
acc[ch_sel] ← ch_clear ? 0 : sat                   // written on posedge clk when ena=1
uo_out  = relu_en ? max(0, acc[ch_sel]) : acc[ch_sel]
```

## How to test

Apply reset (`rst_n=0`) for several cycles, then release (`rst_n=1`).

**Basic accumulation (channel A, weight=3, ADC=4):**

```
ui_in  = 0x34   (weight=3 in bits[7:4], adc_in=4 in bits[3:0])
uio_in = 0x00   (ch_sel=A, relu=0, clear=0)
```

After each clock: `uo_out` increments by 12 (3 × 4).

**Negative weight (channel C, weight=−2, ADC=5):**

```
ui_in  = 0xE5   (weight=0xE = −2 in signed 4-bit, adc_in=5)
uio_in = 0x02   (ch_sel=C)
```

After each clock: `uo_out` decrements by 10.  When it reaches −128 it saturates.

**ReLU:**

Set `uio_in[2]=1`.  Negative accumulator values appear as 0 on `uo_out`; positive values pass
through unchanged.

**Per-channel clear:**

Set `uio_in[3]=1` for one cycle to zero the currently selected channel without affecting others.

**Multi-channel inference (neural-network layer):**

Stream weight/ADC pairs across channels A–D in rotation to accumulate a 4-element partial dot
product on each channel simultaneously, then read all four results sequentially.

## External hardware

No external hardware required for simulation or FPGA testing.

For integration with a photonic co-processor: connect each `adc_in[3:0]` to a 4-bit ADC
sampling the corresponding photodetector output, and drive `weight[3:0]` from a DAC register
loaded with the MZI phase coefficients for the selected channel.
