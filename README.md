![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)

# LightRail AI Gen3 – LR-P8A Photonic Inference Core

A Tiny Tapeout ASIC tile that forms the digital post-processor of the
**LightRail AI Gen3 LR-P8A** photonic accelerator PCIe card.

- [Read the full project datasheet](docs/info.md)

## What it does

Photonic AI accelerators perform matrix-vector multiplication at the speed of
light using Mach-Zehnder Interferometer (MZI) meshes.  After the optical
computation, photodetectors convert results back to electrical signals and an
ADC digitises them.  This chip receives those 4-bit ADC readings and:

1. **Multiplies** each reading by a signed 4-bit MZI weight (−8 to +7).
2. **Accumulates** the product into one of four independent 8-bit signed
   channels (A, B, C, D) — mirroring the four fiber-optic input bundles on
   the LR-P8A card.
3. **Saturates** at ±127/−128 instead of wrapping, matching the bounded
   optical power of a real photonic system.
4. **Applies ReLU** (`max(0, acc)`) on demand for neural-network activation.

## Pin summary

| Pins | Signal | Description |
|---|---|---|
| `ui[3:0]` | `adc_in` | Photodetector ADC result (unsigned 0–15) |
| `ui[7:4]` | `weight` | MZI weight (signed −8…+7, 2's-complement) |
| `uio[1:0]` | `ch_sel` | Channel select: 0=A 1=B 2=C 3=D |
| `uio[2]` | `relu_en` | 1 = apply ReLU to output |
| `uio[3]` | `ch_clear` | 1 = zero selected channel this cycle |
| `uo[7:0]` | `ch_out` | Selected accumulator (post-ReLU if enabled) |
| `uio[7:4]` | `neg_d…neg_a` | Sign flags – high when channel is negative |

## What is Tiny Tapeout?

Tiny Tapeout is an educational project that makes it easier and cheaper than
ever to get your digital designs manufactured on a real chip.
Visit https://tinytapeout.com to learn more.

## Resources

- [FAQ](https://tinytapeout.com/faq/)
- [Digital design lessons](https://tinytapeout.com/digital_design/)
- [Learn how semiconductors work](https://tinytapeout.com/siliwiz/)
- [Join the community](https://tinytapeout.com/discord)
- [Build your design locally](https://www.tinytapeout.com/guides/local-hardening/)
