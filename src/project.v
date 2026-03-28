`default_nettype none

// LightRail AI Gen3 – LR-P8A Photonic Inference Core
// Tiny Tapeout 1x1 tile · Sky130 PDK
//
// Digital post-processor for a photonic matrix-vector accelerator.
// Inspired by the LR-P8A PCIe card: two photonic dies connected by optical
// waveguides, four fiber-optic input bundles feeding four independent channels.
//
// Each channel accumulates  weight × ADC_reading  into a saturating 8-bit
// signed register.  Signed MZI phase coefficients (weights) allow both
// constructive and destructive optical interference.  A per-output ReLU
// nonlinearity supports neural-network inference directly on-chip.
//
// Photonic inference pipeline:
//   Laser → MZI mesh (weights loaded via DAC) → Photodetectors → ADC
//   → THIS CORE (accumulate · saturate · ReLU) → inference result
//
// ─── Pin map ────────────────────────────────────────────────────────────────
//  ui_in[3:0]   adc_in[3:0]  – photodetector ADC result  (unsigned, 0–15)
//  ui_in[7:4]   weight[3:0]  – MZI phase coefficient     (signed 2's-comp, −8…+7)
//  uio_in[1:0]  ch_sel[1:0]  – channel select            (0=A, 1=B, 2=C, 3=D)
//  uio_in[2]    relu_en      – 1 = apply ReLU to uo_out
//  uio_in[3]    ch_clear     – 1 = zero the selected channel this cycle
//  uio_in[7:4]  (reserved, ignored)
//  uo_out[7:0]  channel output   – selected accumulator, optionally ReLU-clipped
//  uio_out[3:0] saturation flags – {sat_d, sat_c, sat_b, sat_a}
//  uio_out[7:4] sign flags       – {neg_d, neg_c, neg_b, neg_a}

module tt_um_lightrail_ai_core (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    // ── Input decode ────────────────────────────────────────────────────────
    wire [3:0] adc_in  = ui_in[3:0];   // photodetector ADC (unsigned)
    wire [3:0] weight  = ui_in[7:4];   // MZI weight (signed 2's-complement)
    wire [1:0] ch_sel  = uio_in[1:0];  // channel select
    wire       relu_en = uio_in[2];    // ReLU enable
    wire       ch_clr  = uio_in[3];    // per-channel clear

    // ── Signed multiply: signed 4-bit weight × unsigned 4-bit ADC ──────────
    // Sign-extend weight to 5 bits; zero-extend adc_in to 5 bits.
    // Product range: −8×15 = −120 … +7×15 = +105  →  fits in 8-bit signed.
    wire signed [9:0] prod_wide =
        $signed({weight[3], weight}) * $signed({1'b0, adc_in});
    wire signed [8:0] product = prod_wide[8:0];   // safe: range fits in 9-bit

    // ── Four 8-bit signed saturating accumulators (channels A–D) ───────────
    reg signed [7:0] acc [0:3];

    // Read current channel value and sign-extend for 9-bit sum
    wire signed [7:0] cur = acc[ch_sel];
    wire signed [8:0] sum = {cur[7], cur} + product;

    // Saturate 9-bit signed → 8-bit signed
    // Overflow detected when the two MSBs of the sum disagree:
    //   sum[8:7] == 2'b01  →  positive overflow  →  clamp to +127 (8'h7F)
    //   sum[8:7] == 2'b10  →  negative overflow  →  clamp to −128 (8'h80)
    wire [7:0] sat = (sum[8:7] == 2'b01) ? 8'h7F :
                     (sum[8:7] == 2'b10) ? 8'h80 :
                     sum[7:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            acc[0] <= 8'h00;
            acc[1] <= 8'h00;
            acc[2] <= 8'h00;
            acc[3] <= 8'h00;
        end else if (ena) begin
            acc[ch_sel] <= ch_clr ? 8'h00 : sat;
        end
    end

    // ── Output: selected channel with optional ReLU ─────────────────────────
    wire [7:0] raw = acc[ch_sel];
    assign uo_out  = relu_en ? (raw[7] ? 8'h00 : raw) : raw;

    // ── Status flags ────────────────────────────────────────────────────────
    // Sign flags on upper 4 bits, lower 4 bits zeroed out (input controls)
    assign uio_out = {acc[3][7], acc[2][7], acc[1][7], acc[0][7], 4'b0000};
    assign uio_oe  = 8'hF0;   // upper 4 are outputs (neg_a..d), lower 4 are inputs (controls)

endmodule
