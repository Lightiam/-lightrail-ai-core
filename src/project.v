`default_nettype none

// LightRail AI Gen3 - LR-P8A Photonic Inference Core
// Tiny Tapeout 1x1 tile, Sky130 PDK
//
// Digital post-processor for a photonic matrix-vector accelerator.
// 4 channels (A-D) each apply a signed MZI weight to an unsigned
// photodetector ADC reading and accumulate into a saturating 8-bit
// signed register. Optional per-output ReLU supports NN inference.
//
// Photonic inference pipeline:
//   Laser -> MZI mesh (weights) -> Photodetectors -> ADC -> THIS CORE -> ReLU
//
// ui_in[3:0]  : adc_in   - photodetector ADC result (unsigned, 0-15)
// ui_in[7:4]  : weight   - MZI phase coefficient (signed 2's-comp, -8 to +7)
// uio_in[1:0] : ch_sel   - channel select (0=A, 1=B, 2=C, 3=D)
// uio_in[2]   : relu_en  - 1 = apply ReLU to uo_out
// uio_in[3]   : ch_clear - 1 = zero selected channel this cycle
// uo_out[7:0] : channel accumulator output (post-ReLU if relu_en=1)
// uio_out[3:0]: saturation flags {sat_d, sat_c, sat_b, sat_a}
// uio_out[7:4]: sign flags       {neg_d, neg_c, neg_b, neg_a}

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

    // -- Input decode ---------------------------------------------------------
    wire [3:0] adc_in  = ui_in[3:0];   // unsigned photodetector ADC
    wire [3:0] weight  = ui_in[7:4];   // signed MZI weight (2's complement)
    wire [1:0] ch_sel  = uio_in[1:0];
    wire       relu_en = uio_in[2];
    wire       ch_clr  = uio_in[3];

    // -- Signed multiply: 4-bit signed weight x 4-bit unsigned ADC -----------
    // Sign-extend weight to 9 bits; zero-extend adc_in to 9 bits.
    // Full 18-bit product avoids truncation warnings; range -120 to +105.
    wire signed [8:0] w9 = {{5{weight[3]}}, weight};
    wire signed [8:0] a9 = {5'b00000, adc_in};
    wire signed [17:0] prod_full = w9 * a9;
    // Lower 9 bits hold the exact result for our value range.
    wire signed [8:0] product = $signed(prod_full[8:0]);

    // -- Four named 8-bit signed saturating accumulators (channels A-D) ------
    reg signed [7:0] acc_a, acc_b, acc_c, acc_d;

    // Select current channel for read (explicit mux, no dynamic indexing)
    wire signed [7:0] cur = (ch_sel == 2'd1) ? acc_b :
                            (ch_sel == 2'd2) ? acc_c :
                            (ch_sel == 2'd3) ? acc_d : acc_a;

    // 9-bit signed sum: range -248 to +232, within 9-bit signed (-256..+255)
    wire signed [8:0] sum = $signed({cur[7], cur}) + product;

    // Saturate to 8-bit signed: overflow when the two MSBs of sum disagree
    //   sum[8:7] == 2'b01 -> positive overflow -> clamp +127
    //   sum[8:7] == 2'b10 -> negative overflow -> clamp -128
    wire [7:0] sat = (sum[8:7] == 2'b01) ? 8'h7F :
                     (sum[8:7] == 2'b10) ? 8'h80 :
                     sum[7:0];

    // -- Synchronous accumulate/clear ----------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            acc_a <= 8'h00;
            acc_b <= 8'h00;
            acc_c <= 8'h00;
            acc_d <= 8'h00;
        end else if (ena) begin
            case (ch_sel)
                2'd0: acc_a <= ch_clr ? 8'h00 : sat;
                2'd1: acc_b <= ch_clr ? 8'h00 : sat;
                2'd2: acc_c <= ch_clr ? 8'h00 : sat;
                2'd3: acc_d <= ch_clr ? 8'h00 : sat;
            endcase
        end
    end

    // -- Output: selected channel with optional ReLU -------------------------
    wire [7:0] raw = (ch_sel == 2'd1) ? acc_b :
                     (ch_sel == 2'd2) ? acc_c :
                     (ch_sel == 2'd3) ? acc_d : acc_a;
    assign uo_out = relu_en ? (raw[7] ? 8'h00 : raw) : raw;

    // -- Status flags ---------------------------------------------------------
    wire sat_a = (acc_a == 8'h7F) | (acc_a == 8'h80);
    wire sat_b = (acc_b == 8'h7F) | (acc_b == 8'h80);
    wire sat_c = (acc_c == 8'h7F) | (acc_c == 8'h80);
    wire sat_d = (acc_d == 8'h7F) | (acc_d == 8'h80);

    assign uio_out = {acc_d[7], acc_c[7], acc_b[7], acc_a[7],
                      sat_d, sat_c, sat_b, sat_a};
    assign uio_oe  = 8'hFF;

endmodule
