`default_nettype none

module tt_um_lightrail_ai_core (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 1=output, 0=input)
    input  wire       ena,      // will go high when the design is enabled
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

    // =========================================================================
    // LightRail Gen3 - Phase 4: SIMD Dual-Lane Tensor MAC Core
    // (Tiny Tapeout Prototype — 1x1 tile, Sky130 PDK)
    //
    // Architecture: 2-lane SIMD MAC array enabling neural network dot products.
    //
    //   Lane A (ui_in):   weight_a[3:0] = ui_in[7:4],   input_a[3:0] = ui_in[3:0]
    //   Lane B (uio_in):  weight_b[3:0] = uio_in[7:4],  input_b[3:0] = uio_in[3:0]
    //
    //   Each clock cycle (when ena=1):
    //       acc_a <= acc_a + (weight_a * input_a)   // Lane A accumulates
    //       acc_b <= acc_b + (weight_b * input_b)   // Lane B accumulates (SIMD)
    //
    //   uo_out  = acc_a  (Lane A 8-bit accumulator output)
    //   uio_out = acc_b  (Lane B 8-bit accumulator output)
    //
    // Dot-Product computation for an N-element vector pair:
    //   Feed ceil(N/2) dimension pairs per cycle (interleaved across lanes).
    //   After ceil(N/2) cycles:  dot_product = uo_out + uio_out
    //
    // Example — 4-element dot product [a0,a1,a2,a3] · [w0,w1,w2,w3] = 20:
    //   Cycle 1: Lane A = (w0=4, a0=1) → acc_a=4
    //            Lane B = (w1=3, a1=2) → acc_b=6
    //   Cycle 2: Lane A = (w2=2, a2=3) → acc_a=4+6=10
    //            Lane B = (w3=1, a3=4) → acc_b=6+4=10
    //   Result:  dot_product = acc_a + acc_b = 10 + 10 = 20  ✓
    //
    // Area estimate: 2× 4-bit multiplier + 2× 8-bit adder/reg ≈ ~200 sky130 cells
    // Well within the ~1000-cell budget of a 1×1 Tiny Tapeout tile.
    // =========================================================================

    // --- Lane A: dedicated input pins ---
    wire [3:0] weight_a = ui_in[7:4];
    wire [3:0] input_a  = ui_in[3:0];

    // --- Lane B: bidirectional pins used as inputs ---
    wire [3:0] weight_b = uio_in[7:4];
    wire [3:0] input_b  = uio_in[3:0];

    // Parallel combinatorial 4x4 multipliers (8-bit products, no overflow)
    wire [7:0] product_a = weight_a * input_a;
    wire [7:0] product_b = weight_b * input_b;

    // 8-bit accumulation registers — one per SIMD lane
    // Intentional wrapping on overflow (mod-256 arithmetic)
    reg [7:0] acc_a;
    reg [7:0] acc_b;

    always @(posedge clk) begin
        if (!rst_n) begin
            acc_a <= 8'h00;   // Reset Lane A accumulator
            acc_b <= 8'h00;   // Reset Lane B accumulator
        end else if (ena) begin
            acc_a <= acc_a + product_a;   // Lane A: accumulate MAC result
            acc_b <= acc_b + product_b;   // Lane B: accumulate MAC result (SIMD)
        end
        // ena=0: both accumulators hold their current value (frozen)
    end

    // Lane A accumulator → dedicated output pins
    assign uo_out  = acc_a;

    // Lane B accumulator → bidirectional pins (all driven as outputs)
    assign uio_out = acc_b;
    assign uio_oe  = 8'hFF;   // All uio pins are outputs

endmodule
