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

    // LightRail Gen3 - SIMD Dual-Lane Tensor MAC Core
    // 1x1 Tiny Tapeout tile, Sky130 PDK
    //
    // Lane A: ui_in[7:4]=weight_a,  ui_in[3:0]=input_a  -> acc_a on uo_out
    // Lane B: uio_in[7:4]=weight_b, uio_in[3:0]=input_b -> acc_b on uio_out
    //
    // Each clock (ena=1): acc_a += weight_a*input_a, acc_b += weight_b*input_b
    // Dot product of 2N-element vectors in N cycles: dot = uo_out + uio_out

    wire [3:0] weight_a = ui_in[7:4];
    wire [3:0] input_a  = ui_in[3:0];
    wire [3:0] weight_b = uio_in[7:4];
    wire [3:0] input_b  = uio_in[3:0];

    // Zero-extend operands to 8 bits before multiply to guarantee
    // full-precision 8-bit products (max 15*15=225 fits in 8 bits).
    wire [7:0] product_a = {4'b0, weight_a} * {4'b0, input_a};
    wire [7:0] product_b = {4'b0, weight_b} * {4'b0, input_b};

    reg [7:0] acc_a;
    reg [7:0] acc_b;

    always @(posedge clk) begin
        if (!rst_n) begin
            acc_a <= 8'h00;
            acc_b <= 8'h00;
        end else if (ena) begin
            acc_a <= acc_a + product_a;
            acc_b <= acc_b + product_b;
        end
    end

    assign uo_out  = acc_a;
    assign uio_out = acc_b;
    assign uio_oe  = 8'hFF;

endmodule
