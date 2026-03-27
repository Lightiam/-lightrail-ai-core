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
    // LightRail Gen3 - Phase 4: Beyond Binary Core (Tiny Tapeout Prototype)
    // This is a foundational MAC (Multiply-Accumulate) unit used in AI Tensor Cores.
    // It takes a 4-bit weight and 4-bit input, multiplies them, and accumulates.
    // =========================================================================

    wire [3:0] ai_weight = ui_in[7:4];
    wire [3:0] ai_input  = ui_in[3:0];
    
    reg [7:0] accumulator;
    wire [7:0] product = ai_weight * ai_input;

    // Hardware logic running on the silicon clock
    always @(posedge clk) begin
        if (!rst_n) begin
            accumulator <= 8'b0; // Reset the AI core
        end else if (ena) begin
            accumulator <= accumulator + product; // Tensor Math Step
        end
    end

    // Route the final calculation to the physical output pins of the microchip
    assign uo_out = accumulator;

    // Tie off unused IO pins to 0 as required by Tiny Tapeout factory rules
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

endmodule
