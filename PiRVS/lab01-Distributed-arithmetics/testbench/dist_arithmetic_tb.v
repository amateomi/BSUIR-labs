`timescale 1ns / 1ps
module dist_arithmetic_tb();
reg clk = 0;
always #5 clk <= ~clk;

reg reset = 1;

reg signed [15:0] x[7:0];
reg signed [7:0] temp_x_lsb = 0;
reg signed [7:0] temp_x_msb = 0;

wire [16:0] y;
wire valid;

integer i = 0;

initial begin
    @(posedge clk);
    reset <= 0;
    x[0] = 16'b0111111111111111;
    x[1] = 16'b0000000000000000;
    x[2] = 16'b0000000000000000;
    x[3] = 16'b0000000000000000;
    x[4] = 16'b0000000000000000;
    x[5] = 16'b0000000000000000;
    x[6] = 16'b0000000000000000;
    x[7] = 16'b0000000000000000;
    for (i = 0; i < 16; i = i + 2)
    begin
        temp_x_lsb = {x[0][i],
                      x[1][i],
                      x[2][i],
                      x[3][i],
                      x[4][i],
                      x[5][i],
                      x[6][i],
                      x[7][i]};
        temp_x_msb = {x[0][i + 1],
                      x[1][i + 1],
                      x[2][i + 1],
                      x[3][i + 1],
                      x[4][i + 1],
                      x[5][i + 1],
                      x[6][i + 1],
                      x[7][i + 1]};
        @(posedge clk);
    end
    reset <= 1;
    @(posedge clk);
    $finish;
end
dist_arithmetic da(.clk(clk),
                   .reset(reset),
                   .x_msb(temp_x_msb),
                   .x_lsb(temp_x_lsb),
                   .y(y),
                   .valid(valid));
endmodule
