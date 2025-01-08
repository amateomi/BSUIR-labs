`timescale 1ns / 1ps
module filter_tb();
reg clk = 0;
always #5 clk <= ~clk;

reg [15:0] x = 0;
reg valid_in = 0;
reg ready_in = 0;

wire [16:0] y;
wire ready_out;
wire valid_out;

integer i = 0;

initial begin
    @(posedge clk);
    x <= 16'b0111111111111111;
    valid_in <= 1;
    ready_in <= 1;

    @(negedge ready_out);
    x <= 16'b0000000000000000;
    valid_in <= 0;
    
    for(i=0; i<8; i=i+1) begin
        @(posedge valid_out);
        $display("a%1d=%17b", i, y);
        valid_in <= 1;
    end

    @(posedge clk);
    $finish;
end
filter filter(.clk(clk),
              .x(x),
              .valid_in(valid_in),
              .ready_in(ready_in),
              .y(y),
              .ready_out(ready_out),
              .valid_out(valid_out));
endmodule
