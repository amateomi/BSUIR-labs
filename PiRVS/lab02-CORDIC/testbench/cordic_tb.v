`timescale 1ns / 1ps
module cordic_tb();
reg clk = 0;
always #5 clk <= ~clk;

reg reset = 1;

reg is_direct = 1;
reg signed [15:0] x_in = 0;
reg signed [15:0] y_in = 0;
reg signed [15:0] z_in = 0;

wire signed [15:0] x_out;
wire signed [15:0] y_out;
wire signed [15:0] z_out;
wire valid;

initial begin
    @(posedge clk);
    x_in <= 16'b0010000000000000; // 1
    y_in <= 16'b0000000000000000; // 0
    z_in <= 16'b0010000110000010; // 60 degrees
    reset <= 0;
    @(posedge valid);
    $display("1: 0b%3b.%13b 0b%3b.%13b 0b%3b.%13b",
             x_out[15:13], x_out[12:0],
             y_out[15:13], y_out[12:0],
             z_out[15:13], z_out[12:0]);
    @(posedge clk);
    reset <= 1;
    @(posedge clk);
    x_in <= 16'b0010000000000000; // 1
    y_in <= 16'b0000000000000000; // 0
    z_in <= 16'b0101001111000110; // 150 degrees
    reset <= 0;
    @(posedge valid);
    $display("2: 0b%3b.%13b 0b%3b.%13b 0b%3b.%13b",
             x_out[15:13], x_out[12:0],
             y_out[15:13], y_out[12:0],
             z_out[15:13], z_out[12:0]);
    @(posedge clk);
    reset <= 1;
    @(posedge clk);
    // Example from presentation: conversion from cartesian to polar system
    is_direct <= 0;
    x_in <= 16'b0001111110101110; // 0.99
    y_in <= 16'b0000111110101110; // 0.49
    z_in <= 16'b0000000000000000;
    reset <= 0;
    @(posedge valid);
    $display("3: 0b%3b.%13b 0b%3b.%13b 0b%3b.%13b",
             x_out[15:13], x_out[12:0],
             y_out[15:13], y_out[12:0],
             z_out[15:13], z_out[12:0]);
    @(posedge clk);
    $finish;
end
cordic cordic(.clk(clk),
              .reset(reset),
              .is_direct(is_direct),
              .x_in(x_in),
              .y_in(y_in),
              .z_in(z_in),
              .x_out(x_out),
              .y_out(y_out),
              .z_out(z_out),
              .valid(valid));
endmodule
