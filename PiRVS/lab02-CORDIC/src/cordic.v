module cordic(
    input clk,
    input reset,
    input is_direct,
    input signed [15:0] x_in,
    input signed [15:0] y_in,
    input signed [15:0] z_in,
    output signed [15:0] x_out,
    output signed [15:0] y_out,
    output signed [15:0] z_out,
    output valid
);

integer counter = 0;
always @(posedge clk or negedge reset)
begin
    if (reset)
        counter <= 0;
    else
        counter <= counter + 1;
end

reg signed [15:0] x = 0;
reg signed [15:0] y = 0;
reg signed [15:0] z = 0;

assign valid = counter > 13 && !reset;
assign x_out = valid ? x : 0;
assign y_out = valid ? y : 0;
assign z_out = valid ? z : 0;

reg signed [15:0] ROM[11:0];
initial begin
    ROM[0] =  16'b0001100100100001;
    ROM[1] =  16'b0000111011010110;
    ROM[2] =  16'b0000011111010110;
    ROM[3] =  16'b0000001111111010;
    ROM[4] =  16'b0000000111111111;
    ROM[5] =  16'b0000000011111111;
    ROM[6] =  16'b0000000001111111;
    ROM[7] =  16'b0000000000111111;
    ROM[8] =  16'b0000000000011111;
    ROM[9] =  16'b0000000000001111;
    ROM[10] = 16'b0000000000000111;
    ROM[11] = 16'b0000000000000011;
end

localparam signed [15:0] K       = 16'b0001001101101110;
localparam signed [15:0] pi      = 16'b0110010010000111;
localparam signed [15:0] pi_half = pi >>> 1;

function [15:0] scaler(input signed [15:0] value);
    begin
        // K[15] ignored intentionally because K is positive constant
        scaler = (K[14] ? value <<< 1  : 0) +
                 (K[13] ? value        : 0) +
                 (K[12] ? value >>> 1  : 0) +
                 (K[11] ? value >>> 2  : 0) +
                 (K[10] ? value >>> 3  : 0) +
                 (K[9]  ? value >>> 4  : 0) +
                 (K[8]  ? value >>> 5  : 0) +
                 (K[7]  ? value >>> 6  : 0) +
                 (K[6]  ? value >>> 7  : 0) +
                 (K[5]  ? value >>> 8  : 0) +
                 (K[4]  ? value >>> 9  : 0) +
                 (K[3]  ? value >>> 10 : 0) +
                 (K[2]  ? value >>> 11 : 0) +
                 (K[1]  ? value >>> 12 : 0);
    end
endfunction

always @(posedge clk or negedge reset)
begin
    if (reset) begin
        x <= 0;
        y <= 0;
        z <= 0;
    end else if (counter == 0) begin // preprocessing
        if (is_direct) begin
            if (-pi_half <= z_in && z_in <= pi_half) begin
                x <= x_in;
                y <= y_in;
                z <= z_in;
            end else if (pi_half < z_in && z_in < pi) begin
                x <= -y_in;
                y <= x_in;
                z <= z_in - pi_half;
            end else if (-pi < z_in && z_in < -pi_half) begin
                x <= y_in;
                y <= -x_in;
                z <= z_in + pi_half;
            end else begin
                $display("Invalid z=%16b", z_in);
            end
        end else begin
            if (x_in > 0) begin
                x <= x_in;
                y <= y_in;
                z <= 0;
            end else if (x_in < 0 && y_in > 0) begin
                x <= y_in;
                y <= -x_in;
                z <= pi_half;
            end else if (x_in < 0 && y_in < 0) begin
                x <= -y_in;
                y <= x_in;
                z <= -pi_half;
            end else begin
               $display("Invalid x=%16b or y=%16b", x_in, y_in);
            end
        end
    end else if (counter < 13) begin // cordic core
        if (is_direct) begin
            if (z[15] == 0) begin
                x <= x - (y >>> (counter - 1));
                y <= y + (x >>> (counter - 1));
                z <= z - ROM[counter - 1];
            end else begin
                x <= x + (y >>> (counter - 1));
                y <= y - (x >>> (counter - 1));
                z <= z + ROM[counter - 1];
            end
        end else begin
            if (y[15] == 0) begin
                x <= x + (y >>> (counter - 1));
                y <= y - (x >>> (counter - 1));
                z <= z + ROM[counter - 1];
            end else begin
                x <= x - (y >>> (counter - 1));
                y <= y + (x >>> (counter - 1));
                z <= z - ROM[counter - 1];
            end
        end
    end else if (counter == 13) begin // postprocessing
        x <= scaler(x);
        y <= scaler(y);
    end
end
endmodule
