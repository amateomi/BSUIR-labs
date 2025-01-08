module filter(
    input clk,
    input [15:0] x,
    input valid_in,
    input ready_in,
    output [16:0] y,
    output ready_out,
    output valid_out
);

localparam STATE_READY = 2'b00;
localparam STATE_COMPUTING = 2'b01;
localparam STATE_OUTPUT_READY = 2'b10;
localparam STATE_OUTPUTING = 2'b11;
reg [1:0] state = STATE_READY;

reg [127:0] array = 0;
reg [16:0] ready_y = 0;
reg da_reset = 1;

wire signed [7:0] x_lsb = {array[112],
                           array[96],
                           array[80],
                           array[64],
                           array[48],
                           array[32],
                           array[16],
                           array[0]};
wire signed [7:0] x_msb = {array[113],
                           array[97],
                           array[81],
                           array[65],
                           array[49],
                           array[33],
                           array[17],
                           array[1]};

assign ready_out = state == STATE_READY;
assign valid_out = state == STATE_OUTPUTING;
assign y = valid_out ? ready_y : 0;

wire [16:0] da_y;
wire da_valid;

dist_arithmetic da(.clk(clk),
                   .reset(da_reset),
                   .x_msb(x_msb),
                   .x_lsb(x_lsb),
                   .y(da_y),
                   .valid(da_valid));

always @(posedge clk)
begin
    case (state)
        STATE_READY: begin
            if (valid_in) begin
                array[127] <= x[15];
                array[126] <= x[14];
                array[125] <= x[13];
                array[124] <= x[12];
                array[123] <= x[11];
                array[122] <= x[10];
                array[121] <= x[9];
                array[120] <= x[8];
                array[119] <= x[7];
                array[118] <= x[6];
                array[117] <= x[5];
                array[116] <= x[4];
                array[115] <= x[3];
                array[114] <= x[2];
                array[113] <= x[1];
                array[112] <= x[0];
                state <= STATE_COMPUTING;
            end
        end
        STATE_COMPUTING: begin
            if (da_valid) begin
                ready_y <= da_y;
                da_reset <= 1;
                state <= STATE_OUTPUT_READY;
            end else begin
                da_reset <= 0;
                array <= array >> 2;
            end
        end
        STATE_OUTPUT_READY: begin
            if (ready_in)
               state <= STATE_OUTPUTING;
        end
        STATE_OUTPUTING: begin
            state <= STATE_READY;
        end
    endcase
end
endmodule
