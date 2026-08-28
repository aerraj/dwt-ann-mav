// Six dense layers, Q16.16, 96-bit MAC, ReLU hidden layers.
// Weights are output-major, matching torch.nn.Linear and fixed.py.
module ann_inference #(
    parameter WEIGHTS_FILE="weights.mem", parameter BIASES_FILE="biases.mem",
    parameter MEAN_FILE="mean.mem", parameter SCALE_FILE="inverse_scale.mem"
) (
    input wire clk, input wire rst,
    input wire [287:0] features, input wire features_valid,
    output wire ready,
    output reg signed [31:0] logit, output reg result_valid
);
    reg signed [31:0] weights [0:20799], biases [0:320];
    reg signed [31:0] means [0:8], inverse_scale [0:8];
    reg signed [31:0] source [0:127], destination [0:127];
    reg [287:0] captured;
    initial begin
        $readmemh(WEIGHTS_FILE, weights); $readmemh(BIASES_FILE, biases);
        $readmemh(MEAN_FILE, means); $readmemh(SCALE_FILE, inverse_scale);
    end
    function automatic integer inputs(input integer layer);
        case(layer) 0:inputs=9; 1:inputs=32; 2:inputs=64; 3:inputs=128; 4:inputs=64; default:inputs=32; endcase
    endfunction
    function automatic integer outputs(input integer layer);
        case(layer) 0:outputs=32; 1:outputs=64; 2:outputs=128; 3:outputs=64; 4:outputs=32; default:outputs=1; endcase
    endfunction
    function automatic signed [31:0] sat(input signed [95:0] value);
        if(value > 96'sd2147483647) sat=32'sh7fffffff;
        else if(value < -96'sd2147483648) sat=32'sh80000000;
        else sat=value[31:0];
    endfunction
    localparam IDLE=0, NORM=1, INIT=2, MAC=3, STORE=4, COPY=5;
    reg [2:0] state;
    integer layer, neuron, column, weight_base, bias_base, k;
    reg signed [95:0] acc;
    wire signed [31:0] raw_feature = $signed(captured[column*32 +: 32]);
    wire signed [32:0] centered = {raw_feature[31], raw_feature} - {means[column][31], means[column]};
    wire signed [64:0] normalized = centered * inverse_scale[column];
    wire signed [63:0] product = source[column] * weights[weight_base+neuron*inputs(layer)+column];
    wire signed [31:0] activation = sat(acc >>> 16);
    assign ready = state == IDLE;
    always @(posedge clk) begin
        if(rst) begin
            state<=IDLE; result_valid<=0; logit<=0; captured<=0;
            layer<=0; neuron<=0; column<=0; weight_base<=0; bias_base<=0; acc<=0;
        end else begin
            result_valid<=0;
            case(state)
                IDLE: if(features_valid) begin captured<=features; column<=0; state<=NORM; end
                NORM: begin
                    source[column]<=sat($signed(normalized) >>> 16);
                    if(column==8) begin
                        layer<=0; neuron<=0; column<=0; weight_base<=0; bias_base<=0; state<=INIT;
                    end else column<=column+1;
                end
                INIT: begin
                    acc<={{64{biases[bias_base+neuron][31]}},biases[bias_base+neuron]} <<< 16;
                    column<=0; state<=MAC;
                end
                MAC: begin
                    acc<=acc+product;
                    if(column==inputs(layer)-1) state<=STORE;
                    else column<=column+1;
                end
                STORE: begin
                    destination[neuron] <= (layer<5 && activation<0) ? 32'sd0 : activation;
                    if(layer==5) begin logit<=activation; result_valid<=1; state<=IDLE; end
                    else if(neuron==outputs(layer)-1) state<=COPY;
                    else begin neuron<=neuron+1; state<=INIT; end
                end
                COPY: begin
                    for(k=0;k<128;k=k+1) source[k]<=destination[k];
                    weight_base<=weight_base+inputs(layer)*outputs(layer);
                    bias_base<=bias_base+outputs(layer);
                    layer<=layer+1; neuron<=0; state<=INIT;
                end
                default: state<=IDLE;
            endcase
        end
    end
endmodule
