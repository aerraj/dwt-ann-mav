// Board-independent top. ADC protocol/pins/voltage conditioning belong in a
// board wrapper. adc_valid obeys ready/valid; SW420 is independent of DWT/ANN.
`include "model_config.svh"
module motor_monitor #(
    parameter integer WINDOW=`MODEL_WINDOW, TAPS=`MODEL_TAPS,
    parameter integer ADC_BITS=`ADC_BITS,
    parameter signed [31:0] ADC_GAIN=`ADC_GAIN, ADC_OFFSET=`ADC_OFFSET,
    parameter signed [31:0] THRESHOLD=`MODEL_THRESHOLD,
    parameter integer WATCHDOG_CYCLES=2000000000
) (
    input wire clk, input wire rst,
    input wire [ADC_BITS-1:0] adc_code, input wire adc_valid,
    output wire adc_ready,
    input wire vibration_async, input wire sensor_ok, input wire acknowledge,
    output wire trip, output wire relay_enable,
    output wire signed [31:0] logit,
    output wire [31:0] probability_q16,
    output wire result_valid
);
    wire signed [63:0] converted = $signed({1'b0, adc_code}) * ADC_GAIN - ADC_OFFSET;
    wire signed [31:0] sample = converted > 64'sd2147483647 ? 32'sh7fffffff :
                                converted < -64'sd2147483648 ? 32'sh80000000 : converted[31:0];
    wire [287:0] features;
    wire features_valid, dwt_ready, ann_ready;
    // The next window cannot start until the ANN accepts the previous one.
    assign adc_ready = dwt_ready && ann_ready && !features_valid;
    dwt_features #(.WINDOW(WINDOW), .TAPS(TAPS)) dwt (
        .clk(clk), .rst(rst), .sample(sample), .sample_valid(adc_valid && adc_ready),
        .sample_ready(dwt_ready), .features(features), .features_valid(features_valid));
    ann_inference ann (.clk(clk), .rst(rst), .features(features),
        .features_valid(features_valid), .ready(ann_ready), .logit(logit), .result_valid(result_valid));
    reg [31:0] sigmoid [0:256];
    initial $readmemh("sigmoid.mem", sigmoid);
    wire signed [32:0] shifted = $signed(logit) + 33'sd524288;
    wire [8:0] lut_index = logit <= -32'sd524288 ? 9'd0 :
                           logit >= 32'sd524288 ? 9'd256 : shifted[20:12];
    assign probability_q16 = sigmoid[lut_index];
    protection #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES), .THRESHOLD(THRESHOLD)) guard (
        .clk(clk), .rst(rst), .acknowledge(acknowledge), .vibration_async(vibration_async),
        .sensor_ok(sensor_ok), .result_valid(result_valid), .logit(logit),
        .trip(trip), .relay_enable(relay_enable));
endmodule
