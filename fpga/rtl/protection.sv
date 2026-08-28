// Synchronous, latched de-energize-to-trip policy. Two-flop SW420 synchronizer.
// External E-stop/contactor interlocks are still required. No auto-restart.
module protection #(
    parameter integer WATCHDOG_CYCLES=2000000000,
    parameter signed [31:0] THRESHOLD=0
) (
    input wire clk, input wire rst, input wire acknowledge,
    input wire vibration_async, input wire sensor_ok,
    input wire result_valid, input wire signed [31:0] logit,
    output reg trip, output wire relay_enable
);
    reg vibration_meta, vibration_sync;
    reg has_result, last_fault;
    reg acknowledge_previous;
    reg [31:0] age;
    wire expired = !has_result || (!result_valid && age >= WATCHDOG_CYCLES-1);
    wire fault = result_valid ? logit >= THRESHOLD : last_fault;
    assign relay_enable = !trip && !rst;
    always @(posedge clk) begin
        if(rst) begin
            vibration_meta<=0; vibration_sync<=0; has_result<=0;
            last_fault<=1; age<=0; trip<=1;
            acknowledge_previous<=0;
        end else begin
            acknowledge_previous<=acknowledge;
            vibration_meta<=vibration_async; vibration_sync<=vibration_meta;
            if(result_valid) begin has_result<=1; last_fault<=logit>=THRESHOLD; age<=0; end
            else if(age < WATCHDOG_CYCLES) age<=age+1;
            if(!sensor_ok || vibration_sync || fault || (expired && !result_valid)) trip<=1;
            else if(acknowledge && !acknowledge_previous && (has_result || result_valid)) trip<=0;
        end
    end
endmodule
