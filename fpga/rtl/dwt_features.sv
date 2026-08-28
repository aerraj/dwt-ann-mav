// Valid FIR/decimation, nine mean-absolute detail features, D1 first.
// Reference architecture: one shared pair of MACs and in-place sample RAM.
// Capture a contiguous window through ready/valid; no padding or overlap.
module dwt_features #(
    parameter integer WINDOW = 65536,
    parameter integer TAPS = 88,
    parameter LOW_FILE = "dwt_low.mem",
    parameter HIGH_FILE = "dwt_high.mem"
) (
    input wire clk, input wire rst,
    input wire signed [31:0] sample, input wire sample_valid,
    output wire sample_ready,
    output reg [287:0] features, output reg features_valid
);
    reg signed [31:0] samples [0:WINDOW-1];
    reg signed [31:0] low [0:TAPS-1], high [0:TAPS-1];
    initial begin
        $readmemh(LOW_FILE, low);
        $readmemh(HIGH_FILE, high);
    end
    localparam CAPTURE=0, MAC=1, STORE=2, FEATURE=3, NEXT=4;
    reg [2:0] state;
    integer capture_index, level, length, outputs, index, tap;
    reg signed [95:0] low_acc, high_acc;
    reg [95:0] abs_sum;
    wire signed [31:0] sample_word = samples[2*index+tap];
    wire signed [63:0] low_product = sample_word * low[TAPS-1-tap];
    wire signed [63:0] high_product = sample_word * high[TAPS-1-tap];
    function automatic signed [31:0] sat(input signed [95:0] value);
        if (value > 96'sd2147483647) sat = 32'sh7fffffff;
        else if (value < -96'sd2147483648) sat = 32'sh80000000;
        else sat = value[31:0];
    endfunction
    wire signed [31:0] detail = sat(high_acc >>> 30);
    wire signed [32:0] detail_wide = {detail[31], detail};
    wire [32:0] detail_abs = detail_wide < 0 ? -detail_wide : detail_wide;
    assign sample_ready = state == CAPTURE;
    always @(posedge clk) begin
        if (rst) begin
            state <= CAPTURE; capture_index <= 0;
            features <= 0; features_valid <= 0;
            low_acc <= 0; high_acc <= 0; abs_sum <= 0;
            level <= 0; length <= WINDOW; outputs <= (WINDOW-TAPS)/2+1;
            index <= 0; tap <= 0;
        end else begin
            features_valid <= 0;
            case (state)
                CAPTURE: if (sample_valid) begin
                    samples[capture_index] <= sample;
                    if (capture_index == WINDOW-1) begin
                        capture_index <= 0; state <= MAC;
                        level <= 0; length <= WINDOW; outputs <= (WINDOW-TAPS)/2+1;
                        index <= 0; tap <= 0;
                        low_acc <= 0; high_acc <= 0; abs_sum <= 0;
                    end else capture_index <= capture_index+1;
                end
                MAC: begin
                    low_acc <= low_acc + low_product;
                    high_acc <= high_acc + high_product;
                    if (tap == TAPS-1) state <= STORE;
                    else tap <= tap+1;
                end
                STORE: begin
                    samples[index] <= sat(low_acc >>> 30);
                    abs_sum <= abs_sum + detail_abs;
                    low_acc <= 0; high_acc <= 0; tap <= 0;
                    if (index == outputs-1) state <= FEATURE;
                    else begin index <= index+1; state <= MAC; end
                end
                FEATURE: begin
                    features[level*32 +: 32] <= sat($signed(abs_sum / outputs));
                    state <= NEXT;
                end
                NEXT: begin
                    if (level == 8) begin
                        features_valid <= 1; state <= CAPTURE;
                    end else begin
                        level <= level+1; length <= outputs;
                        outputs <= (outputs-TAPS)/2+1;
                        index <= 0; tap <= 0; abs_sum <= 0;
                        state <= MAC;
                    end
                end
                default: state <= CAPTURE;
            endcase
        end
    end
endmodule
