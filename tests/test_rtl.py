"""Compile and execute the actual RTL against independent Python integer vectors."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dwt_ann_mav.features import FeatureConfig
from dwt_ann_mav.fixed import export_fpga, fixed_ann, fixed_features, quantize, write_mem
from dwt_ann_mav.model import load_model

RTL = Path(__file__).parents[1] / "fpga/rtl"
pytestmark = pytest.mark.skipif(
    not shutil.which("iverilog") or not shutil.which("vvp"), reason="Icarus Verilog is required"
)


def simulate(directory, body, files, timeout=120):
    (directory / "tb.sv").write_text(body)
    result = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-s",
            "tb",
            "-I",
            str(directory),
            "-o",
            str(directory / "sim.vvp"),
            str(directory / "tb.sv"),
            *[str(RTL / name) for name in files],
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        ["vvp", str(directory / "sim.vvp")],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("wavelet,size", [("db4", 4096), ("db44", 65536)])
def test_dwt_bit_exact(tmp_path, wavelet, size):
    from dwt_ann_mav.features import filters

    config = FeatureConfig(wavelet=wavelet, window_size=size, stride=size)
    rng = np.random.default_rng(5)
    # Includes negative samples and a transient, not just zeros.
    samples = quantize(rng.normal(0, 2, size) + np.sin(np.arange(size) * 0.17))
    expected = fixed_features(samples, config, already_quantized=True)
    low, high = filters(wavelet)
    write_mem(tmp_path / "dwt_low.mem", quantize(low, 30))
    write_mem(tmp_path / "dwt_high.mem", quantize(high, 30))
    write_mem(tmp_path / "samples.mem", samples)
    write_mem(tmp_path / "expected.mem", expected)
    simulate(
        tmp_path,
        f"""
module tb;
reg clk=0; always #5 clk=~clk;
reg rst=1, valid=0; reg signed [31:0] sample=0;
wire ready, done; wire [287:0] features;
reg [31:0] data[0:{size - 1}], expected[0:8]; integer i, frame;
dwt_features #(.WINDOW({size}),.TAPS({len(low)})) dut(clk,rst,sample,valid,ready,features,done);
initial begin
$readmemh("samples.mem", data); $readmemh("expected.mem", expected);
repeat(3) @(negedge clk); rst=0;
for(frame=0;frame<2;frame=frame+1) begin
 for(i=0;i<{size};i=i+1) begin
  @(negedge clk); if(!ready) $fatal(1,"capture not ready"); valid=1; sample=data[i];
 end
 @(negedge clk); valid=0;
 wait(done); @(negedge clk);
 for(i=0;i<9;i=i+1) if(features[i*32 +: 32] !== expected[i])
  $fatal(1,"feature %0d actual %h expected %h",i,features[i*32 +: 32],expected[i]);
end
$display("PASS DWT {wavelet}"); $finish;
end
initial begin #200000000; $fatal(1,"DWT timeout"); end
endmodule
""",
        ["dwt_features.sv"],
    )


def test_ann_bit_exact(artifact, tmp_path):
    model, mean, scale, _, _ = load_model(artifact)
    export_fpga(artifact, tmp_path)
    vectors = quantize(
        np.vstack(
            [
                np.random.default_rng(1).normal(size=(4, 9)),
                np.full((1, 9), 32767.0),
                np.full((1, 9), -32768.0),
            ]
        )
    )
    expected = [fixed_ann(v, model, mean, scale) for v in vectors]
    write_mem(tmp_path / "vectors.mem", vectors)
    write_mem(tmp_path / "expected.mem", expected)
    simulate(
        tmp_path,
        """
module tb;
reg clk=0; always #5 clk=~clk;
reg rst=1, valid=0; reg [287:0] features=0;
wire ready, done; wire signed [31:0] logit;
reg [31:0] data[0:53], expected[0:5]; integer i,j;
ann_inference dut(clk,rst,features,valid,ready,logit,done);
initial begin
$readmemh("vectors.mem",data); $readmemh("expected.mem",expected);
repeat(3) @(negedge clk); rst=0;
for(j=0;j<6;j=j+1) begin
 @(negedge clk); if(!ready) $fatal(1,"ANN busy");
 for(i=0;i<9;i=i+1) features[i*32 +: 32]=data[j*9+i]; valid=1;
 @(negedge clk); valid=0;
 wait(done); @(negedge clk);
 if(logit !== expected[j]) $fatal(1,"ANN actual %h expected %h",logit,expected[j]);
end
$display("PASS ANN"); $finish;
end
initial begin #2000000; $fatal(1,"ANN timeout"); end
endmodule
""",
        ["ann_inference.sv"],
    )


def test_protection_rtl(tmp_path):
    simulate(
        tmp_path,
        """
module tb;
reg clk=0; always #5 clk=~clk;
reg rst=1, ack=0, vibration=0, sensor_ok=1, valid=0;
reg signed [31:0] logit=-65536; wire trip, enable;
protection #(.WATCHDOG_CYCLES(20)) dut(clk,rst,ack,vibration,sensor_ok,valid,logit,trip,enable);
initial begin
repeat(3) @(negedge clk); rst=0;
@(negedge clk); if(!trip || enable) $fatal(1,"startup must trip");
ack=1; valid=1; @(negedge clk); ack=0; valid=0;
if(trip || !enable) $fatal(1,"healthy acknowledgment should arm");
vibration=1; repeat(4) @(negedge clk);
if(!trip) $fatal(1,"independent vibration did not trip");
vibration=0; repeat(4) @(negedge clk);
if(!trip) $fatal(1,"trip must latch");
ack=1; valid=1; @(negedge clk); ack=0; valid=0;
if(trip) $fatal(1,"healthy rearm failed");
repeat(21) @(negedge clk); if(!trip) $fatal(1,"watchdog did not trip");
ack=1; valid=1; logit=0; @(negedge clk);
if(!trip) $fatal(1,"fault at threshold must defeat ack");
logit=-65536; sensor_ok=0; @(negedge clk);
if(!trip) $fatal(1,"sensor error must defeat ack");
sensor_ok=1; @(negedge clk);
if(!trip) $fatal(1,"held acknowledgment must not automatically rearm");
ack=0; @(negedge clk); ack=1; @(negedge clk);
if(trip) $fatal(1,"new acknowledgment edge should rearm");
$display("PASS protection"); $finish;
end
endmodule
""",
        ["protection.sv"],
    )


def test_integrated_adc_to_ann(artifact, tmp_path):
    from dwt_ann_mav.fixed import fixed_probability

    model, mean, scale, config, _ = load_model(artifact)
    exported = export_fpga(artifact, tmp_path)
    codes = np.rint(1800 + 100 * np.sin(np.arange(config.window_size) * 0.3)).astype(int)
    samples = codes * exported["adc_gain_q16"] - exported["adc_offset_q16"]
    features = fixed_features(samples, config, already_quantized=True)
    expected = fixed_ann(features, model, mean, scale)
    probability = fixed_probability(expected)
    write_mem(tmp_path / "adc.mem", codes)
    simulate(
        tmp_path,
        f"""
module tb;
reg clk=0; always #5 clk=~clk;
reg rst=1, valid=0; reg [11:0] adc=0;
wire ready, trip, enable, done; wire signed [31:0] logit; wire [31:0] probability;
reg [31:0] codes[0:{config.window_size - 1}]; integer i;
motor_monitor dut(clk,rst,adc,valid,ready,1'b0,1'b1,1'b0,trip,enable,logit,probability,done);
initial begin
$readmemh("adc.mem",codes); repeat(3) @(negedge clk); rst=0;
for(i=0;i<{config.window_size};i=i+1) begin
 @(negedge clk); if(!ready) $fatal(1,"top not ready"); adc=codes[i]; valid=1;
end
@(negedge clk); valid=0;
wait(done); @(negedge clk);
if(logit !== 32'h{expected & 0xFFFFFFFF:08x}) $fatal(1,"top logit %h",logit);
if(probability !== 32'h{probability:08x}) $fatal(1,"top probability %h",probability);
if(!trip || enable) $fatal(1,"unacknowledged motor must stay off");
$display("PASS integrated ADC-DWT-ANN"); $finish;
end
initial begin #3000000; $fatal(1,"top timeout"); end
endmodule
""",
        ["dwt_features.sv", "ann_inference.sv", "protection.sv", "motor_monitor.sv"],
    )
