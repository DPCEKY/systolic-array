# Yolo Detector Tutorial

This tutorial goes over how to write, build and run the software (C/C++) and Hardware (SDSoC) yolo detector Application.
The design is revised after previous YOLO designed, with systolic array structure implemented, though the performance is not satisfying currently. It can be improved by rearranging buffers on-chip. It is strongly suggested to go over the previous design referred by the current one, before any change is committed.

## Designing an application

This section is a general overview of how to write an application.

### Main
[main](yolo_detector_test.cpp) interfaces with the top-level [gemm2] to be instantiated on the FPGA.

## Software

The software emulation runs the hardware [gemm2] on the host CPU. This is useful 
for functional verification. The design is complied using gcc/g++ and uses a
pure software flow.


## Hardware

The hardware design can be built by SDSoC. First of all, SDSoC will call Vivado HLS to synthesize the hardware [gemm2] into RTL.
Then SDSoC will create datamover and wrap up the whole design. This design currently runs well on SDSoC (Vivado) 2017.1.
For more details of using SDSoC, please refer to UG1028: SDSoC Environment User Guide https://forums.xilinx.com/xlnx/attachments/xlnx/sdsoc/23/2/ug1028-sdsoc-getting-started.pdf

