using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cpu
{
    public class ConvolutionDesc2d
    {
        public int kW;
        public int kH;
        public int dW;
        public int dH;
        public int padW;
        public int padH;

        public ConvolutionDesc2d(int kW, int kH, int dW, int dH, int padW, int padH)
        {
            this.kW = kW;
            this.kH = kH;
            this.dW = dW;
            this.dH = dH;
            this.padW = padW;
            this.padH = padH;
        }
    }

    public static class SpatialConvolutionMM
    {
        public static long[] OutputSize(long[] inputSizes, long[] weightSizes, ConvolutionDesc2d cd)
        {
            //int dimf = 1;
            int dimw = 3;
            int dimh = 2;

            long n = inputSizes[0];
            long inputWidth = inputSizes[dimw];
            long inputHeight = inputSizes[dimh];
            long nOutputPlane = weightSizes[0];

            long outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
            long outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;

            return new long[] { n, nOutputPlane, outputHeight, outputWidth };
        }

        public static long[] FInputSize(long[] inputSizes, long[] outputSizes, ConvolutionDesc2d cd)
        {
            return new long[] { inputSizes[0], cd.kW * cd.kH * inputSizes[1], outputSizes[2] * outputSizes[3] };
        }


        public static void Conv2Forward(Tensor input, Tensor output, Tensor weight, Tensor bias, Tensor finput, ConvolutionDesc2d cd)
        {
            int dimf = 1;
            int dimw = 3;
            int dimh = 2;

            long n = input.Sizes[0];
            long nInputPlane = input.Sizes[dimf];
            long inputWidth = input.Sizes[dimw];
            long inputHeight = input.Sizes[dimh];
            long nOutputPlane = weight.Sizes[0];

            long outputWidth = (inputWidth + 2 * cd.padW - cd.kW) / cd.dW + 1;
            long outputHeight = (inputHeight + 2 * cd.padH - cd.kH) / cd.dH + 1;

            if (bias != null && (bias.Sizes[0] != nOutputPlane))
            {
                throw new InvalidOperationException("bias has incorrect size. Expected 1D tensor of size " + nOutputPlane);
            }

            if (outputWidth < 1 || outputHeight < 1)
            {
                throw new InvalidOperationException(string.Format(
                    "Output size too small; calculated output size = ({0}x{1}x{2}", nOutputPlane, outputHeight, outputWidth));
            }

            if (nInputPlane * cd.kW * cd.kH != weight.Sizes[1])
            {
                throw new InvalidOperationException(
                    string.Format("Input has incorrect number of channels. Got {0}, expected {1}", nInputPlane, weight.Sizes[1] / ((float)(cd.kW * cd.kH))));
            }

            if (input.DimensionCount != 4)
            {
                throw new InvalidOperationException("4D input expected (NCHW order)");
            }

            if (finput.Sizes[0] != n || finput.Sizes[1] != cd.kW * cd.kH * nInputPlane || finput.Sizes[2] != outputHeight * outputWidth)
            {
                throw new InvalidOperationException("finput is incorrect size");
            }

            if (output.Sizes[0] != n || output.Sizes[1] != nOutputPlane || output.Sizes[2] != outputHeight || output.Sizes[3] != outputWidth)
            {
                throw new InvalidOperationException("output is incorrect size");
            }

            for (int i = 0; i < n; ++i)
            {
                using Tensor input_i = input.Select(0, i);
                using Tensor output_i = output.Select(0, i);
                using Tensor finput_i = finput.Select(0, i);
                Conv2ForwardFrame(input_i, output_i, weight, bias, finput_i,
                    cd.kW, cd.kH, cd.dW, cd.dW, cd.padW, cd.padH,
                    nInputPlane, inputWidth, inputHeight,
                    nOutputPlane, outputWidth, outputHeight);
            }
        }

        private static void Conv2ForwardFrame(Tensor input, Tensor output, Tensor weight, Tensor bias, Tensor finput,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          long nInputPlane,
          long inputWidth,
          long inputHeight,
          long nOutputPlane,
          long outputWidth,
          long outputHeight)
        {
            TensorRef64 inputRef = NativeWrapper.AllocTensorRef(input);
            TensorRef64 finputRef = NativeWrapper.AllocTensorRef(finput);

            IntPtr inputPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
            Marshal.StructureToPtr(inputRef, inputPtr, false);
            IntPtr finputPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
            Marshal.StructureToPtr(finputRef, finputPtr, false);

            try
            {
                CpuOpsNative.TS_Unfolded_Copy(finputPtr, inputPtr, kW, kH, dW, dH, padW, padH, (int)nInputPlane, (int)inputWidth, (int)inputHeight, (int)outputWidth, (int)outputHeight);

                using Tensor output2d = output.View(nOutputPlane, outputHeight * outputWidth);
                if (bias != null)
                {
                    using Tensor biasExp = bias.Expand(nOutputPlane, output2d.Sizes[1]);
                    Ops.Copy(output2d, biasExp);
                }
                else
                {
                    Ops.Fill(output, 0);
                }

                Ops.Addmm(output2d, 1, output2d, 1, weight, finput);
            }
            finally
            {
                Marshal.FreeHGlobal(inputPtr);
                Marshal.FreeHGlobal(finputPtr);
                NativeWrapper.FreeTensorRef(inputRef);
                NativeWrapper.FreeTensorRef(finputRef);
            }
        }


        public static void Conv2BackwardInput(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor weight, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            long nOutputPlane = weight.Sizes[0];

            if (gradOutput.Sizes[1] != nOutputPlane)
            {
                throw new InvalidOperationException("Number of output features must equal nOutputPlane");
            }

            if (cd.kW <= 0 && cd.kH <= 0)
            {
                throw new InvalidOperationException("Kernel size should be greater than zero");
            }

            if (cd.dW <= 0 && cd.dH <= 0)
            {
                throw new InvalidOperationException("stride should be greater than zero");
            }

            using Tensor weightT = weight.Transpose();
            long n = input.Sizes[0];

            for (int i = 0; i < n; ++i)
            {
                using Tensor gradInput_i = gradInput.Select(0, i);
                using Tensor gradOutput_i = gradOutput.Select(0, i);
                using Tensor fgradInput_i = fgradInput.Select(0, i);
                Conv2BackwardInputFrame(gradOutput_i, gradInput_i, weightT, fgradInput_i, cd);
            }
        }

        private static void Conv2BackwardInputFrame(Tensor gradOutput, Tensor gradInput, Tensor weight, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            using (Tensor gradOutput2d = gradOutput.View(gradOutput.Sizes[0], gradOutput.Sizes[1] * gradOutput.Sizes[2]))
            {
                Ops.Addmm(fgradInput, 0, fgradInput, 1, weight, gradOutput2d);
            }

            Ops.Fill(gradInput, 0);

            using (NativeWrapper.BuildTensorRefPtr(fgradInput, out IntPtr fgradInputPtr))
            using (NativeWrapper.BuildTensorRefPtr(gradInput, out IntPtr gradInputPtr))
            {
                CpuOpsNative.TS_Unfolded_Acc(fgradInputPtr, gradInputPtr, cd.kW, cd.kH, cd.dW, cd.dH, cd.padW, cd.padH,
                (int)gradInput.Sizes[0], (int)gradInput.Sizes[2], (int)gradInput.Sizes[1],
                (int)gradOutput.Sizes[2], (int)gradOutput.Sizes[1]);
            }
        }

        public static void Conv2BackwardFilter(Tensor input, Tensor gradOutput, Tensor gradWeight, Tensor gradBias, Tensor finput, Tensor fgradInput, ConvolutionDesc2d cd)
        {
            long nOutputPlane = gradWeight.Sizes[0];
            long n = input.Sizes[0];

            if (gradOutput.Sizes[1] != nOutputPlane)
            {
                throw new InvalidOperationException("Number of output features must equal nOutputPlane");
            }

            if (cd.kW <= 0 && cd.kH <= 0)
            {
                throw new InvalidOperationException("Kernel size should be greater than zero");
            }

            if (cd.dW <= 0 && cd.dH <= 0)
            {
                throw new InvalidOperationException("stride should be greater than zero");
            }

            for (int i = 0; i < n; ++i)
            {
                using Tensor gradOutput_i = gradOutput.Select(0, i);
                using Tensor finput_i = finput.Select(0, i);
                Conv2BackwardFilterFrame(gradOutput_i, gradWeight, gradBias, finput_i, cd);
            }
        }

        private static void Conv2BackwardFilterFrame(Tensor gradOutput, Tensor gradWeight, Tensor gradBias, Tensor finput, ConvolutionDesc2d cd)
        {
            if (gradOutput is null)
            {
                throw new ArgumentNullException(nameof(gradOutput));
            }

            if (gradWeight is null)
            {
                throw new ArgumentNullException(nameof(gradWeight));
            }

            if (gradBias is null)
            {
                throw new ArgumentNullException(nameof(gradBias));
            }

            if (finput is null)
            {
                throw new ArgumentNullException(nameof(finput));
            }

            if (cd is null)
            {
                throw new ArgumentNullException(nameof(cd));
            }

            using Tensor gradOutput2d = gradOutput.View(gradOutput.Sizes[0], gradOutput.Sizes[1] * gradOutput.Sizes[2]);
            using Tensor finputT = finput.Transpose();
            Ops.Addmm(gradWeight, 1, gradWeight, 1, gradOutput2d, finputT);
            Ops.Sum(gradBias, gradOutput2d, 1);
        }
    }
}
