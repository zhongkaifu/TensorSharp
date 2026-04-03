using System;

namespace TensorSharp.Cpu
{
    public static class CpuMaxPoolingOps
    {
        public static long[] OutputSize(long[] inputSizes, bool ceilMode, ConvolutionDesc2d cd)
        {
            int dimw = 3;
            int dimh = 2;

            long iwidth = inputSizes[dimw];
            long iheight = inputSizes[dimh];

            long oheight, owidth;
            if (ceilMode)
            {
                oheight = (long)(Math.Ceiling((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Ceiling((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }
            else
            {
                oheight = (long)(Math.Floor((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Floor((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }

            return new long[] { inputSizes[0], inputSizes[1], oheight, owidth };
        }


        public static void SpatialMaxPoolingForward(Tensor input, Tensor output, Tensor indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            if (input.DimensionCount != 4)
            {
                throw new ArgumentException("input must be a 4D tensor");
            }

            int dimw = 3;
            int dimh = 2;
            int dimc = 1;

            if (input.Sizes[dimw] < cd.kW - cd.padW || input.Sizes[dimh] < cd.kH - cd.padH)
            {
                throw new InvalidOperationException("input image is smaller than kernel size");
            }

            if (cd.padW > cd.kW / 2 || cd.padH > cd.kH / 2)
            {
                throw new InvalidOperationException("pad should be smaller than half of the kernel size");
            }

            long nbatch = input.Sizes[0];
            long nslices = input.Sizes[dimc];
            long iheight = input.Sizes[dimh];
            long iwidth = input.Sizes[dimw];

            long owidth;
            long oheight;

            if (ceilMode)
            {
                oheight = (long)(Math.Ceiling((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Ceiling((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }
            else
            {
                oheight = (long)(Math.Floor((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Floor((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }

            if (cd.padW != 0 || cd.padH != 0)
            {
                // ensure that the last pooling starts inside the image
                if ((oheight - 1) * cd.dH >= iheight + cd.padH)
                {
                    --oheight;
                }

                if ((owidth - 1) * cd.dW >= iwidth + cd.padW)
                {
                    --owidth;
                }
            }

            using Tensor inputContig = Ops.AsContiguous(input);
            for (int i = 0; i < nbatch; ++i)
            {
                using Tensor input_i = inputContig.Select(0, i);
                using Tensor output_i = output.Select(0, i);
                using Tensor indices_i = indices.Select(0, i);
                using (NativeWrapper.BuildTensorRefPtr(input_i, out IntPtr input_iPtr))
                using (NativeWrapper.BuildTensorRefPtr(output_i, out IntPtr output_iPtr))
                using (NativeWrapper.BuildTensorRefPtr(indices_i, out IntPtr indices_iPtr))
                {
                    CpuOpsNative.TS_SpatialMaxPooling_updateOutput_frame(input_iPtr, output_iPtr, indices_iPtr,
                        nslices, iwidth, iheight,
                        owidth, oheight,
                        cd.kW, cd.kH, cd.dW, cd.dH, cd.padW, cd.padH);
                }
            }

        }


        public static void SpatialMaxPoolingBackward(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            int dimw = 3;
            int dimh = 2;
            int dimc = 1;

            long nbatch = input.Sizes[0];
            long nslices = input.Sizes[dimc];
            long iheight = input.Sizes[dimh];
            long iwidth = input.Sizes[dimw];
            long owidth = gradOutput.Sizes[dimw];
            long oheight = gradOutput.Sizes[dimh];

            Ops.Fill(gradInput, 0);


            using Tensor gradOutputContig = Ops.AsContiguous(gradOutput);
            for (int i = 0; i < nbatch; ++i)
            {
                using Tensor gradInput_i = gradInput.Select(0, i);
                using Tensor gradOutput_i = gradOutputContig.Select(0, i);
                using Tensor indices_i = indices.Select(0, i);
                using (NativeWrapper.BuildTensorRefPtr(gradInput_i, out IntPtr gradInput_iPtr))
                using (NativeWrapper.BuildTensorRefPtr(gradOutput_i, out IntPtr gradOutput_iPtr))
                using (NativeWrapper.BuildTensorRefPtr(indices_i, out IntPtr indices_iPtr))
                {
                    CpuOpsNative.TS_SpatialMaxPooling_updateGradInput_frame(gradInput_iPtr, gradOutput_iPtr, indices_iPtr,
                        nslices, iwidth, iheight,
                        owidth, oheight,
                        cd.dW, cd.dH);
                }
            }
        }
    }
}
