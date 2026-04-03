using System;
using System.Linq;

namespace TensorSharp.Core
{
    public static class TensorConcatenation
    {
        // Requires an implementation of *copy* for the given tensor types
        public static Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            if (inputs.Length < 2)
            {
                throw new ArgumentException("Concat: at least two tensors required", "inputs");
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] == null)
                {
                    throw new ArgumentException($"Concat: input[{i}] is null.");
                }
            }


            int ndim = Math.Max(dimension, inputs.Max(x => x.DimensionCount));
            long[] size = ConcatTensorSize(ndim, dimension, inputs);

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, inputs[0], true, size);


            // Select each region of the result corresponding to each input tensor,
            // and copy into the result
            long offset = 0;
            for (int j = 0; j < inputs.Length; ++j)
            {
                long dimSize = GetDimSize(inputs[j], dimension);
                using (Tensor nt = writeTarget.Narrow(dimension, offset, dimSize))
                {
                    Ops.Copy(nt, inputs[j]);
                }
                offset += dimSize;
            }

            return writeTarget;
        }



        private static long GetDimSize(Tensor tensor, int dim)
        {
            return dim < tensor.DimensionCount ? tensor.Sizes[dim] : 1;
        }

        private static long[] ConcatTensorSize(int ndim, int dimension, Tensor[] tensors)
        {
            long[] result = new long[ndim];
            for (int i = 0; i < ndim; ++i)
            {
                long dimSize = GetDimSize(tensors[0], i);
                if (i == dimension)
                {
                    for (int j = 1; j < tensors.Length; ++j)
                    {
                        dimSize += GetDimSize(tensors[j], i);
                    }
                }
                else
                {
                    for (int j = 1; j < tensors.Length; ++j)
                    {
                        if (dimSize != GetDimSize(tensors[j], i))
                        {
                            string message = "";
                            for (int k = 0; k < tensors.Length; k++)
                            {
                                message += $"{k}: ({tensors[k].Sizes[0]}, {tensors[k].Sizes[1]}) ";
                            }
                            throw new InvalidOperationException($"Inconsistent tensor sizes. {message}");
                        }
                    }
                }
                result[i] = dimSize;
            }
            return result;
        }

    }
}
