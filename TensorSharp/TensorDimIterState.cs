using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorSharp
{
    public class TensorDimIterState
    {
		long[] sizes;
		long[] strides;
		int dimensionCount;
		int iterationDim;
		long[] counter;


	public long stride, size;
		unsafe public float* data;



		unsafe public TensorDimIterState(float* buffer, int dimCount, long[] sizes, long[] strides, int iterationDim)
		{
			this.sizes = sizes;
			this.strides = strides;
			this.iterationDim = iterationDim;
			this.dimensionCount = dimCount;

			data = buffer;

			this.size = sizes[iterationDim];
			this.stride = strides[iterationDim];


			counter = new long[dimCount];
			for (int i = 0; i < dimCount; ++i)
				counter[i] = 0;
		}


		// Returns true if there is another block to iterate over,
		// returns false if we are at end of iteration
		unsafe public bool NextBlock()
		{
			if (dimensionCount == 1)
			{
				return false;
			}

			for (int i = 0; i < dimensionCount; ++i)
			{
				if (i == iterationDim)
				{
					if (i == dimensionCount - 1)
					{
						return false;
					}
					continue;
				}

				counter[i]++;
				data += strides[i];

				if (counter[i] == sizes[i])
				{
					if (i == dimensionCount - 1)
					{
						return false;
					}
					else
					{
						data -= counter[i] * strides[i];
						counter[i] = 0;
					}
				}
				else
				{
					break;
				}
			}

			return true;
		}
	}
}
