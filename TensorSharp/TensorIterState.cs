using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorSharp
{
	public class TensorIterState
	{

		long[] sizes;
		long[] strides;


		long elementCount, stride, size;
		int dim;
		long[] counter;
		int step;

		long index;
		unsafe public float* data;


		long ElementCount(int dimCount, long[] sizes)
		{
			if (dimCount == 0)
				return 0;

			long total = 1L;
			for (int i = 0; i < dimCount; ++i)
				total *= sizes[i];
			return total;
		}


		unsafe public TensorIterState(float* buffer, int dimCount, long[] sizes, long[] strides, int step = 1)
		{
			this.sizes = sizes;
			this.strides = strides;
			this.step = step;

			index = 0;
			data = buffer;

			for (dim = dimCount - 1; dim >= 0; dim--)
			{
				if (sizes[dim] != 1)
					break;
			}

			// Get stride for dimension
			stride = (dim == -1 ? 0 : strides[dim]);


			// Find largest contiguous section
			// Note: this updates dim and size
			size = 1;
			for (dim = dimCount - 1; dim >= 0; dim--)
			{
				if (strides[dim] == size)
				{
					size *= sizes[dim];
				}
				else
				{
					break;
				}
			}

			if (size % step != 0)
			{
				throw new ArgumentException($"Size '{size}' mod step '{step}' must be zero.");
			}


			// Counter keeps track of how many iterations have been performed on each dimension
			// that is *not* part of the above contiguous block
			// Iterations are performed from highest dimension index to lowest.
			// When a complete iteration of dimension i is finished, the counter for dim i-1 gets incremented by 1
			counter = new long[dim + 1];
			for (int i = 0; i < dim + 1; ++i)
				counter[i] = 0;

			elementCount = ElementCount(dimCount, sizes);
		}


		public bool ReachedBlockEnd()
		{
			return !(index < size);
		}

		public void BlockStep()
		{
			unsafe
			{
				index += step;
				data += (stride * step);
			}
		}

		// Returns true if there is another block to iterate over,
		// returns false if we are at end of iteration
		public bool NextBlock()
		{
			unsafe
			{
				// If not at end of current block yet, do nothing
				if (index == size)
				{
					// If contiguous block encompassed all dimensions, we are done
					if (dim == -1)
						return false;

					// Reset data offset
					data -= size * stride;

					// Update counter and data for next contiguous block
					for (long j = dim; j >= 0; --j)
					{
						counter[j]++;
						data += strides[j];

						if (counter[j] == sizes[j])
						{
							if (j == 0)
							{
								return false;
							}
							else
							{
								data -= counter[j] * strides[j];
								counter[j] = 0;
							}
						}
						else
						{
							break;
						}
					}

					index = 0;
				}

				return true;
			}
		}
	}
}
