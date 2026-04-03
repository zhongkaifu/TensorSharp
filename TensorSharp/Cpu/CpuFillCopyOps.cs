// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using System;
using System.Reflection;

namespace TensorSharp.Cpu
{
    [OpsClass]
    public class CpuFillCopyOps
    {
        public CpuFillCopyOps()
        {
        }

        [RegisterOpStorageType("fill", typeof(CpuStorage))]
        public void Fill(Tensor result, float value)
        {
            TensorApplyCPU.Fill(result, value);
        }


        [RegisterOpStorageType("copy", typeof(CpuStorage))]
        public void Copy(Tensor result, Tensor src)
        {
            try
            {
                var resEC = result.ElementCount();
                var srcEC = src.ElementCount();
                if (resEC != srcEC)
                {
                    throw new InvalidOperationException($"Tensors must have equal numbers of elements. result element count = '{resEC}', source element count = '{srcEC}', result tensor = '{result.ToString()}', source tensor = '{src.ToString()}'");
                }

                TensorApplyCPU.Copy(result, src);
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Failed to run Copy operation on CPU. Message = '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call stack = '{err.StackTrace}'");
                throw;
            }
        }

    }
}
