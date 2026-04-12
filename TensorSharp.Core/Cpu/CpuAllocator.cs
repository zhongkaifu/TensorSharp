// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using AdvUtils;
using System;

namespace TensorSharp.Cpu
{
    public class CpuAllocator : IAllocator
    {
        private BlasEnum m_blasEnum;
        public BlasEnum BlasEnum => m_blasEnum;
        public int DeviceId => 0;

        public CpuAllocator(BlasEnum blasEnum, string mklInstructions = "AVX2")
        {
            m_blasEnum = blasEnum;
            if (m_blasEnum == BlasEnum.MKL)
            {
                Logger.WriteLine($"MKL Instrucation = '{mklInstructions}'");
                Environment.SetEnvironmentVariable("MKL_ENABLE_INSTRUCTIONS", mklInstructions);
            }
        }

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CpuStorage(this, elementType, elementCount);
        }

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }
    }
}
