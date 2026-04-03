using AdvUtils;
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
