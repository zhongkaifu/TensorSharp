using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using TensorSharp.Core;

namespace TensorSharp.Cpu
{
    public static class NativeWrapper
    {
        public static MethodInfo GetMethod(string name)
        {
            return typeof(CpuOpsNative).GetMethod(name, BindingFlags.Public | BindingFlags.Static);
        }

        public static Tensor InvokeNullableResultElementwise(MethodInfo method, params object[] args)
        {
            Tensor resultTensor;
            if (args[0] == null)
            {
                Tensor otherTensor = args.OfType<Tensor>().First();
                resultTensor = TensorResultBuilder.GetWriteTarget(null, otherTensor, false, otherTensor.Sizes);
            }
            else
            {
                Tensor resultSrc = (Tensor)args[0];
                Tensor otherTensor = args.OfType<Tensor>().Skip(1).First();
                resultTensor = TensorResultBuilder.GetWriteTarget(resultSrc, otherTensor, false, otherTensor.Sizes);
            }

            args[0] = resultTensor;
            InvokeTypeMatch(method, args);
            return resultTensor;
        }

        public static Tensor InvokeNullableResultDimensionwise(MethodInfo method, Tensor result, Tensor src, int dimension, params object[] extraArgs)
        {
            if (dimension < 0 || dimension >= src.Sizes.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(dimension));
            }

            long[] desiredSize = (long[])src.Sizes.Clone();
            desiredSize[dimension] = 1;
            Tensor resultTensor = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);

            List<object> finalArgs = new List<object>(extraArgs.Length + 3)
            {
                resultTensor,
                src,
                dimension
            };
            finalArgs.AddRange(extraArgs);
            InvokeTypeMatch(method, finalArgs.ToArray());
            return resultTensor;
        }


        public static Tensor CreateResultDimensionwise(Tensor result, Tensor src, int dimension)
        {
            if (dimension < 0 || dimension >= src.Sizes.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(dimension));
            }

            long[] desiredSize = (long[])src.Sizes.Clone();
            desiredSize[dimension] = 1;
            Tensor resultTensor = TensorResultBuilder.GetWriteTarget(result, src, false, desiredSize);

            return resultTensor;
        }

        public static void InvokeTypeMatch(MethodInfo method, params object[] args)
        {
            IEnumerable<Tensor> tensors = args.OfType<Tensor>();
            if (tensors.Any())
            {
                DType elemType = tensors.First().ElementType;
                if (!tensors.All(x => x.ElementType == elemType))
                {
                    string allTypes = string.Join(", ", tensors.Select(x => x.ElementType));
                    throw new InvalidOperationException("All tensors must have the same argument types. Given: " + allTypes);
                }
            }

            Invoke(method, args);
        }


        public static IDisposable BuildTensorRefPtr(Tensor tensor, out IntPtr tensorRefPtr)
        {
            TensorRef64 tensorRef = NativeWrapper.AllocTensorRef(tensor);
            IntPtr tensorPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
            Marshal.StructureToPtr(tensorRef, tensorPtr, false);

            tensorRefPtr = tensorPtr;

            return new DelegateDisposable(() =>
            {
                Marshal.FreeHGlobal(tensorPtr);
                NativeWrapper.FreeTensorRef(tensorRef);
            });
        }

        public static void Invoke(MethodInfo method, params object[] args)
        {
            List<TensorRef64> freeListTensor = new List<TensorRef64>();
            List<IntPtr> freeListPtr = new List<IntPtr>();

            try
            {
                for (int i = 0; i < args.Length; ++i)
                {
                    if (args[i] is Tensor tensor)
                    {
                        if (!(tensor.Storage is CpuStorage))
                        {
                            throw new InvalidOperationException("Argument " + i + " is not a Cpu tensor");
                        }

                        TensorRef64 tensorRef = AllocTensorRef(tensor);
                        IntPtr tensorPtr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(TensorRef64)));
                        Marshal.StructureToPtr(tensorRef, tensorPtr, false);

                        args[i] = tensorPtr;

                        freeListTensor.Add(tensorRef);
                        freeListPtr.Add(tensorPtr);
                    }
                }

                //return method.Invoke(null, args);
                int result = (int)method.Invoke(null, args);
                if (result != 0)
                {
                    throw new ApplicationException(GetLastError());
                }
            }
            finally
            {
                foreach (TensorRef64 tensorRef in freeListTensor)
                {
                    FreeTensorRef(tensorRef);
                }

                foreach (IntPtr tensorPtr in freeListPtr)
                {
                    Marshal.FreeHGlobal(tensorPtr);
                }
            }
        }

        public static void CheckResult(int result)
        {
            if (result != 0)
            {
                throw new ApplicationException(GetLastError());
            }
        }

        private static string GetLastError()
        {
            IntPtr strPtr = CpuOpsNative.TS_GetLastError();
            return Marshal.PtrToStringAnsi(strPtr);
        }


        public static TensorRef64 AllocTensorRef(Tensor tensor)
        {
            TensorRef64 tensorRef = new TensorRef64
            {
                buffer = CpuNativeHelpers.GetBufferStart(tensor),
                dimCount = tensor.Sizes.Length,
                sizes = AllocArray(tensor.Sizes),
                strides = AllocArray(tensor.Strides),
                elementType = (CpuDType)tensor.ElementType
            };
            return tensorRef;
        }

        private static IntPtr AllocArray(long[] data)
        {
            IntPtr result = Marshal.AllocHGlobal(sizeof(long) * data.Length);
            Marshal.Copy(data, 0, result, data.Length);
            return result;
        }

        public static void FreeTensorRef(TensorRef64 tensorRef)
        {
            Marshal.FreeHGlobal(tensorRef.sizes);
            Marshal.FreeHGlobal(tensorRef.strides);
        }
    }
}
