// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace TensorSharp
{
    public static class AssemblyExtensions
    {
        public static IEnumerable<Tuple<Type, IEnumerable<T>>> TypesWithAttribute<T>(this Assembly assembly, bool inherit)
        {
            foreach (Type type in assembly.GetTypes())
            {
                object[] attributes = type.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(type, attributes.Cast<T>());
                }
            }
        }
    }

    public static class TypeExtensions
    {
        public static IEnumerable<Tuple<MethodInfo, IEnumerable<T>>> MethodsWithAttribute<T>(this Type type, bool inherit)
        {
            foreach (MethodInfo method in type.GetMethods())
            {
                object[] attributes = method.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(method, attributes.Cast<T>());
                }
            }
        }
    }

    public static class MethodExtensions
    {
        public static IEnumerable<Tuple<ParameterInfo, IEnumerable<T>>> ParametersWithAttribute<T>(this MethodInfo method, bool inherit)
        {
            foreach (ParameterInfo paramter in method.GetParameters())
            {
                object[] attributes = paramter.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(paramter, attributes.Cast<T>());
                }
            }
        }
    }
}
