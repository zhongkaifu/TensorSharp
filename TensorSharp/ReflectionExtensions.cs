using System;
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
