// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Reflection;
using System.Runtime.ExceptionServices;

namespace InferenceWeb.Tests;

/// <summary>Reflection over a model's private members, for tests that build a model without loading one.</summary>
internal static class TestReflection
{
    internal static void Set(Type type, object target, string field, object? value) => type.GetField(field,
        BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.SetValue(target, value);

    /// <summary>A field declared on <paramref name="target"/>'s type or any base type.</summary>
    internal static T Get<T>(object target, string field)
    {
        for (Type? t = target.GetType(); t != null; t = t.BaseType)
            if (t.GetField(field, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly) is { } f)
                return (T)f.GetValue(target)!;
        throw new MissingFieldException(target.GetType().Name, field);
    }

    internal static object? Invoke(object target, string method, object?[] args)
    {
        try { return target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(target, args); }
        catch (TargetInvocationException e) when (e.InnerException != null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }
}
