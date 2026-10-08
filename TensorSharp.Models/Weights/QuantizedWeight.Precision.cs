// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public partial class QuantizedWeight
{
    private bool _q8F32Activations;
    private HashSet<IntPtr> _q8F32Keys;

    internal void EnableQ8F32Activations()
    {
        if (_q8F32Activations || GgmlType != (int)GgmlTensorType.Q8_0) return;
        _q8F32Activations = true;
        try
        {
            RegisterQ8F32Key(Data);
            RegisterQ8F32Key(CacheKey);
        }
        catch
        {
            DisableQ8F32Activations();
            throw;
        }
    }

    internal void DisableQ8F32Activations()
    {
        if (_q8F32Keys != null)
        {
            // Native unregister is a no-throw, reference-counted operation.
            // Remove a key only after it has been retired by the native owner.
            while (_q8F32Keys.Count != 0)
            {
                var keys = _q8F32Keys.GetEnumerator();
                keys.MoveNext();
                UnregisterQ8F32Key(keys.Current);
            }
        }
        _q8F32Activations = false;
    }

    private void RegisterQ8F32Key(IntPtr key)
    {
        if (!_q8F32Activations || key == IntPtr.Zero) return;
        _q8F32Keys ??= new();
        if (!_q8F32Keys.Add(key)) return;
        try { GgmlQ8Precision.RegisterWeight(key); }
        catch { _q8F32Keys.Remove(key); throw; }
    }

    private void UnregisterQ8F32Key(IntPtr key)
    {
        if (_q8F32Keys?.Contains(key) != true) return;
        GgmlQ8Precision.UnregisterWeight(key);
        _q8F32Keys.Remove(key);
    }
}
