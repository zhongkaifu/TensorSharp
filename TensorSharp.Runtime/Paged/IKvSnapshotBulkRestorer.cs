// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;

namespace TensorSharp.Runtime.Paged;

/// <summary>Optional serial restoration of a contiguous prefix starting at zero.
/// The model starts empty. Acquire returns one scoped read lease by block index;
/// implementations must release it before acquiring another, including on failure.
/// Returns the accepted prefix length with ALL model state valid at that endpoint.
/// A refused block preserves the preceding accepted state. An exception invalidates
/// the restore and the caller must reset before further execution. No Forward or
/// external observer may run during restoration. Intermediate recurrent copies may
/// therefore be deferred, but a successful/short return must materialize the endpoint.</summary>
public interface IKvSnapshotBulkRestorer
{
    int RestoreKvSnapshots(int blockTokens, int tokens, Func<int, KvSnapshotLease> acquire);
}
