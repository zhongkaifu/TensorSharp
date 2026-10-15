// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class ReclaimQueueFailureTests
{
    [Fact]
    public void FailedPartialReleaseKeepsOwnershipAndBytesUntilRetryConfirmsRelease()
    {
        var queue = new ReclaimQueue();
        queue.Enqueue("first", ReleaseReason.Pressure, new ResourceVector { DeviceKv = 64 });
        queue.Enqueue("second", ReleaseReason.Evicted, new ResourceVector { HostKv = 32 });
        var live = new HashSet<string> { "first", "second" };
        Assert.Throws<IOException>(() => queue.Drain((keys, reason) =>
        {
            Assert.Equal(ReleaseReason.Pressure, reason);
            live.Remove(keys[0]);
            throw new IOException("Second native payload release failed");
        }));
        Assert.Equal(2, queue.Count);
        Assert.Equal(0, queue.ReleasedKeys);
        Assert.True(queue.Contains("first") && queue.Contains("second"));
        Assert.Equal(64, queue.PendingReclaim.DeviceKv);
        Assert.Equal(32, queue.PendingReclaim.HostKv);
        Assert.False(queue.Enqueue("second", ReleaseReason.Pressure, new ResourceVector { HostKv = 32 }));
        Assert.Equal(32, queue.PendingReclaim.HostKv);

        Assert.Equal(2, queue.Drain((keys, reason) =>
        {
            Assert.Equal(ReleaseReason.Pressure, reason);
            foreach (string key in keys) live.Remove(key); // Idempotent model contract.
        }));
        Assert.Empty(live);
        Assert.Equal(0, queue.Count);
        Assert.Equal(2, queue.ReleasedKeys);
        Assert.True(queue.PendingReclaim.IsZero);
    }
}
